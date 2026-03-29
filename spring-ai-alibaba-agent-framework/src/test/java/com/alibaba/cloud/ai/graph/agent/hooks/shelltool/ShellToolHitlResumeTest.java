/*
 * Copyright 2024-2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.alibaba.cloud.ai.graph.agent.hooks.shelltool;

import com.alibaba.cloud.ai.graph.NodeOutput;
import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.agent.ReactAgent;
import com.alibaba.cloud.ai.graph.agent.hook.hip.HumanInTheLoopHook;
import com.alibaba.cloud.ai.graph.agent.hook.shelltool.ShellToolAgentHook;
import com.alibaba.cloud.ai.graph.agent.hook.hip.ToolConfig;
import com.alibaba.cloud.ai.graph.agent.tools.ShellSessionManager;
import com.alibaba.cloud.ai.graph.agent.tools.ShellTool2;
import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ShellToolHitlResumeTest {

	@Test
	void testHitlApprovedShellCallResumesSuccessfully(@TempDir Path workspace) throws Exception {
		ShellTool2 shellTool2 = ShellTool2.builder(workspace.toString()).build();
		ShellToolAgentHook shellToolAgentHook = ShellToolAgentHook.builder().shellTool2(shellTool2).build();
		ReactAgent agent = createAgent("shell-hitl-approved", new SingleShellApprovalChatModel(), shellToolAgentHook, shellTool2);

		String threadId = "shell-hitl-approved-thread";
		InterruptionMetadata interruptionMetadata = firstInvokeExpectInterruption(agent, threadId, "执行 shell 命令");
		assertEquals(1, trackedSessionCount(shellTool2.getSessionManager()), "shell session should be tracked while waiting for approval");

		NodeOutput finalOutput = resumeWithApproval(agent, threadId, interruptionMetadata);
		assertFalse(finalOutput instanceof InterruptionMetadata, "approved HITL flow should complete without another interruption");
		assertTrue(lastAssistantText(finalOutput).contains(workspace.toString()),
				"final assistant message should contain the shell output from the resumed command");
		assertEquals(0, trackedSessionCount(shellTool2.getSessionManager()), "shell session should be cleaned up after agent completion");
	}

	@Test
	void testHitlResumeReusesSameShellSession(@TempDir Path workspace) throws Exception {
		Path nestedDir = Files.createDirectories(workspace.resolve("nested"));
		ShellTool2 shellTool2 = ShellTool2.builder(workspace.toString()).build();
		ShellToolAgentHook shellToolAgentHook = ShellToolAgentHook.builder().shellTool2(shellTool2).build();
		ReactAgent agent = createAgent("shell-hitl-reuse",
				new PersistentSessionChatModel(nestedDir),
				shellToolAgentHook,
				shellTool2);

		String threadId = "shell-hitl-reuse-thread";
		InterruptionMetadata firstInterruption = firstInvokeExpectInterruption(agent, threadId, "先切目录再确认");
		assertEquals(1, trackedSessionCount(shellTool2.getSessionManager()), "session should survive the first interruption");

		NodeOutput secondOutput = resumeWithApproval(agent, threadId, firstInterruption);
		InterruptionMetadata secondInterruption = assertInstanceOf(InterruptionMetadata.class, secondOutput,
				"second model step should ask for approval again before executing the second shell command");
		assertEquals(1, trackedSessionCount(shellTool2.getSessionManager()), "session should still be tracked between resume rounds");

		NodeOutput finalOutput = resumeWithApproval(agent, threadId, secondInterruption);
		assertTrue(lastAssistantText(finalOutput).contains(nestedDir.toString()),
				"second resumed shell command should observe the directory change from the first shell command");
		assertEquals(0, trackedSessionCount(shellTool2.getSessionManager()), "session should be cleaned after the multi-round flow completes");
	}

	@Test
	void testCleanupRemovesThreadScopedSession(@TempDir Path workspace) throws Exception {
		ShellSessionManager sessionManager = ShellSessionManager.builder().workspaceRoot(workspace).build();

		RunnableConfig initialConfig = RunnableConfig.builder().threadId("cleanup-thread").build();
		sessionManager.initialize(initialConfig);
		assertEquals(1, trackedSessionCount(sessionManager), "initialization should register one tracked shell session");

		RunnableConfig cleanupConfig = RunnableConfig.builder().threadId("cleanup-thread").build();
		sessionManager.cleanup(cleanupConfig);
		assertEquals(0, trackedSessionCount(sessionManager), "cleanup should remove tracked sessions even when the current config has no runtime context");
	}

	private ReactAgent createAgent(String name, ChatModel chatModel, ShellToolAgentHook shellToolAgentHook, ShellTool2 shellTool2) {
		return ReactAgent.builder()
				.name(name)
				.model(chatModel)
				.saver(new MemorySaver())
				.methodTools(shellTool2)
				.hooks(List.of(
						shellToolAgentHook,
						HumanInTheLoopHook.builder().approvalOn(Map.of(
								"shell", ToolConfig.builder().description("执行 shell 命令需要确认").build()))
							.build()))
				.build();
	}

	private InterruptionMetadata firstInvokeExpectInterruption(ReactAgent agent, String threadId, String input) throws Exception {
		Optional<NodeOutput> result = agent.invokeAndGetOutput(input, RunnableConfig.builder().threadId(threadId).build());
		assertTrue(result.isPresent(), "first invocation should produce a result");
		return assertInstanceOf(InterruptionMetadata.class, result.get(),
				"first invocation should pause for human approval");
	}

	private NodeOutput resumeWithApproval(ReactAgent agent, String threadId, InterruptionMetadata interruptionMetadata) throws Exception {
		RunnableConfig resumeConfig = RunnableConfig.builder()
				.threadId(threadId)
				.addMetadata(RunnableConfig.HUMAN_FEEDBACK_METADATA_KEY, buildApprovalFeedback(interruptionMetadata))
				.build();

		Optional<NodeOutput> result = agent.invokeAndGetOutput("", resumeConfig);
		assertTrue(result.isPresent(), "resume invocation should produce a result");
		return result.get();
	}

	private InterruptionMetadata buildApprovalFeedback(InterruptionMetadata interruptionMetadata) {
		InterruptionMetadata.Builder builder = InterruptionMetadata.builder()
				.nodeId(interruptionMetadata.node())
				.state(interruptionMetadata.state());

		interruptionMetadata.toolFeedbacks().forEach(toolFeedback -> builder.addToolFeedback(
				InterruptionMetadata.ToolFeedback.builder(toolFeedback)
						.result(InterruptionMetadata.ToolFeedback.FeedbackResult.APPROVED)
						.build()));

		return builder.build();
	}

	private String lastAssistantText(NodeOutput nodeOutput) {
		@SuppressWarnings("unchecked")
		List<Message> messages = (List<Message>) nodeOutput.state().value("messages").orElse(List.of());
		assertFalse(messages.isEmpty(), "final state should contain messages");
		Message lastMessage = messages.get(messages.size() - 1);
		AssistantMessage assistantMessage = assertInstanceOf(AssistantMessage.class, lastMessage,
				"last message should be an assistant response");
		assertNotNull(assistantMessage.getText(), "assistant text should not be null");
		return assistantMessage.getText();
	}

	@SuppressWarnings("unchecked")
	private int trackedSessionCount(ShellSessionManager sessionManager) throws Exception {
		Field trackedSessionsField = ShellSessionManager.class.getDeclaredField("trackedSessions");
		trackedSessionsField.setAccessible(true);
		return ((ConcurrentMap<String, ?>) trackedSessionsField.get(sessionManager)).size();
	}

	private static final class SingleShellApprovalChatModel implements ChatModel {

		private final AtomicInteger callCount = new AtomicInteger();

		@Override
		public ChatResponse call(Prompt prompt) {
			int currentCall = callCount.incrementAndGet();
			if (currentCall == 1) {
				AssistantMessage.ToolCall toolCall = new AssistantMessage.ToolCall("call-1", "function", "shell",
						"{\"command\":\"pwd\"}");
				AssistantMessage assistantMessage = AssistantMessage.builder()
						.content("")
						.toolCalls(List.of(toolCall))
						.build();
				return new ChatResponse(List.of(new Generation(assistantMessage)));
			}

			ToolResponseMessage toolResponse = lastToolResponse(prompt);
			String responseData = toolResponse.getResponses().get(0).responseData();
			System.out.println("[spring-ai-alibaba-repro] resumed shell output: " + responseData);
			return new ChatResponse(List.of(new Generation(new AssistantMessage("shell output: " + responseData))));
		}

		@Override
		public Flux<ChatResponse> stream(Prompt prompt) {
			return Flux.just(call(prompt));
		}
	}

	private static final class PersistentSessionChatModel implements ChatModel {

		private final Path nestedDir;
		private final AtomicInteger callCount = new AtomicInteger();

		private PersistentSessionChatModel(Path nestedDir) {
			this.nestedDir = nestedDir;
		}

		@Override
		public ChatResponse call(Prompt prompt) {
			int currentCall = callCount.incrementAndGet();
			if (currentCall == 1) {
				AssistantMessage.ToolCall toolCall = new AssistantMessage.ToolCall("call-1", "function", "shell",
						"{\"command\":\"cd " + escapeForJson(doubleQuotedPath(nestedDir)) + "\"}");
				AssistantMessage assistantMessage = AssistantMessage.builder()
						.content("")
						.toolCalls(List.of(toolCall))
						.build();
				return new ChatResponse(List.of(new Generation(assistantMessage)));
			}

			if (currentCall == 2) {
				lastToolResponse(prompt);
				AssistantMessage.ToolCall toolCall = new AssistantMessage.ToolCall("call-2", "function", "shell",
						"{\"command\":\"pwd\"}");
				AssistantMessage assistantMessage = AssistantMessage.builder()
						.content("")
						.toolCalls(List.of(toolCall))
						.build();
				return new ChatResponse(List.of(new Generation(assistantMessage)));
			}

			ToolResponseMessage toolResponse = lastToolResponse(prompt);
			String responseData = toolResponse.getResponses().get(0).responseData();
			System.out.println("[spring-ai-alibaba-repro] resumed shell reused session output: " + responseData);
			return new ChatResponse(List.of(new Generation(new AssistantMessage("final shell output: " + responseData))));
		}

		@Override
		public Flux<ChatResponse> stream(Prompt prompt) {
			return Flux.just(call(prompt));
		}
	}

	private static ToolResponseMessage lastToolResponse(Prompt prompt) {
		List<Message> instructions = prompt.getInstructions();
		Message lastMessage = instructions.get(instructions.size() - 1);
		return assertInstanceOf(ToolResponseMessage.class, lastMessage,
				"last prompt instruction should be the latest tool response");
	}

	private static String doubleQuotedPath(Path path) {
		return "\"" + path.toString().replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
	}

	private static String escapeForJson(String content) {
		return content.replace("\\", "\\\\").replace("\"", "\\\"");
	}

}
