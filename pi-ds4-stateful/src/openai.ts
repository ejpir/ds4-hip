import type {
	Api,
	AssistantMessage,
	Context,
	Model,
	SimpleStreamOptions,
	TextContent,
	ThinkingContent,
	ToolCall,
} from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type { JsonObject } from "./types.ts";

export function isTextBlock(block: unknown): block is TextContent {
	return !!block && typeof block === "object" && (block as { type?: unknown }).type === "text";
}

function isThinkingBlock(block: unknown): block is ThinkingContent {
	return !!block && typeof block === "object" && (block as { type?: unknown }).type === "thinking";
}

function isToolCallBlock(block: unknown): block is ToolCall {
	return !!block && typeof block === "object" && (block as { type?: unknown }).type === "toolCall";
}

export function textOfContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter(isTextBlock)
		.map((block) => block.text)
		.join("\n");
}

export function assistantToOpenAI(message: AssistantMessage): JsonObject {
	const content = Array.isArray(message.content) ? message.content : [];
	const text = content.filter(isTextBlock).map((b) => b.text).join("");
	const thinkingBlocks = content.filter(isThinkingBlock).filter((b) => b.thinking.trim().length > 0);
	const toolCalls = content.filter(isToolCallBlock);

	const out: JsonObject = {
		role: "assistant",
		content: text.length > 0 ? text : toolCalls.length > 0 ? null : "",
	};

	if (thinkingBlocks.length > 0) {
		const signature = thinkingBlocks[0].thinkingSignature || "reasoning_content";
		out[signature] = thinkingBlocks.map((b) => b.thinking).join("\n");
	} else {
		// DS4's OpenAI-compatible provider config requires reasoning_content on
		// assistant messages so future full payload hashes match pi-ai's replay.
		out.reasoning_content = "";
	}

	if (toolCalls.length > 0) {
		out.tool_calls = toolCalls.map((tc) => ({
			id: tc.id,
			type: "function",
			function: {
				name: tc.name,
				arguments: JSON.stringify(tc.arguments ?? {}),
			},
		}));
	}

	return out;
}

function userContent(content: unknown): unknown {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content.map((item) => {
		if (isTextBlock(item)) return { type: "text", text: item.text };
		const image = item as { type?: string; mimeType?: string; data?: string };
		if (image?.type === "image") {
			return { type: "image_url", image_url: { url: `data:${image.mimeType};base64,${image.data}` } };
		}
		return { type: "text", text: "" };
	});
}

function contextMessagesToOpenAI(context: Context): unknown[] {
	const messages: unknown[] = [];
	if (context.systemPrompt) messages.push({ role: "system", content: context.systemPrompt });
	for (const msg of context.messages) {
		if (msg.role === "user") {
			messages.push({ role: "user", content: userContent(msg.content) });
		} else if (msg.role === "assistant") {
			messages.push(assistantToOpenAI(msg));
		} else if (msg.role === "toolResult") {
			messages.push({
				role: "tool",
				content: textOfContent(msg.content),
				tool_call_id: msg.toolCallId,
				...(msg.toolName ? { name: msg.toolName } : {}),
			});
		}
	}
	return messages;
}

function toolsToOpenAI(context: Context): unknown[] | undefined {
	if (!context.tools || context.tools.length === 0) return undefined;
	return context.tools.map((tool) => ({
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
			strict: false,
		},
	}));
}

function mappedReasoning(model: Model<Api>, reasoning: SimpleStreamOptions["reasoning"]): string | undefined {
	if (!reasoning) return undefined;
	const mapped = model.thinkingLevelMap?.[reasoning];
	return typeof mapped === "string" ? mapped : reasoning;
}

export function buildOpenAIPayload(model: Model<Api>, context: Context, options?: SimpleStreamOptions): JsonObject {
	const reasoning = mappedReasoning(model, options?.reasoning);
	const payload: JsonObject = {
		model: model.id,
		messages: contextMessagesToOpenAI(context),
		stream: true,
		stream_options: { include_usage: true },
		thinking: { type: reasoning ? "enabled" : "disabled" },
	};
	if (reasoning) payload.reasoning_effort = reasoning;
	if (options?.maxTokens) payload.max_tokens = options.maxTokens;
	if (options?.temperature !== undefined) payload.temperature = options.temperature;
	const tools = toolsToOpenAI(context);
	if (tools) payload.tools = tools;
	return payload;
}

export function emptyUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function parseUsage(raw: any) {
	const prompt = raw?.prompt_tokens ?? 0;
	const completion = raw?.completion_tokens ?? 0;
	const cacheRead = raw?.prompt_tokens_details?.cached_tokens ?? raw?.prompt_cache_hit_tokens ?? 0;
	const cacheWrite = raw?.prompt_tokens_details?.cache_write_tokens ?? 0;
	return {
		input: Math.max(0, prompt - cacheRead - cacheWrite),
		output: completion,
		cacheRead,
		cacheWrite,
		totalTokens: Math.max(0, prompt - cacheRead - cacheWrite) + completion + cacheRead + cacheWrite,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function stopReason(reason: unknown): "stop" | "length" | "toolUse" | "error" {
	if (reason === "length") return "length";
	if (reason === "tool_calls" || reason === "function_call") return "toolUse";
	if (reason === "content_filter") return "error";
	return "stop";
}

function safeJson(text: string): Record<string, unknown> {
	try {
		return JSON.parse(text || "{}");
	} catch {
		return {};
	}
}

async function* sseData(response: Response): AsyncGenerator<string> {
	if (!response.body) return;
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	while (true) {
		const { value, done } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
		let idx = buffer.indexOf("\n\n");
		while (idx >= 0) {
			const raw = buffer.slice(0, idx);
			buffer = buffer.slice(idx + 2);
			const data = raw
				.split("\n")
				.filter((line) => line.startsWith("data:"))
				.map((line) => line.slice(5).trimStart())
				.join("\n");
			if (data) yield data;
			idx = buffer.indexOf("\n\n");
		}
	}
}

export async function streamFetchOnce(
	model: Model<Api>,
	payload: unknown,
	apiKey: string,
	options: SimpleStreamOptions | undefined,
	onResponse: SimpleStreamOptions["onResponse"],
) {
	const stream = createAssistantMessageEventStream();
	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: emptyUsage(),
			stopReason: "stop",
			timestamp: Date.now(),
		};
		try {
			const url = `${model.baseUrl.replace(/\/+$/, "")}/chat/completions`;
			const response = await fetch(url, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${apiKey}`,
					...(options?.headers ?? {}),
				},
				body: JSON.stringify(payload),
				signal: options?.signal,
			});
			await onResponse?.({ status: response.status, headers: Object.fromEntries(response.headers.entries()) }, model);
			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${await response.text()}`);
			}

			stream.push({ type: "start", partial: output });
			const blocks = output.content;
			let textBlock: TextContent | undefined;
			let thinkingBlock: ThinkingContent | undefined;
			const toolByIndex = new Map<number, ToolCall & { partialArgs?: string }>();
			const toolById = new Map<string, ToolCall & { partialArgs?: string }>();
			const indexOf = (block: TextContent | ThinkingContent | ToolCall) => blocks.indexOf(block as any);
			const ensureText = () => {
				if (!textBlock) {
					textBlock = { type: "text", text: "" };
					blocks.push(textBlock);
					stream.push({ type: "text_start", contentIndex: indexOf(textBlock), partial: output });
				}
				return textBlock;
			};
			const ensureThinking = () => {
				if (!thinkingBlock) {
					thinkingBlock = { type: "thinking", thinking: "", thinkingSignature: "reasoning_content" };
					blocks.push(thinkingBlock);
					stream.push({ type: "thinking_start", contentIndex: indexOf(thinkingBlock), partial: output });
				}
				return thinkingBlock;
			};
			const ensureTool = (tc: any) => {
				const idx = typeof tc.index === "number" ? tc.index : -1;
				let block = idx >= 0 ? toolByIndex.get(idx) : undefined;
				if (!block && tc.id) block = toolById.get(tc.id);
				if (!block) {
					block = { type: "toolCall", id: tc.id || "", name: tc.function?.name || "", arguments: {}, partialArgs: "" };
					if (idx >= 0) toolByIndex.set(idx, block);
					if (tc.id) toolById.set(tc.id, block);
					blocks.push(block);
					stream.push({ type: "toolcall_start", contentIndex: indexOf(block), partial: output });
				}
				if (tc.id && !block.id) block.id = tc.id;
				if (tc.id) toolById.set(tc.id, block);
				if (tc.function?.name && !block.name) block.name = tc.function.name;
				return block;
			};

			for await (const data of sseData(response)) {
				if (data === "[DONE]") continue;
				const chunk = JSON.parse(data);
				if (chunk.id) output.responseId ||= chunk.id;
				if (chunk.usage) output.usage = parseUsage(chunk.usage);
				const choice = chunk.choices?.[0];
				if (!choice) continue;
				if (choice.finish_reason) output.stopReason = stopReason(choice.finish_reason);
				const delta = choice.delta ?? {};
				if (typeof delta.content === "string" && delta.content.length > 0) {
					const block = ensureText();
					block.text += delta.content;
					stream.push({ type: "text_delta", contentIndex: indexOf(block), delta: delta.content, partial: output });
				}
				const reasoning = delta.reasoning_content ?? delta.reasoning ?? delta.reasoning_text;
				if (typeof reasoning === "string" && reasoning.length > 0) {
					const block = ensureThinking();
					block.thinking += reasoning;
					stream.push({ type: "thinking_delta", contentIndex: indexOf(block), delta: reasoning, partial: output });
				}
				if (Array.isArray(delta.tool_calls)) {
					for (const tc of delta.tool_calls) {
						const block = ensureTool(tc);
						const argDelta = tc.function?.arguments || "";
						if (argDelta) {
							block.partialArgs = (block.partialArgs ?? "") + argDelta;
							block.arguments = safeJson(block.partialArgs);
						}
						stream.push({ type: "toolcall_delta", contentIndex: indexOf(block), delta: argDelta, partial: output });
					}
				}
			}

			for (const block of [...blocks]) {
				const contentIndex = indexOf(block as any);
				if (block.type === "text") stream.push({ type: "text_end", contentIndex, content: block.text, partial: output });
				else if (block.type === "thinking") stream.push({ type: "thinking_end", contentIndex, content: block.thinking, partial: output });
				else if (block.type === "toolCall") {
					const tool = block as ToolCall & { partialArgs?: string };
					tool.arguments = safeJson(tool.partialArgs ?? JSON.stringify(tool.arguments ?? {}));
					delete tool.partialArgs;
					stream.push({ type: "toolcall_end", contentIndex, toolCall: tool, partial: output });
				}
			}
			stream.push({ type: "done", reason: output.stopReason as any, message: output });
			stream.end();
		} catch (error) {
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();
	return stream;
}
