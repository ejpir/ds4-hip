import type { Api, AssistantMessageEvent, Context, Model, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { API_ID, MODEL_ID, PROVIDER_ID } from "./config.ts";
import { buildOpenAIPayload, emptyUsage, streamFetchOnce } from "./openai.ts";
import { StatefulSessionStore } from "./session-state.ts";
import type { PendingCommit, PrefillProgress, RuntimeConfig } from "./types.ts";

function isStatefulConflict(event: AssistantMessageEvent, pending: PendingCommit | undefined): boolean {
	if (event.type !== "error" || pending?.mode !== "delta") return false;
	const msg = event.error.errorMessage ?? "";
	return msg.includes("HTTP 409") || msg.includes("stateful continuation state");
}

function applySamplingConfig(payload: unknown, config: RuntimeConfig): unknown {
	if (!payload || typeof payload !== "object") return payload;
	const out = payload as Record<string, unknown>;
	if (config.presencePenalty !== 0) out.presence_penalty = config.presencePenalty;
	if (config.frequencyPenalty !== 0) out.frequency_penalty = config.frequencyPenalty;
	if (config.repeatPenalty !== 1) out.repeat_penalty = config.repeatPenalty;
	if (config.repeatLastN !== 1024) out.repeat_last_n = config.repeatLastN;
	return out;
}

export function createDs4StatefulStream(
	config: RuntimeConfig,
	sessions: StatefulSessionStore,
	onPrefillProgress?: (progress: PrefillProgress) => void,
) {
	return function streamDs4Stateful(model: Model<Api>, context: Context, options?: SimpleStreamOptions) {
		const outer = createAssistantMessageEventStream();
		const key = sessions.key(options);
		const apiKey = options?.apiKey || process.env.DS4_STATEFUL_API_KEY || process.env.DS4_API_KEY || "dsv4-local";
		const userOnPayload = options?.onPayload;
		const userOnResponse = options?.onResponse;

		const runStateless = async () => {
			const statelessModel: Model<Api> = { ...model, baseUrl: config.statelessBaseUrl };
			const payload = buildOpenAIPayload(statelessModel, context, options);
			const finalPayload = await userOnPayload?.(payload, statelessModel);
			sessions.lastMode = "none";
			sessions.lastReason = "stateful disabled; sent stateless chat/completions";
			return streamFetchOnce(statelessModel, finalPayload === undefined ? payload : finalPayload, apiKey, options, userOnResponse, onPrefillProgress);
		};

		const runOnce = async (forceReset: boolean, pendingRef: { current?: PendingCommit }) => {
			const basePayload = applySamplingConfig(buildOpenAIPayload(model, context, options), config);
			const selected = sessions.choosePayload(basePayload, key, forceReset, config);
			pendingRef.current = selected.pending;
			sessions.lastMode = selected.mode;
			sessions.lastReason = selected.reason;
			const finalPayload = await userOnPayload?.(selected.payload, model);
			return streamFetchOnce(model, finalPayload === undefined ? selected.payload : finalPayload, apiKey, options, userOnResponse, onPrefillProgress);
		};

		(async () => {
			if (!config.enabled) {
				const inner = await runStateless();
				for await (const event of inner) outer.push(event);
				outer.end();
				return;
			}
			let retried = false;
			let pendingRef: { current?: PendingCommit } = {};
			let inner = await runOnce(false, pendingRef);

			while (true) {
				let retry = false;
				for await (const event of inner) {
					if (!retried && isStatefulConflict(event, pendingRef.current)) {
						retried = true;
						retry = true;
						pendingRef = {};
						inner = await runOnce(true, pendingRef);
						break;
					}
					outer.push(event);
					if (event.type === "done") sessions.commit(pendingRef.current, event.message);
				}
				if (!retry) break;
			}
			outer.end();
		})().catch((error) => {
			const errMessage = error instanceof Error ? error.message : String(error);
			outer.push({
				type: "error",
				reason: "error",
				error: {
					role: "assistant",
					content: [],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: emptyUsage(),
					stopReason: "error",
					errorMessage: errMessage,
					timestamp: Date.now(),
				},
			});
			outer.end();
		});

		return outer;
	};
}

export function ds4StatefulProviderConfig(
	config: RuntimeConfig,
	sessions: StatefulSessionStore,
	onPrefillProgress?: (progress: PrefillProgress) => void,
) {
	return {
		name: "ds4.c stateful",
		baseUrl: config.baseUrl,
		apiKey: process.env.DS4_STATEFUL_API_KEY ? "DS4_STATEFUL_API_KEY" : "dsv4-local",
		api: API_ID,
		models: [
			{
				id: MODEL_ID,
				name: "DeepSeek V4 Flash (ds4 stateful)",
				reasoning: true,
				thinkingLevelMap: {
					off: null,
					minimal: "low",
					low: "low",
					medium: "medium",
					high: "high",
					xhigh: "xhigh",
				},
				input: ["text"],
				contextWindow: config.contextWindow,
				maxTokens: config.maxTokens,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				compat: {
					supportsStore: false,
					supportsDeveloperRole: false,
					supportsReasoningEffort: true,
					supportsUsageInStreaming: true,
					maxTokensField: "max_tokens",
					supportsStrictMode: false,
					thinkingFormat: "deepseek",
					requiresReasoningContentOnAssistantMessages: true,
				},
			},
		],
		streamSimple: createDs4StatefulStream(config, sessions, onPrefillProgress),
	};
}

export { PROVIDER_ID };
