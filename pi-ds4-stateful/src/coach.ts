import type { ExtensionContext, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { MODEL_ID } from "./config.ts";
import { textOfContent } from "./openai.ts";
import { bashFileReadFallbackReason } from "./policies/bash-file-read-guard.ts";
import type { RuntimeConfig } from "./types.ts";
import { sha1, stableJson } from "./util.ts";

interface CoachAdvice {
	severity: "none" | "hint" | "warning";
	guidance: string[];
}

interface CoachInput {
	trigger: string;
	latestUser?: string;
	toolName: string;
	toolInput: unknown;
	toolResult: string;
	isError: boolean;
}

const COACH_SYSTEM_PROMPT = `You are a coding-agent tool-use coach for Pi.

Given the user's latest goal, a recent tool call, and its result, produce concise guidance for the main coding agent.

Rules:
- Do not solve the user's task.
- Do not invent file contents, command output, file names, languages, or facts not present in the input.
- Write bullets as direct imperatives to the main agent, e.g. "Read path/to/file.zig next." Do not say "guide the agent".
- If search output contains relevant file paths, tell the agent to READ 1-3 exact paths next, not run more searches.
- For support/feature yes-no questions, prefer source/implementation files over plans; use docs/plans only as secondary evidence.
- Do not suggest new extensions/languages (such as *.rs) unless they appear in the tool input/result.
- After empty searches, discourage equivalent case variants; after search hits, discourage repeated grep/rg and unrelated docs-first detours.
- Focus only on the next tool/action or what to avoid.
- Max 3 bullets, each under 180 characters.
- If no useful guidance is needed, return {"severity":"none","guidance":[]}.
- Return strict JSON only: {"severity":"none|hint|warning","guidance":["..."]}.`;

function truncate(text: string, max: number): string {
	if (text.length <= max) return text;
	return `${text.slice(0, max)}\n…[truncated ${text.length - max} chars]`;
}

function duration(ms: number): string {
	return ms < 10000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms / 1000)}s`;
}

function contentText(content: unknown): string {
	return truncate(textOfContent(content).trim(), 4000);
}

function userText(message: any): string {
	return contentText(message?.content ?? "");
}

function latestUserMessage(ctx: ExtensionContext): string | undefined {
	for (const entry of [...ctx.sessionManager.getBranch()].reverse()) {
		const message = (entry as any).message;
		if (message?.role === "user") {
			const text = userText(message).trim();
			return text ? truncate(text, 1200) : undefined;
		}
	}
	return undefined;
}

function commandText(input: unknown): string {
	return input && typeof input === "object" && typeof (input as { command?: unknown }).command === "string"
		? (input as { command: string }).command
		: "";
}

function looksLikeSearchCommand(command: string): boolean {
	return /(?:^|[;&|\s])(?:rg|grep)(?:\s|$)/.test(command);
}

function isNoOutput(text: string): boolean {
	return text.length === 0 || text === "(no output)";
}

function failureTrigger(event: ToolResultEvent, resultText: string): string | undefined {
	if (event.isError) return `${event.toolName}_error`;
	if (event.toolName === "bash" && bashFileReadFallbackReason(event.input)) return "bash_file_content_dump";
	if (event.toolName === "bash" && looksLikeSearchCommand(commandText(event.input))) {
		return isNoOutput(resultText) ? "bash_empty_search" : "bash_search_hits";
	}
	if (event.toolName === "bash" && /\b(error|failed|failure|traceback|exception|no such file|not found|permission denied)\b/i.test(resultText)) {
		return "bash_suspicious_output";
	}
	return undefined;
}

function parseCoachJson(raw: unknown): CoachAdvice | undefined {
	const text = typeof raw === "string" ? raw.trim() : "";
	if (!text) return undefined;
	const unwrapped = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
	const start = unwrapped.indexOf("{");
	const end = unwrapped.lastIndexOf("}");
	if (start < 0 || end < start) return undefined;
	try {
		const parsed = JSON.parse(unwrapped.slice(start, end + 1));
		const severity = parsed?.severity === "warning" || parsed?.severity === "hint" ? parsed.severity : "none";
		const guidance = Array.isArray(parsed?.guidance)
			? parsed.guidance.filter((g: unknown) => typeof g === "string" && g.trim()).map((g: string) => g.trim()).slice(0, 3)
			: [];
		return { severity, guidance };
	} catch {
		return undefined;
	}
}

function abortSignalWithTimeout(parent: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; cleanup: () => void } {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);
	const abort = () => controller.abort();
	if (parent?.aborted) abort();
	else parent?.addEventListener("abort", abort, { once: true });
	return {
		signal: controller.signal,
		cleanup: () => {
			clearTimeout(timer);
			parent?.removeEventListener("abort", abort);
		},
	};
}

export class Ds4ToolCoach {
	private seen = new Set<string>();
	calls = 0;
	attempts = 0;
	lastDurationMs = 0;
	lastSummary = "coach disabled";

	constructor(private readonly config: RuntimeConfig) {
		this.lastSummary = config.coachEnabled ? "coach ready" : "coach disabled";
	}

	clear(): void {
		this.seen.clear();
		this.lastSummary = this.config.coachEnabled ? "coach ready" : "coach disabled";
	}

	async reviewToolResult(event: ToolResultEvent, ctx: ExtensionContext): Promise<CoachAdvice | undefined> {
		if (!this.config.coachEnabled) return undefined;
		const resultText = contentText(event.content);
		const trigger = failureTrigger(event, resultText);
		if (!trigger) return undefined;

		const key = sha1({ trigger, toolName: event.toolName, input: event.input, result: resultText.slice(0, 1000), isError: event.isError });
		if (this.seen.has(key)) return undefined;
		this.seen.add(key);

		const input: CoachInput = {
			trigger,
			latestUser: latestUserMessage(ctx),
			toolName: event.toolName,
			toolInput: event.input,
			toolResult: resultText,
			isError: event.isError,
		};

		const timeout = abortSignalWithTimeout(ctx.signal, this.config.coachTimeoutMs);
		const started = Date.now();
		this.attempts++;
		this.lastSummary = `coach running for ${trigger}`;
		try {
			const apiKey = process.env.DS4_STATEFUL_API_KEY || process.env.DS4_API_KEY || "dsv4-local";
			const response = await fetch(`${this.config.statelessBaseUrl.replace(/\/+$/, "")}/chat/completions`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${apiKey}`,
				},
				body: JSON.stringify({
					model: MODEL_ID,
					messages: [
						{ role: "system", content: COACH_SYSTEM_PROMPT },
						{ role: "user", content: stableJson(input) },
					],
					stream: false,
					temperature: 0,
					max_tokens: this.config.coachMaxTokens,
					thinking: { type: "disabled" },
				}),
				signal: timeout.signal,
			});
			const took = Date.now() - started;
			this.lastDurationMs = took;
			if (!response.ok) {
				this.lastSummary = `coach HTTP ${response.status} in ${duration(took)}`;
				return undefined;
			}
			const json = await response.json() as any;
			const raw = json?.choices?.[0]?.message?.content ?? json?.choices?.[0]?.text;
			const advice = parseCoachJson(raw);
			if (!advice || advice.guidance.length === 0) {
				this.lastSummary = `coach: no guidance for ${trigger} in ${duration(took)}`;
				return undefined;
			}
			this.calls++;
			this.lastSummary = `${advice.severity} in ${duration(took)}: ${advice.guidance[0]}`;
			return advice;
		} catch (error) {
			const took = Date.now() - started;
			this.lastDurationMs = took;
			if (ctx.signal?.aborted) return undefined;
			this.lastSummary = timeout.signal.aborted
				? `coach timed out after ${duration(took)} (limit ${duration(this.config.coachTimeoutMs)})`
				: `coach failed in ${duration(took)}: ${error instanceof Error ? error.message : String(error)}`;
			return undefined;
		} finally {
			timeout.cleanup();
		}
	}
}

export function formatCoachNote(advice: CoachAdvice): string {
	const bullets = advice.guidance.map((g) => `- ${g}`).join("\n");
	return `\n\nDS4 coach control note (${advice.severity}, same-model side call): Follow this before further grep/rg or final answer.\n${bullets}`;
}
