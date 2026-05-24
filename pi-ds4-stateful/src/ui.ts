import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { STATUS_KEY } from "./config.ts";
import { compactReason, decisionSummary, requestInfoFromPayload, requestSummary, sessionTag } from "./protocol.ts";
import { ReadGuard } from "./policies/read-guard.ts";
import { StatefulSessionStore } from "./session-state.ts";
import type { RuntimeConfig, StatefulUiRequest, UiPhase } from "./types.ts";
import { chars, plural, shortId } from "./util.ts";

function elapsed(info: StatefulUiRequest): string {
	const seconds = Math.max(0, (Date.now() - info.startedAt) / 1000);
	return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
}

export class StatefulUi {
	private activeRequest: StatefulUiRequest | undefined;
	private activePhase: UiPhase = "idle";
	private activeTextChars = 0;
	private activeThinkingChars = 0;
	private activeToolCalls = 0;
	private activeTools = new Map<string, string>();
	private lastUiRefresh = 0;
	lastHttpSummary = "no response yet";

	constructor(
		private readonly config: RuntimeConfig,
		private readonly sessions: StatefulSessionStore,
		private readonly readGuard: ReadGuard,
	) {}

	beginAgent(ctx: ExtensionContext): void {
		this.activePhase = "request";
		this.activeRequest = undefined;
		this.activeTools = new Map<string, string>();
		this.activeTextChars = 0;
		this.activeThinkingChars = 0;
		this.activeToolCalls = 0;
		this.refresh(ctx, true);
	}

	beforeProviderRequest(payload: unknown, ctx: ExtensionContext): void {
		const info = requestInfoFromPayload(payload);
		if (!info) return;
		this.activeRequest = info;
		this.activePhase = "request";
		this.activeTextChars = 0;
		this.activeThinkingChars = 0;
		this.activeToolCalls = 0;
		this.activeTools = new Map<string, string>();
		this.sessions.lastRequestSummary = decisionSummary(info);
		this.sessions.lastDecisionSummary = this.sessions.lastRequestSummary;
		this.refresh(ctx, true);
	}

	afterProviderResponse(status: number, headers: Record<string, string> | undefined, ctx: ExtensionContext): void {
		if (!this.activeRequest) return;
		this.activeRequest.httpStatus = status;
		const revision = headers?.["x-ds4-session-revision"] ?? headers?.["X-DS4-Session-Revision"];
		this.lastHttpSummary = `HTTP ${status}${this.activeRequest ? ` for ${requestSummary(this.activeRequest)}` : ""}${revision ? `; server r${revision}` : ""}`;
		this.activePhase = status === 409 && this.activeRequest.mode === "delta" ? "retry" : status >= 400 ? "error" : "response";
		this.refresh(ctx, true);
	}

	messageUpdate(update: any, ctx: ExtensionContext): void {
		if (!this.activeRequest) return;
		if (update.type === "text_delta") this.activeTextChars += update.delta.length;
		else if (update.type === "thinking_delta") this.activeThinkingChars += update.delta.length;
		else if (update.type === "toolcall_start") this.activeToolCalls = Math.max(this.activeToolCalls, 1);
		if (update.type === "toolcall_start" || update.type === "toolcall_delta" || update.type === "toolcall_end") {
			const blocks = update.partial.content.filter((block: { type?: string }) => block.type === "toolCall");
			this.activeToolCalls = Math.max(this.activeToolCalls, blocks.length);
		}
		this.activePhase = "stream";
		this.refresh(ctx);
	}

	messageEnd(message: { role: string; stopReason?: string }, ctx: ExtensionContext): void {
		if (!this.activeRequest || message.role !== "assistant") return;
		this.activePhase = message.stopReason === "toolUse" ? "tool-call" : message.stopReason === "error" ? "error" : "done";
		this.refresh(ctx, true);
	}

	toolExecutionStart(toolCallId: string, toolName: string, ctx: ExtensionContext): void {
		this.activeTools.set(toolCallId, toolName);
		this.activePhase = "tools";
		this.refresh(ctx, true);
	}

	toolExecutionEnd(toolCallId: string, ctx: ExtensionContext): void {
		this.activeTools.delete(toolCallId);
		this.activePhase = this.activeTools.size > 0 ? "tools" : "done";
		this.refresh(ctx, true);
	}

	clearWorking(ctx: ExtensionContext): void {
		this.activePhase = "idle";
		this.activeRequest = undefined;
		this.activeTools = new Map<string, string>();
		this.activeTextChars = 0;
		this.activeThinkingChars = 0;
		this.activeToolCalls = 0;
		ctx.ui.setStatus(STATUS_KEY, this.statusText(ctx));
		if (ctx.hasUI) ctx.ui.setWorkingMessage();
	}

	applyStatus(ctx: ExtensionContext): void {
		ctx.ui.setStatus(STATUS_KEY, this.statusText(ctx));
	}

	markReset(): void {
		this.lastHttpSummary = "no response since reset";
	}

	detailedStatus(ctx: ExtensionContext): string {
		const t = ctx.ui.theme;
		const model = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "none";
		const usage = ctx.getContextUsage();
		const usageLine = usage
			? `${usage.tokens === null ? "unknown" : `${usage.tokens}/${usage.contextWindow}`}${usage.percent === null ? "" : ` (${usage.percent.toFixed(1)}%)`}`
			: "unknown";
		const activeLine = this.activeRequest
			? `${this.activePhase}; ${decisionSummary(this.activeRequest)}${this.activeRequest.httpStatus ? `; HTTP ${this.activeRequest.httpStatus}` : ""}`
			: "idle";
		const labelWidth = 22;
		const title = `${t.fg("accent", "◆")} ${t.fg("accent", t.bold("DS4 stateful provider"))}`;
		const rule = (name: string) => `${t.fg("dim", "┌─")} ${t.fg("accent", name)} ${t.fg("dim", "─".repeat(Math.max(8, 56 - name.length)))}`;
		const row = (name: string, value: string) => `${t.fg("dim", "│")} ${t.fg("muted", name.padEnd(labelWidth, " "))} ${value}`;
		const foot = () => t.fg("dim", "└" + "─".repeat(60));
		const onOff = (enabled: boolean, on = "on", off = "off") => enabled ? t.fg("success", on) : t.fg("dim", off);
		const providerState = this.config.enabled ? t.fg("success", "enabled") : t.fg("warning", "disabled / passive");
		const userTurnNote = this.config.userTurnPolicy === "delta"
			? "new user messages use delta"
			: "new user messages reset; tool results use delta";
		const lines: string[] = [
			`${title} ${providerState}`,
			"",
			rule("Core"),
			row("model", model),
			row("stateful endpoint", this.config.baseUrl),
			row("stateless endpoint", this.config.statelessBaseUrl),
			row("context", usageLine),
			foot(),
			"",
			rule("Policy"),
			row("user turn", `${t.fg("text", this.config.userTurnPolicy)} ${t.fg("dim", `(${userTurnNote})`)}`),
			row("turn focus", onOff(this.config.turnFocusEnabled)),
			row("read guard", `${this.config.readGuardEnabled ? t.fg("success", this.config.readGuardMode) : t.fg("dim", "off")} ${t.fg("dim", `seen=${this.readGuard.seenCount}`)}`),
			row("last guard block", t.fg("dim", this.readGuard.lastSummary)),
			foot(),
			"",
			rule("Runtime"),
			row("active", activeLine),
			row("last decision", this.sessions.lastDecisionSummary),
			row("last request", this.sessions.lastRequestSummary),
			row("last HTTP", this.lastHttpSummary),
			row("last commit", this.sessions.lastCommitSummary),
			foot(),
			"",
			rule("Sessions"),
		];
		for (const line of this.sessions.sessionLines()) {
			if (line.startsWith("-")) lines.push(row("", t.fg("dim", line.slice(2))));
			else lines.push(row("count", t.fg("muted", line.replace(/^sessions:\s*/, ""))));
		}
		lines.push(foot());
		return lines.join("\n");
	}

	refresh(ctx: ExtensionContext, force = false): void {
		ctx.ui.setStatus(STATUS_KEY, this.statusText(ctx));
		if (!ctx.hasUI) return;
		const now = Date.now();
		if (!force && now - this.lastUiRefresh < 500) return;
		this.lastUiRefresh = now;
		ctx.ui.setWorkingMessage(this.workingMessage(ctx));
	}

	private phaseLabel(ctx: ExtensionContext): string {
		const t = ctx.ui.theme;
		switch (this.activePhase) {
			case "request": return t.fg("accent", "prefill");
			case "response": return t.fg("accent", "response");
			case "stream": return t.fg("accent", "stream");
			case "tool-call": return t.fg("warning", "tool-call");
			case "tools": return t.fg("warning", "tools");
			case "done": return t.fg("success", "done");
			case "retry": return t.fg("warning", "retry");
			case "error": return t.fg("error", "error");
			default: return t.fg("dim", this.activePhase);
		}
	}

	private modeBadge(ctx: ExtensionContext, mode: string): string {
		const t = ctx.ui.theme;
		if (mode === "delta") return t.fg("success", "Δ delta");
		if (mode === "reset") return t.fg("warning", "↺ reset");
		return t.fg("dim", mode);
	}

	private httpBadge(ctx: ExtensionContext): string {
		const status = this.activeRequest?.httpStatus;
		if (!status) return "";
		const t = ctx.ui.theme;
		const color = status >= 500 || status === 409 ? "warning" : status >= 400 ? "error" : "success";
		return ` ${t.fg(color, `HTTP ${status}`)}`;
	}

	private statusText(ctx: ExtensionContext): string | undefined {
		if (!this.config.enabled) return undefined;
		const t = ctx.ui.theme;
		const brand = `${t.fg("accent", "◆")} ${t.fg("accent", "DS4")}`;
		if (this.activeRequest && this.activePhase !== "idle") {
			const info = this.activeRequest;
			const sent = info.sentMessages >= 0 ? info.sentMessages : 0;
			const full = info.fullMessages >= 0 ? info.fullMessages : 0;
			const detail = this.activePhase === "stream"
				? ` ${t.fg("dim", `think ${chars(this.activeThinkingChars)} · text ${chars(this.activeTextChars)} · tools ${this.activeToolCalls}`)}`
				: this.activePhase === "tools"
					? ` ${t.fg("dim", `${plural(this.activeTools.size, "tool")}`)}`
					: "";
			return `${brand} ${this.modeBadge(ctx, info.mode)} ${this.phaseLabel(ctx)} ${t.fg("dim", `r${info.parentRevision} · ${sent}/${full} msg`)}${this.httpBadge(ctx)}${detail}`;
		}
		if (this.sessions.sessions.size === 1) {
			const first = this.sessions.sessions.entries().next().value as [string, any] | undefined;
			if (first) {
				const [key, state] = first;
				return `${brand} ${this.modeBadge(ctx, this.sessions.lastMode)} ${t.fg("dim", `${key} · r${state.revision} · ${state.messages.length} msgs · ${shortId(state.ds4SessionId, 10, 4)}`)}`;
			}
		}
		return `${brand} ${this.modeBadge(ctx, this.sessions.lastMode)} ${t.fg("dim", `${this.sessions.sessions.size} sessions`)}`;
	}

	private workingMessage(ctx: ExtensionContext): string | undefined {
		if (!this.config.enabled || !this.activeRequest || this.activePhase === "idle") return undefined;
		const t = ctx.ui.theme;
		const label = `${this.modeBadge(ctx, this.activeRequest.mode)} ${t.fg("dim", sessionTag(this.activeRequest))}`;
		const http = this.activeRequest.httpStatus ? ` ${this.httpBadge(ctx).trim()}` : "";
		const reason = t.fg("dim", compactReason(this.activeRequest.reason));
		const clock = t.fg("dim", elapsed(this.activeRequest));
		if (this.activePhase === "request") {
			return `${t.fg("accent", "◆ DS4")} ${label} ${t.fg("accent", "prefill")} ${reason} · ${clock}`;
		}
		if (this.activePhase === "retry") {
			return `${t.fg("warning", "↻ DS4 stale delta")} ${label} ${t.fg("dim", "→ full reset")} · ${clock}`;
		}
		if (this.activePhase === "stream") {
			const tool = this.activeToolCalls > 0 ? ` · ${plural(this.activeToolCalls, "tool call")}` : "";
			return `${t.fg("accent", "◒ DS4 streaming")} ${label}${http} ${t.fg("dim", `think ${chars(this.activeThinkingChars)} · text ${chars(this.activeTextChars)}${tool}`)}`;
		}
		if (this.activePhase === "tool-call") {
			return `${t.fg("warning", "⚙ DS4 tool call")} ${label} ${t.fg("dim", `${plural(this.activeToolCalls, "call")} ready`)}`;
		}
		if (this.activePhase === "tools") {
			const names = [...this.activeTools.values()].slice(0, 3).join(", ");
			return `${t.fg("warning", "⚙ Pi tools")} ${label} ${t.fg("dim", `${plural(this.activeTools.size, "tool")}${names ? `: ${names}` : ""}`)}`;
		}
		if (this.activePhase === "done") {
			return `${t.fg("success", "✓ DS4 done")} ${label}${http}`;
		}
		if (this.activePhase === "error") {
			return `${t.fg("error", "✕ DS4 error")} ${label}${http}`;
		}
		return `${t.fg("accent", "◆ DS4")} ${label}${http} ${reason}`;
	}
}
