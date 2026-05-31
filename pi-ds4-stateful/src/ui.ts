import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { STATUS_KEY } from "./config.ts";
import { compactReason, decisionSummary, requestInfoFromPayload, requestSummary, sessionTag } from "./protocol.ts";
import { ReadGuard } from "./policies/read-guard.ts";
import { StatefulSessionStore } from "./session-state.ts";
import type { Ds4ToolCoach } from "./coach.ts";
import type { PrefillProgress, RuntimeConfig, StatefulUiRequest, UiPhase } from "./types.ts";
import { chars, plural, shortId } from "./util.ts";

const PROGRESS_WIDGET_KEY = `${STATUS_KEY}:progress`;

function duration(seconds: number): string {
	return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
}

function elapsed(info: StatefulUiRequest): string {
	return duration(Math.max(0, (Date.now() - info.startedAt) / 1000));
}

function rate(tps: number): string {
	if (!Number.isFinite(tps) || tps <= 0) return "-- tok/s";
	return tps >= 1000 ? `${(tps / 1000).toFixed(1)}k tok/s` : `${tps.toFixed(tps >= 100 ? 0 : 1)} tok/s`;
}

function progressPercent(progress: PrefillProgress): number {
	if (Number.isFinite(progress.percent)) return Math.max(0, Math.min(100, progress.percent));
	return progress.total > 0 ? Math.max(0, Math.min(100, 100 * progress.current / progress.total)) : 0;
}

export class StatefulUi {
	private activeRequest: StatefulUiRequest | undefined;
	private activePhase: UiPhase = "idle";
	private activeTextChars = 0;
	private activeThinkingChars = 0;
	private activeToolCalls = 0;
	private activeTools = new Map<string, string>();
	private activePrefill: PrefillProgress | undefined;
	private activeDecode: PrefillProgress | undefined;
	private activeCtx: ExtensionContext | undefined;
	private lastUiRefresh = 0;
	lastHttpSummary = "no response yet";

	constructor(
		private readonly config: RuntimeConfig,
		private readonly sessions: StatefulSessionStore,
		private readonly readGuard: ReadGuard,
		private readonly coach?: Ds4ToolCoach,
	) {}

	beginAgent(ctx: ExtensionContext): void {
		this.activeCtx = ctx;
		this.activePhase = "request";
		this.activeRequest = undefined;
		this.activeTools = new Map<string, string>();
		this.activePrefill = undefined;
		this.activeDecode = undefined;
		this.activeTextChars = 0;
		this.activeThinkingChars = 0;
		this.activeToolCalls = 0;
		this.refresh(ctx, true);
	}

	beforeProviderRequest(payload: unknown, ctx: ExtensionContext): void {
		this.activeCtx = ctx;
		const info = requestInfoFromPayload(payload);
		if (!info) return;
		this.activeRequest = info;
		this.activePhase = "request";
		this.activePrefill = undefined;
		this.activeDecode = undefined;
		this.activeTextChars = 0;
		this.activeThinkingChars = 0;
		this.activeToolCalls = 0;
		this.activeTools = new Map<string, string>();
		this.sessions.lastRequestSummary = decisionSummary(info);
		this.sessions.lastDecisionSummary = this.sessions.lastRequestSummary;
		this.refresh(ctx, true);
	}

	afterProviderResponse(status: number, headers: Record<string, string> | undefined, ctx: ExtensionContext): void {
		this.activeCtx = ctx;
		if (!this.activeRequest) return;
		this.activeRequest.httpStatus = status;
		const revision = headers?.["x-ds4-session-revision"] ?? headers?.["X-DS4-Session-Revision"];
		this.lastHttpSummary = `HTTP ${status}${this.activeRequest ? ` for ${requestSummary(this.activeRequest)}` : ""}${revision ? `; server r${revision}` : ""}`;
		this.activePhase = status === 409 && this.activeRequest.mode === "delta" ? "retry" : status >= 400 ? "error" : "response";
		this.refresh(ctx, true);
	}

	prefillProgress(progress: PrefillProgress): void {
		if (!this.activeRequest) return;
		if (progress.phase === "decode") {
			this.activeDecode = progress;
			this.activePrefill = undefined;
			if (this.activePhase !== "retry" && this.activePhase !== "error") this.activePhase = "stream";
		} else {
			this.activePrefill = progress;
			this.activeDecode = undefined;
			if (this.activePhase !== "retry" && this.activePhase !== "error" && this.activePhase !== "stream") {
				this.activePhase = "request";
			}
		}
		if (this.activeCtx) this.refresh(this.activeCtx);
	}

	messageUpdate(update: any, ctx: ExtensionContext): void {
		this.activeCtx = ctx;
		if (!this.activeRequest) return;
		this.activePrefill = undefined;
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
		this.activeCtx = ctx;
		if (!this.activeRequest || message.role !== "assistant") return;
		this.activePhase = message.stopReason === "toolUse" ? "tool-call" : message.stopReason === "error" ? "error" : "done";
		this.refresh(ctx, true);
	}

	toolExecutionStart(toolCallId: string, toolName: string, ctx: ExtensionContext): void {
		this.activeCtx = ctx;
		this.activeTools.set(toolCallId, toolName);
		this.activePhase = "tools";
		this.refresh(ctx, true);
	}

	toolExecutionEnd(toolCallId: string, ctx: ExtensionContext): void {
		this.activeCtx = ctx;
		this.activeTools.delete(toolCallId);
		this.activePhase = this.activeTools.size > 0 ? "tools" : "done";
		this.refresh(ctx, true);
	}

	clearWorking(ctx: ExtensionContext): void {
		this.activePhase = "idle";
		this.activeRequest = undefined;
		this.activeTools = new Map<string, string>();
		this.activePrefill = undefined;
		this.activeDecode = undefined;
		this.activeCtx = undefined;
		this.activeTextChars = 0;
		this.activeThinkingChars = 0;
		this.activeToolCalls = 0;
		ctx.ui.setStatus(STATUS_KEY, this.statusText(ctx));
		if (ctx.hasUI) {
			ctx.ui.setWidget(PROGRESS_WIDGET_KEY, undefined);
			ctx.ui.setWorkingMessage();
		}
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
		const samplingLine = `presence=${this.config.presencePenalty.toFixed(2)} frequency=${this.config.frequencyPenalty.toFixed(2)} repeat=${this.config.repeatPenalty.toFixed(2)} last_n=${this.config.repeatLastN}`;
		const coachLine = this.config.coachEnabled
			? t.fg("success", `on calls=${this.coach?.calls ?? 0}/${this.coach?.attempts ?? 0} last=${this.coach?.lastDurationMs ? duration(this.coach.lastDurationMs / 1000) : "--"} max=${this.config.coachMaxTokens} timeout=${this.config.coachTimeoutMs}ms`)
			: t.fg("dim", "off");
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
			row("read guard", this.config.coachEnabled
				? `${t.fg("dim", "bypassed by coach")} ${t.fg("dim", `configured=${this.config.readGuardEnabled ? this.config.readGuardMode : "off"}`)}`
				: `${this.config.readGuardEnabled ? t.fg("success", this.config.readGuardMode) : t.fg("dim", "off")} ${t.fg("dim", `seen=${this.readGuard.seenCount}`)}`),
			row("sampling", t.fg("dim", samplingLine)),
			row("coach", coachLine),
			row("last coach", t.fg("dim", this.coach?.lastSummary ?? "none")),
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
		this.updateProgressWidget(ctx);
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

	private progressBar(progress: PrefillProgress, ctx: ExtensionContext, width = 18): string {
		const t = ctx.ui.theme;
		const pct = progressPercent(progress);
		const filled = Math.max(0, Math.min(width, Math.round(width * pct / 100)));
		return t.fg("success", "█".repeat(filled)) + t.fg("dim", "░".repeat(width - filled));
	}

	private progressSummary(progress: PrefillProgress | undefined, ctx: ExtensionContext, withBar = false): string | undefined {
		if (!progress) return undefined;
		const speed = progress.chunkTps > 0 ? progress.chunkTps : progress.avgTps;
		const avg = progress.avgTps > 0 && progress.chunkTps > 0 ? ` avg ${rate(progress.avgTps)}` : "";
		if (progress.phase === "decode") {
			// Decode length is not known until a stop/tool call/length condition is hit.
			// The server's `total` is the max-token cap, not the expected response size,
			// so render decode as an indeterminate generated-token counter.
			const current = Math.max(0, progress.current);
			return `${current} tok · ${rate(speed)}${avg} · ${duration(progress.elapsed)}`;
		}
		const pct = progressPercent(progress);
		const total = progress.total > 0 ? progress.total : 0;
		const current = total > 0 ? Math.max(0, Math.min(total, progress.current)) : Math.max(0, progress.current);
		const count = total > 0 ? `${current}/${total}` : `${current}`;
		const bar = withBar ? `${this.progressBar(progress, ctx)} ` : "";
		return `${bar}${pct.toFixed(1)}% ${count} tok · ${rate(speed)}${avg} · ${duration(progress.elapsed)}`;
	}

	private prefillSummary(ctx: ExtensionContext, withBar = false): string | undefined {
		return this.progressSummary(this.activePrefill, ctx, withBar);
	}

	private decodeSummary(ctx: ExtensionContext, withBar = false): string | undefined {
		return this.progressSummary(this.activeDecode, ctx, withBar);
	}

	private updateProgressWidget(ctx: ExtensionContext): void {
		const line = this.progressWidgetLine(ctx);
		ctx.ui.setWidget(PROGRESS_WIDGET_KEY, line ? [line] : undefined);
	}

	private progressWidgetLine(ctx: ExtensionContext): string | undefined {
		if (!this.config.enabled || !this.activeRequest) return undefined;
		// Show prefill in the widget: during prefill, normal assistant streaming has
		// not started yet, and this is the most reliable visible TUI location. Decode
		// stays in the normal working message to avoid duplicate decode lines.
		if (this.activeDecode || this.activePhase === "stream") return undefined;
		const t = ctx.ui.theme;
		const label = this.activePrefill ? "prefill" : this.activePhase === "request" || this.activePhase === "response" ? "prefill" : undefined;
		if (!label) return undefined;
		const progress = this.activePrefill;
		const summary = progress
			? this.progressSummary(progress, ctx, true)
			: `${this.progressBar({ phase: label, current: 0, total: 1, percent: 0, chunkTps: 0, avgTps: 0, elapsed: Math.max(0, (Date.now() - this.activeRequest.startedAt) / 1000), timestamp: Date.now() }, ctx)} waiting for server progress · ${elapsed(this.activeRequest)}`;
		return `${t.fg("accent", "◆ DS4")} ${t.fg("accent", label)} ${summary}`;
	}

	private statusText(ctx: ExtensionContext): string | undefined {
		if (!this.config.enabled) return undefined;
		const t = ctx.ui.theme;
		const brand = `${t.fg("accent", "◆")} ${t.fg("accent", "DS4")}`;
		if (this.activeRequest && this.activePhase !== "idle") {
			const info = this.activeRequest;
			const sent = info.sentMessages >= 0 ? info.sentMessages : 0;
			const full = info.fullMessages >= 0 ? info.fullMessages : 0;
			const prefill = this.activePhase === "request" ? this.prefillSummary(ctx) : undefined;
			const decode = this.activePhase === "stream" ? this.decodeSummary(ctx) : undefined;
			const streamDetail = decode ?? `think ${chars(this.activeThinkingChars)} · text ${chars(this.activeTextChars)} · tools ${this.activeToolCalls}`;
			const detail = this.activePhase === "stream"
				? ` ${t.fg("dim", streamDetail)}`
				: this.activePhase === "tools"
					? ` ${t.fg("dim", `${plural(this.activeTools.size, "tool")}`)}`
					: prefill
						? ` ${t.fg("dim", prefill)}`
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
		if (this.activePhase === "request" || this.activePhase === "response") {
			// Prefill is rendered in PROGRESS_WIDGET_KEY; keep the spinner line clear so
			// the same prefill progress is not shown twice.
			return undefined;
		}
		if (this.activePhase === "retry") {
			return `${t.fg("warning", "↻ DS4 stale delta")} ${label} ${t.fg("dim", "→ full reset")} · ${clock}`;
		}
		if (this.activePhase === "stream") {
			const decode = this.decodeSummary(ctx, true);
			if (decode) return `${t.fg("accent", "◒ DS4 decode")} ${label}${http} ${decode}`;
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
