import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { API_ID, PROVIDER_ID } from "./config.ts";
import type { StatefulMode, StatefulUiRequest } from "./types.ts";
import { asRecord, intField, plural, stringField } from "./util.ts";

export function isDs4StatefulContext(ctx: ExtensionContext): boolean {
	return ctx.model?.provider === PROVIDER_ID || ctx.model?.api === API_ID;
}

export function messageRole(message: unknown): unknown {
	return message && typeof message === "object" ? (message as Record<string, unknown>).role : undefined;
}

export function isDeltaMessage(message: unknown): boolean {
	const role = messageRole(message);
	return role === "user" || role === "tool" || role === "function";
}

export function hasUserMessage(messages: unknown[]): boolean {
	return messages.some((message) => messageRole(message) === "user");
}

export function isToolOnlyDelta(messages: unknown[]): boolean {
	return messages.length > 0 && messages.every((message) => {
		const role = messageRole(message);
		return role === "tool" || role === "function";
	});
}

export function requestInfoFromPayload(payload: unknown): StatefulUiRequest | undefined {
	const p = asRecord(payload);
	if (!p) return undefined;
	const mode: StatefulMode | undefined = p.mode === "reset" || p.mode === "delta" ? p.mode : undefined;
	const sessionId = stringField(p.session_id);
	if (!mode || !sessionId) return undefined;
	const debug = asRecord(p.stateful_debug);
	return {
		mode,
		sessionId,
		reason: stringField(debug?.reason, stringField(p.stateful_debug_reason, mode)),
		parentRevision: intField(p.parent_revision, 0),
		storedRevision: intField(debug?.stored_revision, intField(p.stateful_debug_stored_revision, 0)),
		fullMessages: intField(debug?.full_messages, intField(p.stateful_debug_full_messages, Array.isArray(p.messages) ? p.messages.length : -1)),
		sentMessages: intField(debug?.sent_messages, intField(p.stateful_debug_sent_messages, Array.isArray(p.messages) ? p.messages.length : -1)),
		previousMessages: intField(debug?.previous_messages, intField(p.stateful_debug_previous_messages, -1)),
		startedAt: Date.now(),
	};
}

export function compactReason(reason: string): string {
	return reason
		.replace(/^append-only delta /, "")
		.replace("delta rejected by server; retrying full reset", "409 -> reset")
		.replace("message prefix changed", "prefix changed")
		.replace("tool schema changed", "tools changed")
		.replace("append contained assistant/system replay", "assistant/system replay")
		.replace("auto reset for new user turn", "new user reset")
		.replace("new user turn reset", "new user reset")
		.replace("user-turn delta by policy", "user delta")
		.replace("initial/reset", "initial reset");
}

export function requestSummary(info: StatefulUiRequest): string {
	const sent = info.sentMessages >= 0 ? info.sentMessages : 0;
	const full = info.fullMessages >= 0 ? info.fullMessages : 0;
	if (info.mode === "delta") {
		return `delta parent r${info.parentRevision} +${sent}/${full} msgs`;
	}
	const rev = info.storedRevision > 0 ? ` from stored r${info.storedRevision}` : "";
	return `reset${rev} ${plural(full, "msg")}`;
}

export function sessionTag(info: StatefulUiRequest): string {
	return `session=${info.sessionId.length > 21 ? `${info.sessionId.slice(0, 14)}…${info.sessionId.slice(-6)}` : info.sessionId}`;
}

export function decisionSummary(info: StatefulUiRequest): string {
	const previous = info.previousMessages >= 0 ? info.previousMessages : 0;
	const sent = info.sentMessages >= 0 ? info.sentMessages : 0;
	const full = info.fullMessages >= 0 ? info.fullMessages : 0;
	const parent = info.mode === "delta" ? `parent r${info.parentRevision}` : "parent r0";
	const stored = `stored r${info.storedRevision}`;
	return `${requestSummary(info)}; ${compactReason(info.reason)}; ${sessionTag(info)}; ${parent}; ${stored}; prev=${previous}; sent=${sent}/${full}`;
}
