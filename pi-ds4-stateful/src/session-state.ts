import { randomBytes } from "node:crypto";
import type { AssistantMessage, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { assistantToOpenAI } from "./openai.ts";
import { decisionSummary, hasUserMessage, isDeltaMessage, isToolOnlyDelta, messageRole, requestInfoFromPayload } from "./protocol.ts";
import type { JsonObject, PayloadSelection, PendingCommit, RuntimeConfig, StatefulMode, StatefulSession } from "./types.ts";
import { asRecord, plural, sha1, shortId } from "./util.ts";

function makeDs4SessionId(instanceId: string, key: string): string {
	return `pi_${instanceId}_${sha1(key).slice(0, 12)}`;
}

function isPrefix(prefix: string[], full: string[]): boolean {
	if (prefix.length > full.length) return false;
	for (let i = 0; i < prefix.length; i++) {
		if (prefix[i] !== full[i]) return false;
	}
	return true;
}

function sessionKey(options?: SimpleStreamOptions): string {
	return options?.sessionId || "default";
}

function turnFocusText(): string {
	return "[DS4 turn focus: The next user text is the active task. Before any tool call or final answer, verify the topic/search terms come from the latest user message, not an earlier question. If your hidden plan mentions an older task, discard that plan and answer the latest user message directly. Do not restate earlier project summaries unless the latest user asks for a summary. For a short follow-up, answer that follow-up first.]";
}

function addFocusToUserContent(content: unknown): unknown {
	const focus = `${turnFocusText()}\n\n`;
	if (typeof content === "string") return `${focus}${content}`;
	if (Array.isArray(content)) return [{ type: "text", text: focus }, ...content];
	return focus;
}

function addTurnFocusToLatestUser(messages: unknown[]): unknown[] {
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messageRole(messages[i]) !== "user") continue;
		const msg = asRecord(messages[i]);
		if (!msg) return messages;
		const out = [...messages];
		out[i] = { ...msg, content: addFocusToUserContent(msg.content) };
		return out;
	}
	return messages;
}

export class StatefulSessionStore {
	private readonly instanceId = randomBytes(8).toString("hex");
	readonly sessions = new Map<string, StatefulSession>();

	lastMode: StatefulMode | "none" = "none";
	lastReason = "not used yet";
	lastDecisionSummary = "not used yet";
	lastRequestSummary = "not used yet";
	lastCommitSummary = "not committed yet";

	key(options?: SimpleStreamOptions): string {
		return sessionKey(options);
	}

	clear(reason = "state cleared by user"): void {
		this.sessions.clear();
		this.lastMode = "none";
		this.lastReason = reason;
		this.lastDecisionSummary = reason;
		this.lastRequestSummary = reason;
		this.lastCommitSummary = "not committed since reset";
	}

	choosePayload(payload: unknown, key: string, forceReset: boolean, config: RuntimeConfig): PayloadSelection {
		const p = payload && typeof payload === "object" ? { ...(payload as JsonObject) } : ({} as JsonObject);
		const fullMessages = Array.isArray(p.messages) ? [...p.messages] : [];
		const fullHashes = fullMessages.map((m) => sha1(m));
		const toolsHash = sha1(p.tools ?? []);
		let state = this.sessions.get(key);
		if (!state) {
			state = {
				ds4SessionId: makeDs4SessionId(this.instanceId, key),
				revision: 0,
				messageHashes: [],
				messages: [],
				toolsHash: "",
			};
			this.sessions.set(key, state);
		}

		let mode: StatefulMode = "reset";
		let reason = "initial/reset";
		let requestMessages = fullMessages;
		let parentRevision = 0;

		if (forceReset) {
			reason = "delta rejected by server; retrying full reset";
		} else if (!config.enabled) {
			reason = "extension disabled";
		} else if (state.revision > 0 && state.toolsHash === toolsHash && isPrefix(state.messageHashes, fullHashes)) {
			const appended = fullMessages.slice(state.messageHashes.length);
			if (appended.length > 0 && appended.every(isDeltaMessage)) {
				if (isToolOnlyDelta(appended) || (config.userTurnPolicy === "delta" && hasUserMessage(appended))) {
					mode = "delta";
					reason = hasUserMessage(appended)
						? `user-turn delta by policy (${appended.length} message${appended.length === 1 ? "" : "s"})`
						: `append-only delta (${appended.length} message${appended.length === 1 ? "" : "s"})`;
					requestMessages = appended;
					parentRevision = state.revision;
				} else if (hasUserMessage(appended)) {
					reason = config.userTurnPolicy === "reset"
						? `new user turn reset (${appended.length} appended message${appended.length === 1 ? "" : "s"})`
						: `auto reset for new user turn (${appended.length} appended message${appended.length === 1 ? "" : "s"})`;
				} else {
					reason = "append contained no tool/function delta messages";
				}
			} else if (appended.length === 0) {
				reason = "no appended messages";
			} else {
				reason = "append contained assistant/system replay";
			}
		} else if (state.revision > 0 && state.toolsHash !== toolsHash) {
			reason = "tool schema changed";
		} else if (state.revision > 0) {
			reason = "message prefix changed";
		}

		p.session_id = state.ds4SessionId;
		p.mode = mode;
		p.parent_revision = parentRevision;
		p.stateful_debug = {
			reason,
			full_messages: fullMessages.length,
			sent_messages: requestMessages.length,
			previous_messages: state.messageHashes.length,
			stored_revision: state.revision,
		};
		// Legacy top-level debug fields are retained while older ds4-server builds
		// are still in use. New servers prefer the stateful_debug object above.
		p.stateful_debug_reason = reason;
		p.stateful_debug_full_messages = fullMessages.length;
		p.stateful_debug_sent_messages = requestMessages.length;
		p.stateful_debug_previous_messages = state.messageHashes.length;
		p.stateful_debug_stored_revision = state.revision;
		const focusLatestUser = config.turnFocusEnabled && state.revision > 0 && hasUserMessage(requestMessages);
		if (focusLatestUser) {
			(p.stateful_debug as JsonObject).focus = "latest_user";
			p.stateful_debug_focus = "latest_user";
		}
		const outgoingMessages = focusLatestUser ? addTurnFocusToLatestUser(requestMessages) : requestMessages;
		p.messages = outgoingMessages;
		const debugInfo = requestInfoFromPayload(p);
		if (debugInfo) this.lastDecisionSummary = decisionSummary(debugInfo);
		if (debugInfo) this.lastRequestSummary = this.lastDecisionSummary;
		if (mode === "delta") {
			p.delta = { messages: outgoingMessages };
		} else {
			delete p.delta;
		}

		return {
			payload: p,
			mode,
			reason,
			pending: {
				key,
				mode,
				parentRevision,
				requestMessages,
				toolsHash,
				ds4SessionId: state.ds4SessionId,
			},
		};
	}

	commit(pending: PendingCommit | undefined, message: AssistantMessage): void {
		if (!pending) return;
		const assistant = assistantToOpenAI(message);
		const existing = this.sessions.get(pending.key);
		const previousMessages = pending.mode === "delta" && existing ? existing.messages : [];
		const messages = [...previousMessages, ...pending.requestMessages, assistant];
		const revision = pending.parentRevision + 1;
		this.sessions.set(pending.key, {
			ds4SessionId: pending.ds4SessionId,
			revision,
			messages,
			messageHashes: messages.map((m) => sha1(m)),
			toolsHash: pending.toolsHash,
		});
		this.lastMode = pending.mode;
		this.lastReason = `committed revision ${revision}`;
		this.lastCommitSummary = `session=${shortId(pending.ds4SessionId)} key=${pending.key} committed r${revision} via ${pending.mode}; appended ${plural(pending.requestMessages.length, "msg")} + assistant; stored ${plural(messages.length, "msg")}; stop=${message.stopReason}`;
	}

	sessionLines(limit = 5): string[] {
		if (this.sessions.size === 0) return ["sessions: none"];
		const lines = [`sessions: ${this.sessions.size}`];
		let shown = 0;
		for (const [key, state] of this.sessions) {
			if (shown >= limit) break;
			lines.push(
				`- ${key}: r${state.revision}, ${plural(state.messages.length, "stored msg")}, ds4=${shortId(state.ds4SessionId)}, tools=${state.toolsHash ? state.toolsHash.slice(0, 10) : "none"}`,
			);
			shown++;
		}
		if (this.sessions.size > shown) lines.push(`- ... ${this.sessions.size - shown} more`);
		return lines;
	}
}
