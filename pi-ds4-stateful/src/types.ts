import type { AssistantMessage } from "@earendil-works/pi-ai";

export type JsonObject = Record<string, unknown>;
export type StatefulMode = "reset" | "delta";
export type UserTurnPolicy = "auto" | "reset" | "delta";
export type ReadGuardMode = "exact" | "strict";
export type UiPhase = "idle" | "request" | "response" | "stream" | "tool-call" | "tools" | "done" | "retry" | "error";

export interface PrefillProgress {
	phase: string;
	current: number;
	total: number;
	percent: number;
	chunkTps: number;
	avgTps: number;
	elapsed: number;
	timestamp: number;
}

export interface RuntimeConfig {
	baseUrl: string;
	statelessBaseUrl: string;
	contextWindow: number;
	maxTokens: number;
	enabled: boolean;
	readGuardEnabled: boolean;
	readGuardMode: ReadGuardMode;
	turnFocusEnabled: boolean;
	userTurnPolicy: UserTurnPolicy;
	presencePenalty: number;
	frequencyPenalty: number;
	repeatPenalty: number;
	repeatLastN: number;
	coachEnabled: boolean;
	coachMaxTokens: number;
	coachTimeoutMs: number;
}

export interface StatefulSession {
	ds4SessionId: string;
	revision: number;
	messageHashes: string[];
	messages: unknown[];
	toolsHash: string;
}

export interface PendingCommit {
	key: string;
	mode: StatefulMode;
	parentRevision: number;
	requestMessages: unknown[];
	toolsHash: string;
	ds4SessionId: string;
}

export interface StatefulUiRequest {
	mode: StatefulMode;
	sessionId: string;
	reason: string;
	parentRevision: number;
	storedRevision: number;
	fullMessages: number;
	sentMessages: number;
	previousMessages: number;
	httpStatus?: number;
	startedAt: number;
}

export interface PayloadSelection {
	payload: unknown;
	pending?: PendingCommit;
	reason: string;
	mode: StatefulMode;
}

export type AssistantCommitMessage = AssistantMessage;
