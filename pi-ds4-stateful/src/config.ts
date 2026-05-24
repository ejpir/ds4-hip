import type { ReadGuardMode, RuntimeConfig, UserTurnPolicy } from "./types.ts";

export const PROVIDER_ID = "ds4-stateful";
export const MODEL_ID = "deepseek-v4-flash";
export const STATUS_KEY = "ds4-stateful";
export const API_ID = "ds4-stateful-chat-completions";

export const COMMAND_USAGE =
	"/ds4-stateful [on|off|reset|status|focus on|off|read-guard on|off|exact|strict|user-turn auto|reset|delta]";

function envInt(name: string, fallback: number, min: number, max: number): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const n = Number.parseInt(raw, 10);
	if (!Number.isFinite(n)) return fallback;
	return Math.max(min, Math.min(max, n));
}

export function parseUserTurnPolicy(value: unknown): UserTurnPolicy | undefined {
	return value === "auto" || value === "reset" || value === "delta" ? value : undefined;
}

function envUserTurnPolicy(): UserTurnPolicy {
	return parseUserTurnPolicy(process.env.PI_DS4_USER_TURN_MODE) ??
		parseUserTurnPolicy(process.env.DS4_STATEFUL_USER_TURN_MODE) ??
		"auto";
}

export function parseReadGuardMode(value: unknown): ReadGuardMode | undefined {
	return value === "exact" || value === "strict" ? value : undefined;
}

function envReadGuardMode(): ReadGuardMode {
	return parseReadGuardMode(process.env.PI_DS4_READ_GUARD_MODE) ??
		parseReadGuardMode(process.env.DS4_STATEFUL_READ_GUARD_MODE) ??
		"exact";
}

function deriveStatelessBaseUrl(statefulBaseUrl: string): string {
	const trimmed = statefulBaseUrl.replace(/\/+$/, "");
	return trimmed.endsWith("/ds4/stateful")
		? trimmed.slice(0, -"/ds4/stateful".length)
		: trimmed;
}

export function loadRuntimeConfig(): RuntimeConfig {
	const baseUrl = process.env.DS4_STATEFUL_BASE_URL || "http://127.0.0.1:8000/v1/ds4/stateful";
	return {
		baseUrl,
		statelessBaseUrl: process.env.DS4_STATELESS_BASE_URL || process.env.DS4_BASE_URL || deriveStatelessBaseUrl(baseUrl),
		contextWindow: envInt("DS4_STATEFUL_CONTEXT", 100000, 1024, 1048576),
		maxTokens: envInt("DS4_STATEFUL_MAX_TOKENS", 384000, 1, 1048576),
		enabled: process.env.PI_DS4_STATEFUL !== "0",
		readGuardEnabled: process.env.PI_DS4_READ_GUARD !== "0",
		readGuardMode: envReadGuardMode(),
		turnFocusEnabled: process.env.PI_DS4_TURN_FOCUS !== "0",
		userTurnPolicy: envUserTurnPolicy(),
	};
}
