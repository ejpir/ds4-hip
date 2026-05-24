import { createHash } from "node:crypto";
import type { JsonObject } from "./types.ts";

export function asRecord(value: unknown): JsonObject | undefined {
	return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonObject) : undefined;
}

export function intField(value: unknown, fallback = -1): number {
	return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : fallback;
}

export function stringField(value: unknown, fallback = ""): string {
	return typeof value === "string" ? value : fallback;
}

export function numberField(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : undefined;
}

export function plural(n: number, word: string): string {
	return `${n} ${word}${n === 1 ? "" : "s"}`;
}

export function chars(n: number): string {
	if (n < 1000) return String(n);
	if (n < 1000000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}k`;
	return `${(n / 1000000).toFixed(1)}m`;
}

export function shortId(value: string, head = 18, tail = 8): string {
	if (value.length <= head + tail + 1) return value;
	return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

function stable(value: unknown): unknown {
	if (value === undefined) return null;
	if (value === null || typeof value !== "object") return value;
	if (Array.isArray(value)) return value.map(stable);
	const out: JsonObject = {};
	for (const key of Object.keys(value as JsonObject).sort()) {
		const v = (value as JsonObject)[key];
		if (v !== undefined) out[key] = stable(v);
	}
	return out;
}

export function stableJson(value: unknown): string {
	return JSON.stringify(stable(value));
}

export function sha1(value: unknown): string {
	return createHash("sha1").update(stableJson(value)).digest("hex");
}
