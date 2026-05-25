import type { ReadGuardMode } from "../types.ts";
import { numberField } from "../util.ts";
import { textOfContent } from "../openai.ts";

interface ReadArgs {
	path?: string;
	file_path?: string;
	offset?: number;
	limit?: number;
}

interface SeenReadRange {
	key: string;
	path: string;
	offset: number;
	limit: number | "all";
	label: string;
	count: number;
	nextOffset?: number;
}

export interface ReadGuardDecision {
	block: true;
	reason: string;
}

function asReadArgs(input: unknown): ReadArgs | undefined {
	return input && typeof input === "object" && !Array.isArray(input) ? (input as ReadArgs) : undefined;
}

function readPath(input: unknown): string | undefined {
	const r = asReadArgs(input);
	const path = r?.path ?? r?.file_path;
	return typeof path === "string" && path.length > 0 ? path : undefined;
}

function readRangeLabel(path: string, offset: number, limit: number | "all"): string {
	return `${path}:${offset}${limit === "all" ? "-end" : `-${offset + limit - 1}`}`;
}

function readRange(input: unknown): SeenReadRange | undefined {
	const r = asReadArgs(input);
	const path = readPath(input);
	if (!path) return undefined;
	const offset = Math.max(1, numberField(r?.offset) ?? 1);
	const limitValue = numberField(r?.limit);
	const limit = limitValue && limitValue > 0 ? limitValue : "all";
	const key = `${path}:${offset}:${limit}`;
	const label = readRangeLabel(path, offset, limit);
	return { key, path, offset, limit, label, count: 0 };
}

function readNextOffset(content: unknown): number | undefined {
	const text = textOfContent(content);
	const match = text.match(/Use offset=(\d+) to continue/);
	return match ? Number.parseInt(match[1], 10) : undefined;
}

function readRangeEnd(range: SeenReadRange): number {
	return range.limit === "all" ? Number.POSITIVE_INFINITY : range.offset + range.limit - 1;
}

export class ReadGuard {
	private readonly seenReadRanges = new Map<string, SeenReadRange>();
	private readonly blockedReadPathsThisTurn = new Map<string, number>();
	lastSummary = "no read guard blocks yet";

	get seenCount(): number {
		return this.seenReadRanges.size;
	}

	get blockedThisTurnCount(): number {
		let total = 0;
		for (const count of this.blockedReadPathsThisTurn.values()) total += count;
		return total;
	}

	hasBlockedReadsThisTurn(): boolean {
		return this.blockedThisTurnCount > 0;
	}

	clearAll(summary = "no read guard blocks yet"): void {
		this.seenReadRanges.clear();
		this.blockedReadPathsThisTurn.clear();
		this.lastSummary = summary;
	}

	beginTurn(): void {
		this.blockedReadPathsThisTurn.clear();
	}

	rememberRead(input: unknown, content: unknown): void {
		const range = readRange(input);
		if (!range) return;
		const existing = this.seenReadRanges.get(range.key);
		const nextOffset = readNextOffset(content) ?? existing?.nextOffset;
		const actualLimit = range.limit === "all" && nextOffset && nextOffset > range.offset
			? nextOffset - range.offset
			: range.limit;
		this.seenReadRanges.set(range.key, {
			...range,
			limit: actualLimit,
			label: readRangeLabel(range.path, range.offset, actualLimit),
			count: (existing?.count ?? 0) + 1,
			nextOffset,
		});
	}

	clearReadsForPath(input: unknown): void {
		const path = readPath(input);
		if (!path) return;
		for (const [key, range] of this.seenReadRanges) {
			if (range.path === path) this.seenReadRanges.delete(key);
		}
	}

	checkRead(input: unknown, mode: ReadGuardMode): ReadGuardDecision | undefined {
		const range = readRange(input);
		if (!range) return undefined;
		const seen = this.seenReadRanges.get(range.key);
		if (seen) return { block: true, reason: this.duplicateReadReason(seen) };
		const coveredBy = this.coveringReadRange(range);
		if (coveredBy) return { block: true, reason: this.coveredReadReason(range, coveredBy) };
		if (mode === "strict" && (this.blockedReadPathsThisTurn.get(range.path) ?? 0) > 0) {
			return { block: true, reason: this.followupReadBlockedReason(range) };
		}
		return undefined;
	}

	private readRangesForPath(path: string): SeenReadRange[] {
		return [...this.seenReadRanges.values()]
			.filter((range) => range.path === path)
			.sort((a, b) => a.offset - b.offset);
	}

	private coveringReadRange(range: SeenReadRange): SeenReadRange | undefined {
		const end = readRangeEnd(range);
		return this.readRangesForPath(range.path).find((seen) => seen.offset <= range.offset && readRangeEnd(seen) >= end);
	}

	private seenReadRangesSummary(path: string): string {
		const ranges = this.readRangesForPath(path);
		if (ranges.length === 0) return "none";
		const labels = ranges.slice(0, 8).map((range) => {
			const end = range.limit === "all" ? "end" : String(range.offset + range.limit - 1);
			return `${range.offset}-${end}`;
		});
		return `${labels.join(", ")}${ranges.length > labels.length ? `, ... +${ranges.length - labels.length}` : ""}`;
	}

	private bumpBlocked(path: string): number {
		const count = (this.blockedReadPathsThisTurn.get(path) ?? 0) + 1;
		this.blockedReadPathsThisTurn.set(path, count);
		return count;
	}

	private duplicateReadReason(range: SeenReadRange): string {
		const count = this.bumpBlocked(range.path);
		this.lastSummary = `blocked duplicate read ${range.label} (turn blocks for path=${count})`;
		return `Duplicate read blocked: ${range.label} was already read and is still in model context. Stop reading this range and answer from the existing context unless one very specific missing fact is required. If needed, use grep/rg or an explicitly targeted unread range instead of rereading. Seen ranges for this file: ${this.seenReadRangesSummary(range.path)}.`;
	}

	private coveredReadReason(range: SeenReadRange, coveredBy: SeenReadRange): string {
		const count = this.bumpBlocked(range.path);
		this.lastSummary = `blocked covered read ${range.label} covered by ${coveredBy.label} (turn blocks for path=${count})`;
		return `Covered read blocked: ${range.label} is already covered by earlier read ${coveredBy.label}, which is still in model context. Stop reading this file range and answer from existing context unless a different precise fact is needed. If needed, use grep/rg or an unread targeted range instead of splitting/rereading already-seen ranges. Seen ranges for this file: ${this.seenReadRangesSummary(range.path)}.`;
	}

	private followupReadBlockedReason(range: SeenReadRange): string {
		const count = this.bumpBlocked(range.path);
		this.lastSummary = `strict-blocked follow-up read ${range.label} after duplicate/read-loop signal (turn blocks for path=${count})`;
		return `Further read blocked by strict read guard for ${range.label}: an earlier duplicate-read attempt on this file indicates a read loop. Stop reading this file for the current turn and answer the latest user message from existing context. If the needed fact is not in context, use grep/rg for that exact fact instead of sequential README paging. Seen ranges for this file: ${this.seenReadRangesSummary(range.path)}.`;
	}
}
