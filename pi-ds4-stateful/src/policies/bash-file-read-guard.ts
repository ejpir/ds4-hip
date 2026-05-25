interface BashArgs {
	command?: string;
}

export interface BashFileReadGuardDecision {
	block: true;
	reason: string;
}

const FILE_DUMP_COMMANDS = new Set(["cat", "head", "tail", "sed", "awk"]);
const SCRIPT_READ_COMMANDS = new Set(["python", "python3", "perl", "node"]);

function asBashArgs(input: unknown): BashArgs | undefined {
	return input && typeof input === "object" && !Array.isArray(input) ? (input as BashArgs) : undefined;
}

function shellWords(text: string): string[] {
	const words: string[] = [];
	let cur = "";
	let quote: "'" | '"' | undefined;
	let escaped = false;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (escaped) {
			cur += ch;
			escaped = false;
			continue;
		}
		if (ch === "\\" && quote !== "'") {
			escaped = true;
			continue;
		}
		if (quote) {
			if (ch === quote) quote = undefined;
			else cur += ch;
			continue;
		}
		if (ch === "'" || ch === '"') {
			quote = ch;
			continue;
		}
		if (/\s/.test(ch)) {
			if (cur) {
				words.push(cur);
				cur = "";
			}
			continue;
		}
		cur += ch;
	}
	if (cur) words.push(cur);
	return words;
}

function commandSegments(command: string): string[] {
	return command
		.split(/\|\||&&|[;\n]/)
		.flatMap((part) => part.split("|"))
		.map((part) => part.trim())
		.filter(Boolean);
}

function stripAssignments(words: string[]): string[] {
	let i = 0;
	while (i < words.length && /^[A-Za-z_][A-Za-z0-9_]*=.*/.test(words[i])) i++;
	return words.slice(i);
}

function stripWrappers(words: string[]): string[] {
	let out = stripAssignments(words);
	while (out[0] === "command" || out[0] === "builtin" || out[0] === "sudo") out = out.slice(1);
	if (out[0] === "env") out = stripAssignments(out.slice(1));
	return out;
}

function hasInputRedirection(words: string[]): boolean {
	return words.some((word) => word === "<" || /^<[^(<]/.test(word));
}

function nonOptionArgs(words: string[]): string[] {
	const args: string[] = [];
	for (let i = 0; i < words.length; i++) {
		const word = words[i];
		if (word === "--") {
			args.push(...words.slice(i + 1));
			break;
		}
		if (word === "<") {
			i++;
			continue;
		}
		if (word.startsWith("<")) continue;
		if (word.startsWith("-")) {
			if ((word === "-n" || word === "-c" || word === "-F" || word === "-f") && i + 1 < words.length) i++;
			continue;
		}
		args.push(word);
	}
	return args;
}

function looksLikeCatHeadTailDump(cmd: string, args: string[]): boolean {
	if (hasInputRedirection(args)) return true;
	const positional = nonOptionArgs(args);
	if (cmd === "cat") return positional.length > 0;
	return positional.length > 0;
}

function hasSedInputFile(args: string[]): boolean {
	if (hasInputRedirection(args)) return true;
	let sawScript = false;
	for (let i = 0; i < args.length; i++) {
		const word = args[i];
		if (word === "--") return args.length > i + 1;
		if (word === "<" || word.startsWith("<")) return true;
		if (word === "-e" || word === "-f") {
			i++;
			sawScript = true;
			continue;
		}
		if (word.startsWith("-e") || word.startsWith("-f")) {
			sawScript = true;
			continue;
		}
		if (word.startsWith("-")) continue;
		if (!sawScript) {
			sawScript = true;
			continue;
		}
		return true;
	}
	return false;
}

function hasAwkInputFile(args: string[]): boolean {
	if (hasInputRedirection(args)) return true;
	let sawProgram = false;
	for (let i = 0; i < args.length; i++) {
		const word = args[i];
		if (word === "--") return args.length > i + 1;
		if (word === "<" || word.startsWith("<")) return true;
		if (word === "-F" || word === "-v" || word === "-f") {
			i++;
			if (word === "-f") sawProgram = true;
			continue;
		}
		if (word.startsWith("-F") || word.startsWith("-v") || word.startsWith("-f")) {
			if (word.startsWith("-f")) sawProgram = true;
			continue;
		}
		if (word.startsWith("-")) continue;
		if (!sawProgram) {
			sawProgram = true;
			continue;
		}
		return true;
	}
	return false;
}

function looksLikeScriptFileRead(segment: string): boolean {
	return /\b(readFileSync|readFile|read_text|open\s*\(|Path\s*\([^)]*\)\.read_text)\b/.test(segment);
}

export function bashFileReadFallbackReason(input: unknown): string | undefined {
	const command = asBashArgs(input)?.command;
	if (typeof command !== "string" || command.trim().length === 0) return undefined;
	for (const segment of commandSegments(command)) {
		const words = stripWrappers(shellWords(segment));
		const cmd = words[0];
		if (!cmd) continue;
		const base = cmd.split("/").pop() ?? cmd;
		const args = words.slice(1);
		if (FILE_DUMP_COMMANDS.has(base)) {
			const dumpsFile = base === "sed" ? hasSedInputFile(args) :
				base === "awk" ? hasAwkInputFile(args) :
				looksLikeCatHeadTailDump(base, args);
			if (dumpsFile) return `bash command '${base}' appears to dump file contents`;
		}
		if (SCRIPT_READ_COMMANDS.has(base) && looksLikeScriptFileRead(segment)) {
			return `bash command '${base}' appears to read file contents`;
		}
	}
	return undefined;
}

export function checkBashFileReadFallback(input: unknown, afterReadGuardBlock: boolean): BashFileReadGuardDecision | undefined {
	if (!afterReadGuardBlock) return undefined;
	const reason = bashFileReadFallbackReason(input);
	if (!reason) return undefined;
	return {
		block: true,
		reason: `${reason} after a read guard block. Do not bypass the read guard with cat/head/tail/sed/awk or scripts; answer from existing context, use rg/grep for a precise search, or request one targeted unread read range.`,
	};
}
