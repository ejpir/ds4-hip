import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { loadRuntimeConfig, PROVIDER_ID } from "./config.ts";
import { registerDs4StatefulCommand } from "./commands.ts";
import { Ds4ToolCoach, formatCoachNote } from "./coach.ts";
import { ds4StatefulProviderConfig } from "./provider.ts";
import { isDs4StatefulContext } from "./protocol.ts";
import { checkBashFileReadFallback } from "./policies/bash-file-read-guard.ts";
import { appendToolUseGuidance } from "./policies/prompt-guidance.ts";
import { ReadGuard } from "./policies/read-guard.ts";
import { StatefulSessionStore } from "./session-state.ts";
import { StatefulUi } from "./ui.ts";

export default function registerDs4StatefulExtension(pi: ExtensionAPI) {
	const config = loadRuntimeConfig();
	const sessions = new StatefulSessionStore();
	const readGuard = new ReadGuard();
	const coach = new Ds4ToolCoach(config);
	const ui = new StatefulUi(config, sessions, readGuard, coach);
	const isActiveDs4StatefulContext = (ctx: ExtensionContext) =>
		config.enabled && isDs4StatefulContext(ctx);

	pi.registerProvider(PROVIDER_ID, ds4StatefulProviderConfig(config, sessions, (progress) => ui.prefillProgress(progress)));

	pi.on("session_start", (_event, ctx) => {
		readGuard.clearAll();
		coach.clear();
		ui.applyStatus(ctx);
	});
	pi.on("model_select", (_event, ctx) => ui.applyStatus(ctx));
	pi.on("session_shutdown", (_event, ctx) => ui.clearWorking(ctx));

	pi.on("before_agent_start", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		readGuard.beginTurn();
		ui.beginAgent(ctx);
		return { systemPrompt: config.coachEnabled ? event.systemPrompt : appendToolUseGuidance(event.systemPrompt) };
	});

	pi.on("before_provider_request", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.beforeProviderRequest(event.payload, ctx);
	});

	pi.on("after_provider_response", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.afterProviderResponse(event.status, event.headers, ctx);
	});

	pi.on("message_update", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.messageUpdate(event.assistantMessageEvent, ctx);
	});

	pi.on("message_end", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.messageEnd(event.message, ctx);
	});

	pi.on("tool_execution_start", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.toolExecutionStart(event.toolCallId, event.toolName, ctx);
	});

	pi.on("tool_execution_end", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.toolExecutionEnd(event.toolCallId, ctx);
	});

	pi.on("tool_call", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		const input = (event as any).input;
		if (config.coachEnabled) return;
		if (event.toolName === "bash" && config.readGuardEnabled) {
			const bashBlock = checkBashFileReadFallback(input, readGuard.hasBlockedReadsThisTurn());
			if (bashBlock) return bashBlock;
		}
		if (event.toolName === "edit" || event.toolName === "write") {
			readGuard.clearReadsForPath(input);
			return;
		}
		if (!config.readGuardEnabled || event.toolName !== "read") return;
		const readBlock = readGuard.checkRead(input, config.readGuardMode);
		if (readBlock) return readBlock;
		readGuard.markReadRequested(input);
	});

	pi.on("tool_result", async (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		if (!config.coachEnabled && config.readGuardEnabled && event.toolName === "read") {
			readGuard.finishRead((event as any).input, event.content, event.isError);
		}
		const advice = await coach.reviewToolResult(event, ctx);
		if (!advice) return;
		return {
			content: [...event.content, { type: "text" as const, text: formatCoachNote(advice) }],
		};
	});

	pi.on("agent_end", (_event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.clearWorking(ctx);
	});

	registerDs4StatefulCommand(pi, config, sessions, readGuard, ui, coach);
}
