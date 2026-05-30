import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { loadRuntimeConfig, PROVIDER_ID } from "./config.ts";
import { registerDs4StatefulCommand } from "./commands.ts";
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
	const ui = new StatefulUi(config, sessions, readGuard);
	const isActiveDs4StatefulContext = (ctx: ExtensionContext) =>
		config.enabled && isDs4StatefulContext(ctx);

	pi.registerProvider(PROVIDER_ID, ds4StatefulProviderConfig(config, sessions, (progress) => ui.prefillProgress(progress)));

	pi.on("session_start", (_event, ctx) => {
		readGuard.clearAll();
		ui.applyStatus(ctx);
	});
	pi.on("model_select", (_event, ctx) => ui.applyStatus(ctx));
	pi.on("session_shutdown", (_event, ctx) => ui.clearWorking(ctx));

	pi.on("before_agent_start", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		readGuard.beginTurn();
		ui.beginAgent(ctx);
		return { systemPrompt: appendToolUseGuidance(event.systemPrompt) };
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
		if (event.toolName === "bash" && config.readGuardEnabled) {
			const bashBlock = checkBashFileReadFallback(input, readGuard.hasBlockedReadsThisTurn());
			if (bashBlock) return bashBlock;
		}
		if (event.toolName === "edit" || event.toolName === "write") {
			readGuard.clearReadsForPath(input);
			return;
		}
		if (!config.readGuardEnabled || event.toolName !== "read") return;
		return readGuard.checkRead(input, config.readGuardMode);
	});

	pi.on("tool_result", (event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx) || !config.readGuardEnabled || event.toolName !== "read" || event.isError) return;
		readGuard.rememberRead((event as any).input, event.content);
	});

	pi.on("agent_end", (_event, ctx) => {
		if (!isActiveDs4StatefulContext(ctx)) return;
		ui.clearWorking(ctx);
	});

	registerDs4StatefulCommand(pi, config, sessions, readGuard, ui);
}
