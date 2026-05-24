import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { COMMAND_USAGE, parseReadGuardMode, parseUserTurnPolicy } from "./config.ts";
import { ReadGuard } from "./policies/read-guard.ts";
import { StatefulSessionStore } from "./session-state.ts";
import type { RuntimeConfig } from "./types.ts";
import { StatefulUi } from "./ui.ts";

export function registerDs4StatefulCommand(
	pi: ExtensionAPI,
	config: RuntimeConfig,
	sessions: StatefulSessionStore,
	readGuard: ReadGuard,
	ui: StatefulUi,
): void {
	pi.registerCommand("ds4-stateful", {
		description: `Control the DS4 stateful provider. Usage: ${COMMAND_USAGE}`,
		handler: async (args, ctx) => {
			const mode = args.trim().toLowerCase();
			const parts = mode.split(/\s+/).filter(Boolean);
			if (parts[0] === "read-guard" || parts[0] === "guard") {
				if (parts[1] === "on") config.readGuardEnabled = true;
				else if (parts[1] === "off") config.readGuardEnabled = false;
				else if (parts[1] === "exact") {
					config.readGuardEnabled = true;
					config.readGuardMode = "exact";
				} else if (parts[1] === "strict") {
					config.readGuardEnabled = true;
					config.readGuardMode = "strict";
				} else if (parts.length > 1 && parts[1] !== "status") {
					ctx.ui.notify("Usage: /ds4-stateful read-guard [on|off|exact|strict|status]", "error");
					return;
				}
			} else if (parts[0] === "user-turn") {
				const policy = parseUserTurnPolicy(parts[1]);
				if (policy) config.userTurnPolicy = policy;
				else if (parts.length > 1 && parts[1] !== "status") {
					ctx.ui.notify("Usage: /ds4-stateful user-turn [auto|reset|delta|status]", "error");
					return;
				}
			} else if (parts[0] === "focus" || parts[0] === "turn-focus") {
				if (parts[1] === "on") config.turnFocusEnabled = true;
				else if (parts[1] === "off") config.turnFocusEnabled = false;
				else if (parts.length > 1 && parts[1] !== "status") {
					ctx.ui.notify("Usage: /ds4-stateful focus [on|off|status]", "error");
					return;
				}
			} else if (mode === "on") config.enabled = true;
			else if (mode === "off") {
				config.enabled = false;
				sessions.clear("stateful disabled by user");
				readGuard.clearAll("stateful disabled by user");
				ui.clearWorking(ctx);
			} else if (mode === "reset") {
				sessions.clear();
				readGuard.clearAll("state cleared by user");
				ui.markReset();
			} else if (mode !== "" && mode !== "status") {
				ctx.ui.notify(`Usage: ${COMMAND_USAGE}`, "error");
				return;
			}

			ui.applyStatus(ctx);
			ctx.ui.notify(ui.detailedStatus(ctx), "info");
		},
	});
}
