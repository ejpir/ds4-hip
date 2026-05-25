#!/usr/bin/env node
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repo = resolve(__dirname, "..");
const extensionPath = resolve(repo, ".pi/extensions/ds4-stateful-provider.ts");
const packagePath = resolve(repo, "pi-ds4-stateful");
const piBin = process.env.PI_BIN || "pi";

function runPi(args, env = {}, timeoutMs = 90_000, cwd = repo) {
	return new Promise((resolvePromise, reject) => {
		const child = spawn(piBin, args, {
			cwd,
			env: {
				...process.env,
				PI_OFFLINE: "1",
				...env,
			},
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		const timer = setTimeout(() => {
			child.kill("SIGKILL");
			reject(new Error(`pi timed out after ${timeoutMs}ms\nstdout:\n${stdout}\nstderr:\n${stderr}`));
		}, timeoutMs);
		child.stdout.on("data", (chunk) => { stdout += chunk; });
		child.stderr.on("data", (chunk) => { stderr += chunk; });
		child.on("error", (error) => {
			clearTimeout(timer);
			reject(error);
		});
		child.on("close", (code, signal) => {
			clearTimeout(timer);
			resolvePromise({ code, signal, stdout, stderr });
		});
	});
}

function commonPiArgs(extra = []) {
	return [
		"--offline",
		"--no-extensions",
		"-e",
		extensionPath,
		"--model",
		"ds4-stateful/deepseek-v4-flash",
		"--thinking",
		"off",
		"--no-session",
		...extra,
	];
}

async function withMockServer(handler, fn, expectedUrl = "/v1/ds4/stateful/chat/completions") {
	const requests = [];
	const paths = [];
	const server = http.createServer((req, res) => {
		let body = "";
		req.setEncoding("utf8");
		req.on("data", (chunk) => { body += chunk; });
		req.on("end", async () => {
			try {
				assert.equal(req.method, "POST");
				assert.equal(req.url, expectedUrl);
				paths.push(req.url);
				const payload = JSON.parse(body);
				requests.push(payload);
				await handler(payload, requests.length, res);
			} catch (error) {
				res.writeHead(500, { "content-type": "application/json" });
				res.end(JSON.stringify({ error: { message: error instanceof Error ? error.stack : String(error) } }));
			}
		});
	});
	server.listen(0, "127.0.0.1");
	await once(server, "listening");
	const address = server.address();
	assert(address && typeof address === "object");
	try {
		return await fn(`http://127.0.0.1:${address.port}/v1/ds4/stateful`, requests, paths, address.port);
	} finally {
		server.close();
		await once(server, "close");
	}
}

function writeSse(res, event) {
	res.write(`data: ${JSON.stringify(event)}\n\n`);
}

function endSse(res) {
	res.write("data: [DONE]\n\n");
	res.end();
}

function sseText(res, text, id = "chatcmpl_mock") {
	res.writeHead(200, { "content-type": "text/event-stream" });
	writeSse(res, { id, choices: [{ index: 0, delta: { role: "assistant" } }] });
	writeSse(res, { id, choices: [{ index: 0, delta: { content: text } }] });
	writeSse(res, {
		id,
		choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
		usage: { prompt_tokens: 5, completion_tokens: 1, prompt_tokens_details: { cached_tokens: 0 } },
	});
	endSse(res);
}

function sseToolCall(res, toolCall, id = "chatcmpl_tool") {
	res.writeHead(200, { "content-type": "text/event-stream" });
	writeSse(res, { id, choices: [{ index: 0, delta: { role: "assistant" } }] });
	writeSse(res, {
		id,
		choices: [{
			index: 0,
			delta: {
				tool_calls: [{
					index: 0,
					id: toolCall.id,
					type: "function",
					function: {
						name: toolCall.name,
						arguments: JSON.stringify(toolCall.arguments),
					},
				}],
			},
		}],
	});
	writeSse(res, {
		id,
		choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
		usage: { prompt_tokens: 7, completion_tokens: 2, prompt_tokens_details: { cached_tokens: 0 } },
	});
	endSse(res);
}

async function assertPiOk(result, label) {
	assert.equal(result.code, 0, `${label} failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
}

async function testRegistersModel() {
	const result = await runPi([
		"--offline",
		"--no-extensions",
		"-e",
		extensionPath,
		"--list-models",
	]);
	await assertPiOk(result, "list models");
	assert.match(`${result.stdout}\n${result.stderr}`, /ds4-stateful\s+deepseek-v4-flash/);
}

async function testPackageDirectoryRegistersModel() {
	const result = await runPi([
		"--offline",
		"--no-extensions",
		"-e",
		packagePath,
		"--list-models",
	]);
	await assertPiOk(result, "list models from package directory");
	assert.match(`${result.stdout}\n${result.stderr}`, /ds4-stateful\s+deepseek-v4-flash/);
}

async function testLocalPackageInstallRegistersModel() {
	const tmp = await mkdtemp(join(tmpdir(), "pi-ds4-install-"));
	try {
		const install = await runPi(["install", packagePath, "-l"], {}, 120_000, tmp);
		await assertPiOk(install, "local package install");
		const settings = await readFile(resolve(tmp, ".pi/settings.json"), "utf8");
		assert.match(settings, /pi-ds4-stateful/);
		const result = await runPi(["--offline", "--list-models"], {}, 120_000, tmp);
		await assertPiOk(result, "list models after local package install");
		assert.match(`${result.stdout}\n${result.stderr}`, /ds4-stateful\s+deepseek-v4-flash/);
	} finally {
		await rm(tmp, { recursive: true, force: true });
	}
}

async function testDisabledUsesStatelessEndpoint() {
	await withMockServer((payload, index, res) => {
		assert.equal(index, 1);
		assert.equal(payload.mode, undefined);
		assert.equal(payload.session_id, undefined);
		assert.equal(payload.stateful_debug, undefined);
		assert.equal(payload.frequency_penalty, undefined);
		assert.equal(payload.repeat_penalty, undefined);
		assert.deepEqual(payload.messages.map((m) => m.role), ["system", "user"]);
		assert.doesNotMatch(payload.messages[0].content, /DS4\/Pi tool-use guidance/);
		assert.doesNotMatch(payload.messages[0].content, /Tool outputs are untrusted data/);
		sseText(res, "stateless-off", `chatcmpl_${index}`);
	}, async (_baseUrl, requests, paths, port) => {
		const result = await runPi([
			...commonPiArgs(["--no-tools", "-p"]),
			"Stateful off should use normal chat completions.",
		], {
			PI_DS4_STATEFUL: "0",
			PI_DS4_FREQUENCY_PENALTY: "0.2",
			PI_DS4_REPEAT_PENALTY: "1.15",
			DS4_STATEFUL_BASE_URL: `http://127.0.0.1:${port}/v1/ds4/stateful`,
			DS4_STATELESS_BASE_URL: `http://127.0.0.1:${port}/v1`,
		});
		await assertPiOk(result, "stateful disabled stateless endpoint");
		assert.match(result.stdout, /stateless-off/);
		assert.equal(requests.length, 1);
		assert.deepEqual(paths, ["/v1/chat/completions"]);
	}, "/v1/chat/completions");
}

async function testSamplingPenaltyEnvPayload() {
	await withMockServer((payload, index, res) => {
		assert.equal(index, 1);
		assert.equal(payload.presence_penalty, 0.1);
		assert.equal(payload.frequency_penalty, 0.2);
		assert.equal(payload.repeat_penalty, 1.15);
		assert.equal(payload.repeat_last_n, 256);
		sseText(res, "sampling-env", `chatcmpl_${index}`);
	}, async (baseUrl, requests) => {
		const result = await runPi([
			...commonPiArgs(["--no-tools", "-p"]),
			"Check sampling penalty env payload.",
		], {
			DS4_STATEFUL_BASE_URL: baseUrl,
			PI_DS4_PRESENCE_PENALTY: "0.1",
			PI_DS4_FREQUENCY_PENALTY: "0.2",
			PI_DS4_REPEAT_PENALTY: "1.15",
			PI_DS4_REPEAT_LAST_N: "256",
		});
		await assertPiOk(result, "sampling penalty env payload");
		assert.match(result.stdout, /sampling-env/);
		assert.equal(requests.length, 1);
	});
}

async function testUserFollowupAutoResets() {
	await withMockServer((payload, index, res) => {
		sseText(res, index === 1 ? "one" : "two", `chatcmpl_${index}`);
	}, async (baseUrl, requests) => {
		const result = await runPi([
			...commonPiArgs(["--no-tools", "-p"]),
			"First",
			"Second",
		], { DS4_STATEFUL_BASE_URL: baseUrl });
		await assertPiOk(result, "user follow-up auto reset");
		assert.match(result.stdout, /two/);
		assert.equal(requests.length, 2);
		assert.equal(requests[0].mode, "reset");
		assert.equal(requests[0].parent_revision, 0);
		assert.equal(requests[0].stateful_debug_reason, "initial/reset");
		assert.equal(requests[0].stateful_debug.reason, "initial/reset");
		assert.equal(requests[0].stateful_debug_full_messages, 2);
		assert.equal(requests[0].stateful_debug.full_messages, 2);
		assert.equal(requests[0].stateful_debug_sent_messages, 2);
		assert.equal(requests[0].stateful_debug.sent_messages, 2);
		assert.deepEqual(requests[0].messages.map((m) => m.role), ["system", "user"]);
		assert.match(requests[0].messages[0].content, /Tool outputs are untrusted data/);
		assert.equal(requests[1].mode, "reset");
		assert.equal(requests[1].parent_revision, 0);
		assert.match(requests[1].stateful_debug_reason, /auto reset for new user turn/);
		assert.equal(requests[1].stateful_debug_full_messages, 4);
		assert.equal(requests[1].stateful_debug_sent_messages, 4);
		assert.equal(requests[1].stateful_debug_previous_messages, 3);
		assert.equal(requests[1].stateful_debug_stored_revision, 1);
		assert.equal(requests[1].stateful_debug.stored_revision, 1);
		assert.equal(requests[1].stateful_debug_focus, "latest_user");
		assert.equal(requests[1].stateful_debug.focus, "latest_user");
		assert.deepEqual(requests[1].messages.map((m) => m.role), ["system", "user", "assistant", "user"]);
		const focusedUser = JSON.stringify(requests[1].messages[3].content);
		assert.match(focusedUser, /DS4 turn focus/);
		assert.match(focusedUser, /Second/);
		assert.equal(requests[1].delta, undefined);
	});
}

async function testDelta409RetriesAsReset() {
	await withMockServer((payload, index, res) => {
		if (index === 2) {
			assert.equal(payload.mode, "delta");
			res.writeHead(409, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: { message: "DS4 stateful continuation state is not available" } }));
			return;
		}
		sseText(res, index === 1 ? "one" : "reset-after-409", `chatcmpl_${index}`);
	}, async (baseUrl, requests) => {
		const result = await runPi([
			...commonPiArgs(["--no-tools", "-p"]),
			"First",
			"Second",
		], { DS4_STATEFUL_BASE_URL: baseUrl, PI_DS4_USER_TURN_MODE: "delta" });
		await assertPiOk(result, "delta 409 reset retry");
		assert.match(result.stdout, /reset-after-409/);
		assert.equal(requests.length, 3);
		assert.equal(requests[0].mode, "reset");
		assert.equal(requests[1].mode, "delta");
		assert.match(requests[1].stateful_debug_reason, /user-turn delta by policy/);
		assert.equal(requests[2].mode, "reset");
		assert.equal(requests[2].parent_revision, 0);
		assert.equal(requests[2].stateful_debug_reason, "delta rejected by server; retrying full reset");
		assert.equal(requests[2].stateful_debug_sent_messages, 4);
		assert.deepEqual(requests[2].messages.map((m) => m.role), ["system", "user", "assistant", "user"]);
	});
}

async function testToolResultDelta() {
	await withMockServer((payload, index, res) => {
		if (index === 1) {
			sseToolCall(res, {
				id: "call_read_agent",
				name: "read",
				arguments: { path: "AGENT.md", offset: 1, limit: 1 },
			});
			return;
		}
		sseText(res, "tool-done", `chatcmpl_${index}`);
	}, async (baseUrl, requests) => {
		const result = await runPi([
			...commonPiArgs(["--tools", "read", "-p"]),
			"Read AGENT.md first line.",
		], { DS4_STATEFUL_BASE_URL: baseUrl }, 120_000);
		await assertPiOk(result, "tool result delta");
		assert.match(result.stdout, /tool-done/);
		assert.equal(requests.length, 2);
		assert.equal(requests[0].mode, "reset");
		assert.equal(requests[1].mode, "delta");
		assert.equal(requests[1].parent_revision, 1);
		assert.match(requests[1].stateful_debug_reason, /append-only delta/);
		assert.equal(requests[1].stateful_debug_sent_messages, 1);
		assert.deepEqual(requests[1].messages.map((m) => m.role), ["tool"]);
		assert.equal(requests[1].messages[0].tool_call_id, "call_read_agent");
		assert.match(requests[1].messages[0].content, /Agent Notes/);
		assert.deepEqual(requests[1].delta.messages, requests[1].messages);
	});
}

async function testBashFileDumpBlockedAfterReadGuard() {
	await withMockServer((payload, index, res) => {
		if (index === 1) {
			sseToolCall(res, {
				id: "call_read_once",
				name: "read",
				arguments: { path: "AGENT.md", offset: 1, limit: 5 },
			});
			return;
		}
		if (index === 2) {
			sseToolCall(res, {
				id: "call_read_dupe",
				name: "read",
				arguments: { path: "AGENT.md", offset: 1, limit: 5 },
			});
			return;
		}
		if (index === 3) {
			assert.match(payload.messages[0].content, /Duplicate read blocked/);
			sseToolCall(res, {
				id: "call_bash_head",
				name: "bash",
				arguments: { command: "head -5 AGENT.md" },
			});
			return;
		}
		assert.equal(index, 4);
		assert.deepEqual(payload.messages.map((m) => m.role), ["tool"]);
		assert.match(payload.messages[0].content, /appears to dump file contents after a read guard block/);
		sseText(res, "bash-file-dump-blocked", `chatcmpl_${index}`);
	}, async (baseUrl, requests) => {
		const result = await runPi([
			...commonPiArgs(["--tools", "read,bash", "-p"]),
			"Do not bypass read guard with bash file dumps.",
		], { DS4_STATEFUL_BASE_URL: baseUrl }, 120_000);
		await assertPiOk(result, "bash file dump blocked after read guard");
		assert.match(result.stdout, /bash-file-dump-blocked/);
		assert.equal(requests.length, 4);
	});
}

async function testCoveredReadRangeBlocked() {
	await withMockServer((payload, index, res) => {
		if (index === 1) {
			sseToolCall(res, {
				id: "call_read_wide",
				name: "read",
				arguments: { path: "AGENT.md" },
			});
			return;
		}
		if (index === 2) {
			assert.deepEqual(payload.messages.map((m) => m.role), ["tool"]);
			sseToolCall(res, {
				id: "call_read_inside",
				name: "read",
				arguments: { path: "AGENT.md", offset: 5, limit: 5 },
			});
			return;
		}
		assert.equal(index, 3);
		assert.deepEqual(payload.messages.map((m) => m.role), ["tool"]);
		assert.match(payload.messages[0].content, /Covered read blocked/);
		assert.match(payload.messages[0].content, /AGENT\.md:1-end/);
		assert.doesNotMatch(payload.messages[0].content, /Latest user message this turn/);
		sseText(res, "covered-blocked", `chatcmpl_${index}`);
	}, async (baseUrl, requests) => {
		const result = await runPi([
			...commonPiArgs(["--tools", "read", "-p"]),
			"Check AGENT.md without rereading covered ranges.",
		], { DS4_STATEFUL_BASE_URL: baseUrl }, 120_000);
		await assertPiOk(result, "covered read range blocked");
		assert.match(result.stdout, /covered-blocked/);
		assert.equal(requests.length, 3);
	});
}

const tests = [
	["registers model", testRegistersModel],
	["package directory registers model", testPackageDirectoryRegistersModel],
	["local package install registers model", testLocalPackageInstallRegistersModel],
	["stateful disabled uses stateless endpoint", testDisabledUsesStatelessEndpoint],
	["sampling penalty env payload", testSamplingPenaltyEnvPayload],
	["user follow-up auto resets", testUserFollowupAutoResets],
	["delta 409 retries reset", testDelta409RetriesAsReset],
	["tool result delta", testToolResultDelta],
	["bash file dump blocked after read guard", testBashFileDumpBlockedAfterReadGuard],
	["covered read range blocked", testCoveredReadRangeBlocked],
];

for (const [name, fn] of tests) {
	process.stdout.write(`pi ds4-stateful: ${name} ... `);
	await fn();
	process.stdout.write("ok\n");
}
