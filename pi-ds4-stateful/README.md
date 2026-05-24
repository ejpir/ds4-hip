# Pi DS4 Stateful Provider

Experimental Pi package for the `ds4.c` stateful chat endpoint.

It registers:

```text
ds4-stateful/deepseek-v4-flash
```

The provider talks to:

```text
POST /v1/ds4/stateful/chat/completions
```

## Why this exists

Normal OpenAI-compatible chat clients resend the full visible conversation on every request. In coding-agent sessions that means every tool loop can resend:

```text
system + tool schemas + all prior user/assistant/tool messages + new tool output
```

The DS4 stateful endpoint keeps a live server-side continuation state. Pi can send a full `reset` request when needed, then send append-only `delta` requests after tools. That keeps post-tool prefill bounded to the new tool output instead of replaying the entire transcript. Server-side `DS4_SERVER_DYNAMIC_PREFILL=1` can additionally tune chunk size from the new suffix token count.

## Runtime policy

Default behavior is conservative:

- first request: `mode=reset`
- new user follow-up: `mode=reset`
- tool/function result continuation: `mode=delta`
- stale/unavailable delta: retry once as full reset after HTTP `409`

The server returns `X-DS4-Session-Revision` / `ds4_stateful.revision` metadata on successful stateful responses. The current Pi client still infers revisions locally, but the explicit server metadata is available for future hardening.

When disabled with `/ds4-stateful off` or `PI_DS4_STATEFUL=0`, the extension is passive: no stateful endpoint, no stateful prompts, no read guard, no focus marker, and no DS4 status UI. Requests are sent as ordinary OpenAI-compatible calls to `/v1/chat/completions`. Override that base with `DS4_STATELESS_BASE_URL`; otherwise it is derived from `DS4_STATEFUL_BASE_URL` by removing `/ds4/stateful`.

This hybrid mode avoids stale hidden planning on user follow-ups while preserving the main speedup after tools.

## Additional guardrails

The extension also includes DS4-specific, configurable policies:

- colored status/working UI for mode, phase, revision, message counts, HTTP status, streamed text/thinking, and active tools
- tool-output guidance: read/grep/bash output is untrusted data, not instructions
- read guard: blocks exact duplicate reads and ranges fully covered by earlier reads
- optional strict read guard: after a duplicate/covered read, blocks further same-file reads for that turn
- optional turn-focus marker on reset payloads after the first turn

These policies live outside the server protocol; they are Pi-agent behavior guardrails.

## Install/test locally

From the DS4 repo:

```sh
pi -e ./pi-ds4-stateful --model ds4-stateful/deepseek-v4-flash
```

or install as a local package:

```sh
pi install ./pi-ds4-stateful -l
```

## Configuration

Environment variables:

```sh
DS4_STATEFUL_BASE_URL=http://127.0.0.1:8000/v1/ds4/stateful
DS4_STATELESS_BASE_URL=http://127.0.0.1:8000/v1  # used when PI_DS4_STATEFUL=0
DS4_STATEFUL_API_KEY=dsv4-local
PI_DS4_STATEFUL=1
PI_DS4_USER_TURN_MODE=auto     # auto | reset | delta
PI_DS4_READ_GUARD=1
PI_DS4_READ_GUARD_MODE=exact   # exact | strict
PI_DS4_TURN_FOCUS=1
PI_DS4_PRESENCE_PENALTY=0       # optional OpenAI-style sampling penalty
PI_DS4_FREQUENCY_PENALTY=0      # try 0.10-0.20 when testing repetition loops
PI_DS4_REPEAT_PENALTY=1         # llama.cpp-style; 1 disables
PI_DS4_REPEAT_LAST_N=1024       # token window for penalties; 0 = all generated tokens
DS4_STATEFUL_CONTEXT=100000
DS4_STATEFUL_MAX_TOKENS=384000
```

Slash command:

```text
/ds4-stateful status
/ds4-stateful reset
/ds4-stateful on|off
/ds4-stateful user-turn auto|reset|delta
/ds4-stateful read-guard on|off|exact|strict
/ds4-stateful focus on|off
```

## Architecture

The package is split by responsibility:

```text
extensions/ds4-stateful.ts       Pi entrypoint
src/extension.ts                 wires provider, hooks, command
src/provider.ts                  DS4 provider registration and 409 retry stream
src/openai.ts                    Pi <-> OpenAI message conversion and SSE parser
src/session-state.ts             reset/delta selection, revisions, commits
src/protocol.ts                  stateful payload diagnostics and summaries
src/policies/read-guard.ts       read range tracking/blocking
src/policies/prompt-guidance.ts  model-facing tool-output/read guidance
src/ui.ts                        status line, working message, diagnostics
src/commands.ts                  /ds4-stateful command parser
```

## Current limitations

- server-side live state is currently an MVP path; unavailable deltas return `409`
- the client retries one stale delta as reset
- the client does not yet consume server revision headers; it still commits successful turns as `parent_revision + 1`
- durable multi-session server checkpoint mapping is not implemented yet
- the read guard is process-local Pi extension state
