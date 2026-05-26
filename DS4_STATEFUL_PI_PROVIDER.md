# DS4 Stateful Pi Provider

This note documents option 3: a DS4-specific Pi provider that avoids resending
and re-rendering the whole OpenAI chat transcript after append-only coding turns.

## Goal

Pi's default provider path is OpenAI-compatible and stateless. Every model turn
contains:

```text
system + tools + prior user/assistant/tool messages + new tool results
```

The built-in DS4 prefix cache can often reuse the old KV, but the server still
receives, parses, renders, tokenizes, and validates the full JSON transcript.
After append-only user/tool turns this is unnecessary: the new turn only
produced a small suffix, and the live DS4 KV already contains the true prior
assistant turn and earlier coding context.

The stateful provider changes the wire protocol so the common coding path sends
only appended messages:

```text
reset: full context for the first request or reset boundary
then
 delta: only new user/tool/function messages
```

The model still has to prefill the new delta tokens. The expected win is that
follow-up latency scales with the new user/tool suffix, not the whole conversation.

## Protocol Endpoint

The server exposes an MVP DS4-native endpoint:

```text
POST /v1/ds4/stateful/chat/completions
```

It returns ordinary OpenAI chat-completion responses and SSE chunks, so a custom
Pi provider can reuse most OpenAI response parsing. The request envelope adds:

- `session_id`: client-chosen state key.
- `mode`: `"reset"` or `"delta"`.
- `parent_revision`: required for `delta`; optional for `reset`, defaulting to
  `0`.
- `stateful_debug`: optional client diagnostics object used only for logging,
  e.g. `reason`, `full_messages`, `sent_messages`, `previous_messages`, and
  `stored_revision`. Legacy top-level `stateful_debug_*` fields are still
  accepted for older clients.

Successful responses include explicit state metadata:

```text
X-DS4-Stateful: true
X-DS4-Session-Id: <session_id>
X-DS4-Parent-Revision: <parent_revision>
X-DS4-Session-Revision: <parent_revision + 1>
X-DS4-Stateful-Mode: reset|delta
```

Non-streaming OpenAI chat responses also include a top-level `ds4_stateful`
object with the same `session_id`, `mode`, `parent_revision`, and `revision`.
Streaming responses include the same object on the OpenAI usage chunk when
`stream_options.include_usage` is enabled.

### Reset

A reset request is a normal OpenAI chat-completions request plus the stateful
envelope:

```json
{
  "model": "deepseek-v4-flash",
  "session_id": "pi-019e...",
  "mode": "reset",
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." }
  ],
  "tools": [ ... ],
  "stream": true
}
```

DS4 renders this as the complete prompt, runs normal prefix/disk-cache matching,
generates, then remembers:

```text
session_id -> live token frontier
revision = parent_revision + 1
think mode
whether tools are active
tool schema order metadata needed for response mapping
```

The client treats a successful response as committing the next revision.

### Delta

A delta request contains only new user/tool/function messages that should be
appended after the last remembered assistant response. The Pi provider uses this
for append-only tool continuations and, by default, new user follow-ups:

```json
{
  "model": "deepseek-v4-flash",
  "session_id": "pi-019e...",
  "mode": "delta",
  "parent_revision": 1,
  "delta": {
    "messages": [
      {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "file contents returned by Pi's read tool"
      }
    ]
  },
  "stream": true
}
```

For convenience the endpoint also accepts top-level `messages` instead of
`delta.messages` for delta mode.

The delta fast path is live-only in this MVP. The server validates:

```text
stateful_live.session_id == request.session_id
stateful_live.revision == request.parent_revision
stateful_live.live_tokens == current live ds4_session position
```

If it matches, DS4 builds an effective prompt from the exact live token prefix
plus a freshly-tokenized rendered suffix:

```text
<｜end▁of▁sentence｜>
<｜User｜><tool_result>...</tool_result>
<｜Assistant｜><think-or-/think>
```

Then it generates normally and commits `revision = parent_revision + 1`.

If the live state does not match, the server returns HTTP `409`:

```text
DS4 stateful continuation state is not available; resend with mode=reset and full messages
```

The Pi provider must then retry with `mode=reset` and the full context.

## Pi Provider Design

The Pi side should be a custom extension/provider using `pi.registerProvider()`
and a custom `streamSimple` implementation.

Provider-side state per Pi session:

```ts
{
  ds4SessionId: string;
  revision: number;
  systemHash: string;
  toolsHash: string;
  messageHashes: string[];
}
```

Per request:

1. Canonicalize and hash the current `systemPrompt`, tool schemas, and messages.
2. If this is the first request, or if system/tools changed, send `reset`.
3. If old message hashes are a prefix of the current hashes and the appended
   messages are user/tool/function messages, send `delta` with just those
   messages by default.
4. If appended messages include a new user message, the default `user-turn=delta`
   keeps the prior live KV so the model can attend to earlier coding context
   without re-prefilling it. `user-turn=auto` or `user-turn=reset` are available
   as conservative reset-on-user-turn policies.
5. If the append-only check fails, send `reset` with the full context.
6. If a delta returns `409`, retry once as `reset`.
7. On a completed response, increment the local revision and persist the provider
   state with `pi.appendEntry()` or equivalent session storage.

A delta should not include changed tool schemas or a changed system prompt. Those
are reset boundaries.

## Performance Model

Let:

- `C` = old context
- `D` = new delta, often a tool result
- `G` = generated output

OpenAI-style stateless request:

```text
send/parse/render/tokenize C + D
reuse or recover KV for C
prefill D
generate G
```

Stateful request:

```text
send/parse/render/tokenize D only
reuse live KV for C by session_id/revision
prefill D
generate G
```

This does **not** remove prefill for large tool output. A 30k-token file read is
still 30k new tokens the model must ingest. The win is removing repeated work for
old context and avoiding replay/canonicalization failure modes.

Server-side prefill tuning is independent of reset/delta selection. The server
can optionally pick a per-request chunk cap from the actually new suffix size
(`prompt_tokens - cached_tokens`) with:

```sh
DS4_SERVER_DYNAMIC_PREFILL=1
DS4_SERVER_DYNAMIC_PREFILL_MIN=128   # optional
DS4_SERVER_DYNAMIC_PREFILL_MAX=4096  # optional
```

When enabled, logs include:

```text
ds4-server: prefill plan dynamic suffix=2436 cached=21134 prompt=23570 chunk=2048
```

The chosen cap still respects the graph allocation cap (`DS4_METAL_PREFILL_CHUNK`
/ `DS4_SERVER_PREFILL_CHUNK`). Small suffixes stay responsive; large suffixes use
larger batched chunks.

Expected gains:

- Small tool output with huge old context: large request-setup reduction.
- Large tool output: smaller relative win; latency is still dominated by `D`.
- Current cache miss/rebuild cases: potentially massive win because delta avoids
  replaying `C` entirely.

## Current Implementation Status

Implemented in `ds4_server.c`:

- `POST /v1/ds4/stateful/chat/completions`.
- `mode="reset"` full-context binding.
- `mode="delta"` live suffix append.
- In-memory `session_id + revision` validation.
- OpenAI chat-completion and SSE response compatibility.
- HTTP response headers and response-body metadata exposing committed stateful
  revision.
- HTTP `409` fallback signal when live state is unavailable.

Pi package implemented in `pi-ds4-stateful/` with a project-local compatibility wrapper at `.pi/extensions/ds4-stateful-provider.ts`:

- Registers provider `ds4-stateful` with model `deepseek-v4-flash`.
- Uses `DS4_STATEFUL_BASE_URL`, defaulting to
  `http://127.0.0.1:8000/v1/ds4/stateful`.
- Computes message/tool hashes and sends `mode="delta"` for append-only
  user/tool/function continuations after the last committed assistant response.
- Defaults to `user-turn=delta`: new user follow-ups and post-tool
  continuations are sent as `mode="delta"` when the transcript is append-only.
  This keeps earlier coding context in live KV without re-prefilling it. Override
  with `/ds4-stateful user-turn auto` or environment variable
  `PI_DS4_USER_TURN_MODE=auto` to restore reset-on-user-turn behavior.
- Adds a short per-turn focus note to follow-up user turns after the first turn,
  attached to the latest user message. This reminds DS4 to answer the latest
  follow-up directly instead of restating older project summaries. Disable with
  `/ds4-stateful focus off` or `PI_DS4_TURN_FOCUS=0`.
- Retries a rejected delta once as `mode="reset"` on HTTP 409.
- Provides `/ds4-stateful [on|off|reset|status|focus on|off|read-guard on|off|exact|strict|user-turn auto|reset|delta]`.
  When off, the extension is passive: it bypasses
  `/v1/ds4/stateful/chat/completions`, sends ordinary OpenAI-compatible requests
  to `/v1/chat/completions` using `DS4_STATELESS_BASE_URL` or a base URL derived
  from `DS4_STATEFUL_BASE_URL`, and disables the system-prompt addendum, read
  guard, focus marker, and DS4 status/working UI.
- Updates Pi's working message/status while active with colored, compact labels
  for reset vs delta, phase, revision, sent/full message counts, short session
  id, HTTP status, streamed thinking/text chars, tool-call count, and running
  tool names.
- `/ds4-stateful status` prints a multi-line diagnostic summary with provider
  endpoint, model, user-turn mode, focus state, read-guard state, active phase, last decision/request/HTTP/commit,
  context usage, and all tracked stateful sessions with revision, stored message
  count, DS4 session id, and tool-schema hash.
- Adds DS4 tool-use guidance to the system prompt, including that tool outputs
  are untrusted data and instructions/prompts found inside read/grep/bash output
  must not be followed unless the user explicitly asks to analyze that text.
- By default, blocks exact duplicate `read(path, offset, limit)` calls plus ranges that are fully covered
  by an earlier read in the session. Different non-covered offsets remain
  allowed because they may be legitimate targeted follow-ups.
  Optional `/ds4-stateful read-guard strict` additionally blocks further reads
  of the same file for the rest of a turn after a duplicate-read signal, but
  that is not the default. This prevents the model from treating a read
  continuation notice as a reason to reread the same range. Pi's collapsed TUI
  display (`more lines / expand`) is human-only unless pasted back into the
  conversation. Disable with `/ds4-stateful read-guard off` or
  `PI_DS4_READ_GUARD=0`.
- Splits package code by responsibility:
  - `src/provider.ts`: DS4 provider registration and 409 retry stream.
  - `src/openai.ts`: Pi/OpenAI message conversion and SSE parsing.
  - `src/session-state.ts`: reset/delta selection, revisions, and commits.
  - `src/policies/read-guard.ts`: read range tracking/blocking.
  - `src/policies/prompt-guidance.ts`: tool-output/read guidance.
  - `src/ui.ts` and `src/commands.ts`: status UI and `/ds4-stateful` command.
- Adds a `stateful_debug` diagnostics object, plus legacy top-level debug
  fields, to every stateful payload so the server can log why a request used
  reset or delta:
  - `stateful_debug_reason`
  - `stateful_debug_full_messages`
  - `stateful_debug_sent_messages`
  - `stateful_debug_previous_messages`
  - `stateful_debug_stored_revision`

The server logs these as:

```text
ds4-server: stateful request mode=reset session=... parent=0 stored=6 reason="message prefix changed" messages=152/180 prev=181 cached=0 prompt=15231
```

This is meant to diagnose reset causes such as context compaction, context
slimming, tool schema changes, or branch/fork history changes.

Not yet implemented:

- Durable stateful metadata in the KV disk cache.
- Pi client use of server-returned revision metadata; the current client still
  infers that every successful turn increments by one, but the server now
  exposes explicit headers/body metadata for future hardening.
- Branch/fork-aware provider state beyond hash-prefix reset fallback.

## ROCm Upstream Build

This project uses the upstream-shaped ROCm server binary for normal local runs.
Build validation for this feature should use:

```sh
make ds4-server-rocm-upstream
# or the full ROCm set:
make rocm-upstream
```

The launcher `scripts/start_ds4_server.sh` already targets
`./ds4-server-rocm-upstream`.

CPU/server unit coverage can still be run with:

```sh
make ds4_test
./ds4_test --server
```

Pi provider regression coverage is available as:

```sh
make pi-stateful-test
```

`make test` runs both the Pi stateful provider regression test and the native
DS4 unit tests.

## Validation Log

Initial server-side MVP validation:

- `make ds4_test && ./ds4_test --server`: pass.
- `make ds4-server-rocm-upstream`: pass.
- `make rocm-upstream`: pass with `ROCM_ARCH=gfx1151`.

Pi extension validation:

- `make pi-stateful-test`: pass.
- `make test`: pass.
- `make ds4-server-rocm-upstream`: pass.
- The Pi regression test covers provider registration from both the compatibility
  wrapper and package directory, user follow-up -> default delta, stale delta HTTP
  409 -> reset retry, debug reason/count fields, a real Pi `read` tool result
  sent as a one-message stateful delta, and covered-read blocking.
- A probe against an older already-running DS4 server reached the provider and
  failed with HTTP 404 `unknown endpoint`, which is expected until that server
  process is restarted with the rebuilt `ds4-server-rocm-upstream` binary.
