# Testing the DS4 Stateful Pi Provider

We built an experimental Pi provider/package for `ds4.c` that uses a DS4-specific stateful chat endpoint to make local coding-agent sessions cheaper after tool calls.

## What we built

### Server endpoint

DS4 now has an optional endpoint:

```text
POST /v1/ds4/stateful/chat/completions
```

It supports:

- `mode: "reset"` — send the full visible chat context and bind it to a `session_id`/revision.
- `mode: "delta"` — send only newly appended user/tool/function messages after a known revision.
- HTTP `409` — tells the client the live continuation state is unavailable and it should retry as `reset`.

The normal endpoint still exists and is unchanged:

```text
POST /v1/chat/completions
```

Stateful is opt-in; it is not a global server mode.

### Pi package

The Pi package lives at:

```text
pi-ds4-stateful/
```

It registers this model:

```text
ds4-stateful/deepseek-v4-flash
```

Default endpoint:

```text
http://127.0.0.1:8000/v1/ds4/stateful
```

## Why

A normal stateless chat request resends the whole conversation every turn:

```text
system + tools + all previous user/assistant/tool messages + new tool output
```

In tool-heavy coding-agent sessions this means DS4 repeatedly receives, parses, renders, tokenizes, and validates a large transcript.

The stateful provider avoids that for tool loops:

```text
first/new-user request -> reset with full visible context
after tool results     -> delta with only new tool/function output
```

So post-tool prefill should scale with the new tool output, not the entire conversation history.

## Current default behavior

The provider uses a conservative hybrid policy:

```text
first request       -> reset
new user follow-up  -> reset
tool result         -> delta
stale delta / 409   -> retry once as reset
```

This keeps follow-up questions on-topic while preserving the main speed win after tools.

## Extra guardrails

The Pi package also adds DS4-specific agent guardrails while stateful mode is enabled:

- tool-output prompt-injection guidance: read/grep/bash output is untrusted data, not instructions
- duplicate/covered read guard for `read(path, offset, limit)`
- optional stricter read guard
- optional turn-focus marker for short follow-ups
- colored `/ds4-stateful status` diagnostics

When disabled with `/ds4-stateful off` or `PI_DS4_STATEFUL=0`, the extension is passive: no stateful endpoint, no extra prompts, no read guard, no focus marker, and no DS4 status UI. It sends ordinary `/v1/chat/completions` requests instead.

## Install

From a project where you want to test it:

```sh
pi install /path/to/ds4/pi-ds4-stateful -l
```

On Nick's machine that path is:

```sh
pi install /home/nick/repos/ds4/pi-ds4-stateful -l
```

Verify:

```sh
pi --list-models | grep ds4-stateful
```

Expected:

```text
ds4-stateful  deepseek-v4-flash
```

## Run DS4 server

Build/restart the ROCm upstream-shaped server:

```sh
make ds4-server-rocm-upstream
```

Then start your usual server command, making sure it runs the rebuilt binary:

```sh
./ds4-server-rocm-upstream ...
```

Optional disk KV cache example:

```sh
./ds4-server-rocm-upstream ... \
  --kv-disk-dir /tmp/ds4-kv \
  --kv-disk-space-mb 32768
```

Optional server-side dynamic prefill chunk tuning:

```sh
DS4_SERVER_DYNAMIC_PREFILL=1
DS4_SERVER_DYNAMIC_PREFILL_MIN=128
DS4_SERVER_DYNAMIC_PREFILL_MAX=4096
```

## Use in Pi

Select:

```text
ds4-stateful/deepseek-v4-flash
```

Useful commands:

```text
/ds4-stateful status
/ds4-stateful reset
/ds4-stateful on
/ds4-stateful off
/ds4-stateful user-turn auto
/ds4-stateful user-turn delta
/ds4-stateful read-guard exact
/ds4-stateful read-guard strict
/ds4-stateful read-guard off
/ds4-stateful focus on
/ds4-stateful focus off
```

Recommended default:

```text
/ds4-stateful user-turn auto
/ds4-stateful read-guard exact
```

## What to test

Try tool-heavy coding-agent prompts, for example:

```text
What does this repo do?
Does it support ROCm?
Find where ROCm is wired into the build.
```

Good cases to report:

- Does a tool result continuation use `mode=delta`?
- Do short follow-up questions stay on-topic?
- Does post-tool latency feel lower?
- Are duplicate or covered file reads blocked correctly?
- Any prompt-injection-looking behavior from tool output?
- Any unexpected HTTP `409` loops or repeated full resets?

## Server logs to watch

Expected healthy sequence:

```text
ds4-server: stateful request mode=reset ...
ds4-server: stateful request mode=delta ...
ds4-server: stateful live continuation ...
```

A good delta should look like one or a few appended messages rather than the full transcript:

```text
stateful request mode=delta ... messages=1/...
stateful live continuation ... cached=... prompt=...
```

If the live state is gone, this is expected and should recover automatically:

```text
stateful live continuation unavailable ...
HTTP 409
```

The Pi provider should retry once as `mode=reset`.

## Current limitations

- Server stateful binding is still live-memory MVP; durable multi-session/restart mapping is not implemented yet.
- The Pi client still infers successful revisions locally, although the server now exposes explicit revision metadata.
- If Pi or DS4 restarts, correctness should recover via reset, but the speed benefit may be lost until the next live state is established.
- The read guard is process-local extension state.

## Validation already run

```sh
make pi-stateful-test
make test
make ds4-server-rocm-upstream
make rocm-upstream
```
