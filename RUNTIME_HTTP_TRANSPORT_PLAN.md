# Runtime HTTP Transport Implementation Plan

Planning reference: `docs/pending-runtime-http-transport.md` (intentionally ignored by git via `docs/pending-*.md`). This tracked checklist records the implementation scope for the runtime HTTP transport work.

## Target shape

- Keep one lightweight loopback RPC server alive for the `PythonRunner` / process lifetime.
- Create a fresh `RunSession` and bearer token for each `run_python` execution.
- Move runtime structured events from stdout JSON lines to JSON-RPC notifications over HTTP.
- Add runtime-to-host request/response calls through `call_host(name, **kwargs)`.
- Keep stdout and stderr as pure user output.
- Use Python stdlib HTTP pieces; do not add Starlette, uvicorn, or httpx for v1.

## Checklist

- [x] Add host-side RPC server, dispatcher, method registry, auth, and per-run session registry.
- [x] Add runtime stdlib HTTP transport and expose `call_host`.
- [x] Wire `PythonRunner` to the long-lived server and per-run sessions/tokens.
- [x] Remove stdout structured-event parsing from runner output handling.
- [x] Update runtime, runner, and RPC tests for the new transport behavior.
- [x] Update tracked docs describing runtime structured events and environment variables.
- [ ] Run full test suite and commit the finished implementation.
