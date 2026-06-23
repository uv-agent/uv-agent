from __future__ import annotations

try:
    import pluggy
except ImportError:  # pragma: no cover - pluggy is a project dependency in normal use
    pluggy = None  # type: ignore[assignment]

if pluggy is not None:
    hookspec = pluggy.HookspecMarker("uv_agent")
else:
    def hookspec(func):  # type: ignore[no-untyped-def]
        return func


@hookspec
async def uv_agent_start(context):
    """Start a uv-agent plugin."""


@hookspec
async def uv_agent_stop(context):
    """Stop a uv-agent plugin."""


@hookspec
async def uv_agent_prepare_turn(context, request):
    """Return additive pre-user context blocks for the current turn."""
