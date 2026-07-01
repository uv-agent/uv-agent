from __future__ import annotations

from uv_agent.plugins import PluginManifest, SetupPlugin


MANIFEST = PluginManifest(
    id="builtin.scheduler",
    version="0.1.0",
    display_name="Scheduler",
    description="Persistent schedule management runtime namespace.",
    builtin=True,
    priority=400,
    dependencies=("builtin.workflow",),
    capabilities=("runtime_namespace",),
)


def plugin() -> SetupPlugin:
    return SetupPlugin(manifest=MANIFEST, setup=setup)


def setup(context) -> None:
    context.runtime.register_namespace(
        "scheduler",
        doc="Persistent schedule management helpers.",
        transport="local_module",
        module="uv_agent_runtime.scheduler",
    )
