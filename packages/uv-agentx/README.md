# uv-agentx

`uv-agentx` is a small launcher for running `uv-agent` with installable plugins
through `uv tool run` / `uvx`. It keeps plugin startup ephemeral while avoiding
long command lines such as:

```powershell
uvx --with uv-agent-auth-code --with uv-agent-remote-control uv-agent@latest
```

## Usage

```powershell
uvx uv-agentx@latest --latest -p auth-code -p remote-control -- daemon --replace
```

Launcher options appear before `--`; everything after `--` is passed to
`uv-agent`.

```powershell
uvx uv-agentx@latest -p auth-code -- --log-level DEBUG
uvx uv-agentx@latest -p remote-control -- workflow-node --level high "do thing"
```

## Plugins

`-p` / `--plugin` accepts uv-agent plugin short names. For a short name,
`uv-agentx` first checks the official PyPI project `uv-agent-<name>`. If that
project does not exist, it checks the original package name and falls back with a
short stderr message.

```powershell
uvx uv-agentx@latest -p auth-code
```

This normally resolves to:

```powershell
uv tool run --with uv-agent-auth-code uv-agent
```

Use `--raw-plugin` to pass a package requirement through without `uv-agent-`
name expansion:

```powershell
uvx uv-agentx@latest --raw-plugin company-plugin
```

## Latest

`--latest` runs the host as `uv-agent@latest`. It refreshes only unpinned plugins
and plugins explicitly marked with `@latest`.

```powershell
uvx uv-agentx@latest --latest -p auth-code
uvx uv-agentx@latest -p auth-code@latest
uvx uv-agentx@latest --latest -p auth-code==1.2.0
```

`@latest` on a plugin is a launcher syntax. It is translated to a normal
`--with` requirement plus `--refresh-package` because uv does not support
`--with package@latest`.

## Dry Run

Use `--dry-run` to print the `uv tool run` command without executing it:

```powershell
uvx uv-agentx@latest --dry-run --latest -p auth-code -- daemon --replace
```

`--dry-run` still checks official PyPI so the printed command reflects short-name
fallback decisions.

## Notes

- `uv-agentx` only checks official PyPI for package-name existence.
- PyPI checks use a short timeout and an in-process cache.
- Network lookup failures print a warning and let `uv` resolve the package.
- The launcher is intentionally standard-library only.
