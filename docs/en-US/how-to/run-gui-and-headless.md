---
title: Run GUI and Headless Modes
---

# Run GUI and Headless Modes

Koharu can run as a normal desktop app or as a headless local server with a Web UI. Both modes use the same local runtime and HTTP server.

## What stays the same across modes

No matter how you launch Koharu, the runtime model is the same:

- the server binds to `127.0.0.1` by default (override with `--host`)
- the UI and API are served by the same local process
- the page pipeline, model loading, and exports use the same internal code paths

That is why desktop editing and headless automation stay aligned.

## Mode summary

| Mode | Desktop window | Local server | Typical use |
| --- | --- | --- | --- |
| Desktop | yes | yes | normal interactive editing |
| Headless | no | yes | local Web UI, scripting, automation |

## Run the desktop app

Launch Koharu normally from your installed application.

Even in desktop mode, Koharu still starts a local HTTP server internally. The embedded window talks to that local server instead of calling the pipeline directly.

This is the default mode and is the best choice for most users.

## Run headless mode

Headless mode starts the local server without opening the desktop GUI.

```bash
# macOS / Linux
koharu --port 4000 --headless

# Windows
koharu.exe --port 4000 --headless
```

After startup, open the Web UI at `http://localhost:4000`.

Headless mode stays in the foreground until you stop it, typically with `Ctrl+C`.

## Run with a fixed port

By default, Koharu uses a random local port. Use `--port` when you need a stable address for bookmarks, scripts, reverse proxies, or API clients.

```bash
# macOS / Linux
koharu --port 9999

# Windows
koharu.exe --port 9999
```

If you do not specify `--port`, Koharu still starts the server, but the chosen port is dynamic.

## Bind to a non-loopback address

By default the server binds to `127.0.0.1`, which means only the same machine can reach it. Pass `--host` to bind elsewhere.

```bash
koharu --host 0.0.0.0 --port 4000 --headless
```

This is useful in containers, VMs, or remote-development setups where the desktop client lives on a different host than the Koharu process. Anything other than `127.0.0.1` is a deliberate choice — there is no built-in authentication on the local API, so only set `--host` when you actually want non-loopback access and have your own access controls in place.

## Connect to the local API

When Koharu is running on a fixed port, the main endpoints are:

- Web UI: `http://localhost:9999/`
- RPC / HTTP API: `http://localhost:9999/api/v1`

Replace `9999` with the port you chose.

Because Koharu binds to loopback, these endpoints are local by default. If you want access from another machine, you need to expose that port yourself through your own network setup.

For endpoint-level details, see [HTTP API Reference](../reference/http-api.md).

## Force CPU mode

Use `--cpu` when you want to disable GPU inference explicitly.

```bash
# macOS / Linux
koharu --cpu

# Windows
koharu.exe --cpu
```

This is useful for compatibility testing, driver issues, or low-risk debugging when GPU setup is uncertain.

## Download runtime dependencies only

Use `--download` if you want Koharu to prefetch runtime dependencies and exit without starting the app.

```bash
# macOS / Linux
koharu --download

# Windows
koharu.exe --download
```

In the current implementation, this path initializes:

- runtime libraries used by the local inference stack
- the default vision and OCR models

It does not predownload every optional local translation LLM. Those are still fetched when you select them in Settings.

## Enable debug output

Use `--debug` when you want console-oriented startup with log output.

```bash
# macOS / Linux
koharu --debug

# Windows
koharu.exe --debug
```

On Windows, debug and headless runs also influence how Koharu attaches to or creates a console window.

## Credential storage

By default, Koharu stores API keys outside `config.toml`. macOS and Windows use the system keyring. Linux uses Koharu's local filesystem credential store under the app data directory with owner-only file permissions; this Linux store relies on filesystem permissions rather than OS-level encryption.

Headless and container runs use the same credential storage behavior as the desktop app. For containers, keep the app data directory on a persistent volume if you want saved API keys to survive container replacement.
