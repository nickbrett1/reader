# reader

A reader project generated with genproj

## Capabilities

This project includes the following capabilities:

- **Doppler Secrets Management**: Integrates Doppler for secure secrets management. Enables the various MCP servers that rely on privileged tokens to access their services (e.g. Buildkite, GitHub, SonarQube).
- **AI Coding Agents**: Sets up the AI coding agents in the devcontainer: goose (config, MCP servers and spec-first recipes) plus the Antigravity CLI.
- **Container Agent**: Every generated devcontainer brings up and registers its own a2a-goose agent (`<repo>-dev`), reached over the tailnet by the LiteLLM proxy; reuses the a2a-goose GitHub release channel, so the container has the same self-update path as a host.
- **Docker**: Adds Docker support for containerised builds and tooling.
- **Python DevContainer**: Sets up a VS Code DevContainer with Python environment.
- **Docker Container**: Containerize the project and publish to the GitHub Container Registry (GHCR) for deployment to a NAS or self-hosted host via Docker Compose. Mutually exclusive with other deployment systems.
- **Buildkite Integration**: Runs CI on a self-hosted Buildkite agent (Apple silicon) instead of a metered cloud fleet. The pipeline and its GitHub webhook are created during generation, so there is no manual "set up project" step. It can coexist with an existing CI provider, so a repository can migrate without a flag day.
- **Ruff (Python code quality)**: Adds fast, zero-configuration Python linting with Ruff (rules live in pyproject.toml [tool.ruff]). Lint locally with `ruff check`. Requires a Python devcontainer. The generated CI pipeline also runs `ruff check`.
- **Dependabot**: Configures Dependabot for automated dependency updates.

## Setup

1. Clone the repository
2. Create a virtualenv and install the package with dev extras:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the checks:

   ```bash
   ruff check src tests
   pytest -v
   ```

## Corpus scope

The corpus is video-game books only. A staging tree can hold unrelated books
without re-ingesting them: list a directory or file per line in
`.ingestignore` at the root of `READER_INGEST` (blank lines and `#` comments
are ignored; a bare name matches that path component anywhere).

```
# /volume1/homes/nick/Books/.ingestignore
Data Visualization
Machine Learning, AI and Bots
```

To remove books that were already ingested, use the `prune` verb. It deletes
the book, pages, chunks, images and orphaned concepts from SQLite, unlinks the
image blobs, then prunes and compacts the vector index. Rehearse first:

```bash
reader prune --books 4-38 --dry-run   # prints the matching work keys
reader prune --books 4-38 --yes       # ids or ranges; or --work-keys a,b
```

`--delete-sources` also unlinks each book's file under `READER_INGEST` (only
useful when the staging tree is writable; the deployed container mounts it
read-only).

## Doppler

This project uses Doppler for secrets from the shared `common` project
(config `dev`) — no per-repo Doppler project is created. First use (links
the shared project and `dev` config):

```bash
doppler setup --project common --config dev
```

If your repo needs app-specific secrets that shouldn't live in the shared
`common` project, regenerate it with the doppler capability set to
`projectStrategy: "new"` to get a dedicated project.

The Doppler CLI is installed in the devcontainer — it must be on PATH for the
VS Code extension and `doppler run` to work. Auth is persisted via the host
`~/.doppler` bind-mount.

### Env-var precedence (read this if `doppler run` hits the wrong project)

Doppler resolves its target as **environment variables > `doppler.yaml` >
`~/.doppler` scoped config**. If your shell — or the session that launched
the devcontainer (e.g. an agent runtime) — exports `DOPPLER_PROJECT` /
`DOPPLER_CONFIG` / `DOPPLER_ENVIRONMENT`, those silently override this
repo's `doppler.yaml` and every `doppler` command targets the wrong
project. The devcontainer's post-create setup pins this repo's context
(`common`/`dev`) in `~/.bashrc` and `~/.zshrc` and warns at
setup if resolution still mismatches. To force the correct context manually:

```bash
unset DOPPLER_PROJECT DOPPLER_CONFIG DOPPLER_ENVIRONMENT
doppler setup --no-interactive --project common --config dev
```

## Deployment

See `deploy/README.md` for the deployment runbook (Buildkite -> GHCR ->
Watchtower -> Docker host). Deploy with:

```bash
docker compose up -d
```

## The container's agent

This devcontainer brings up its own `a2a-goose` agent, registered in the hub as
`reader-dev` - one agent per repo, so a restart reclaims the same entry
instead of adding a second one. Turns are billed through the LiteLLM proxy
configured in Doppler (`LITELLM_BASE_URL`).

```bash
scripts/agent-dev.sh start    # write secrets + config, fetch the launcher, run it
scripts/agent-dev.sh status   # running or not, the card URL, the log tail
scripts/agent-dev.sh stop     # SIGTERM, wait for a clean deregister, confirm gone
```

`start` runs from the devcontainer's post-start hook, so the agent is normally
already up when you arrive. It fails open: with no network on a first start it
prints why it did not start and leaves the project usable. Secrets come from
Doppler into `~/.config/a2a-goose/env` (mode 0600) and never into the image or
`containerEnv`.

## Generated by genproj

This project was generated using the genproj tool.
