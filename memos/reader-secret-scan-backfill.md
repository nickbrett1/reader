# Reader — gitleaks secret-scan backfill

Backfill of the genproj secret-scan gate into `nickbrett1/reader`, following
`nickbrett1/genproj` commit `6558d97` (`src/generator/templates/*.template`).

Date: 2026-10-18
Status: complete — local hook installed, CI step added, scan exits 0.

## Files added (verbatim from the templates)

| genproj template | reader path | mode |
| --- | --- | --- |
| `githooks-pre-commit.template` | `.githooks/pre-commit` | 100755 |
| `gitleaks-toml.template` | `.gitleaks.toml` | 100644 |
| `gitleaksignore.template` | `.gitleaksignore` | 100644 |
| `scripts-install-gitleaks.sh.template` | `scripts/install-gitleaks.sh` | 100755 |
| `scripts-setup-git-hooks.sh.template` | `scripts/setup-git-hooks.sh` | 100755 |

Contents were copied as rendered file bodies (not the template filenames), and
the exec bit was applied to the hook and both scripts.

## Hook installation

```
$ bash scripts/setup-git-hooks.sh
INFO: Downloading gitleaks 8.30.1 (linux_arm64)...
INFO: Installed gitleaks 8.30.1 at /home/vscode/.local/bin/gitleaks
OK: git hooks installed (core.hooksPath=.githooks).

$ git config --get core.hooksPath
.githooks
```

Verified: `core.hooksPath` is exactly `.githooks`, and gitleaks 8.30.1 is on
disk at the pinned path.

## Legacy hook plumbing removed

There was **no `package.json`** in this repo, so there was no `simple-git-hooks`
config, `prepare` script or devDependency to remove. The only reference was the
devcontainer step, which previously ran:

```
(cd /workspaces/reader && npx --yes simple-git-hooks) || echo "WARN: ..."
```

It is now:

```
(cd /workspaces/reader && bash scripts/setup-git-hooks.sh) || exit 1
```

`lint-staged` is retained: the new `.githooks/pre-commit` invokes
`node_modules/.bin/lint-staged` when it is present, so behaviour is preserved
where a Node toolchain exists.

## CI

`.buildkite/pipeline.yml` gained a native (not docker-plugin) step:

```
bash scripts/install-gitleaks.sh
"$HOME/.local/bin/gitleaks" git --redact --verbose .
```

It runs on the `mac-studio-linux` queue before the build, so the full-history
scan is the gate that cannot be skipped by `--no-verify`.

## Scan result — and an important discrepancy

Final scan:

```
$ "$HOME/.local/bin/gitleaks" git --redact --verbose .
INF 35 commits scanned.
INF scanned ~403612 bytes (403.61 KB)
INF no leaks found
$ echo $?
0
```

**Historical findings: 0. Suppressed fingerprints: 0.**

This differs from the expectation in the task ("it will report the old leaked
key in history"). It does not, and the reason is specific, not a configuration
problem:

* The historical leak is the LiteLLM master key in
  `scripts/watch_embed_pro.sh`, added in `f3b68a0` and removed in `91807c8`.
* In that file the value was written in shell **indirection** form:
  `export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-<key>}"`.
* gitleaks 8.30.1's default `generic-api-key` rule does **not** match that
  `${VAR:-value}` form. It does match the plain assignment form.

Demonstrated with two files in one commit:

```
File:        plain.sh        <-- export X="sk-..."          => generic-api-key finding
(indirect.sh)                <-- export X="${X:-sk-...}"    => no finding
```

Consequences:

* There is no fingerprint to add to `.gitleaksignore`. Adding one would not
  suppress anything (gitleaks never emits it), and inventing a fingerprint
  would create a permanent, unexplained blind-spot entry in a reviewable file.
  `.gitleaksignore` was therefore left as the pristine template.
* No history rewrite and no broad allowlist was performed, per instructions.
* The key was already rotated by the owner on 2026-10-18, so exposure is
  bounded regardless of the scan result.

If we want the gate to catch this shape in future, the correct fix is a narrow
custom rule in `.gitleaks.toml` (or an upstream rule), not a suppression — that
is deliberately out of scope for this verbatim backfill and should be a
separate, reviewed change.

## Verification

* `ruff check src tests` — All checks passed.
* `pytest -q` — 91 passed.
* `git config --get core.hooksPath` — `.githooks`.
* Full-history scan — exit 0.
