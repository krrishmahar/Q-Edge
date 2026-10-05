# Design Decisions

Defaults chosen where the brief was ambiguous. Each entry: decision, then rationale.

## Tooling
- **Nested git repository.** The project lives in `del/`, which sits inside an unrelated parent
  repository owned by another user. `git init` in `del/` creates an independent repository and
  leaves the parent untouched.
- **uv installed with the official Astral installer** (no pip). The virtualenv is created with
  `uv venv`; every command runs through `uv run`.
- **Task runner: `justfile`.** `just` is installed with `uv tool install rust-just`. `make` is not
  available on the Windows host.
- **Configuration uses frozen dataclasses**, not pydantic, to avoid an extra dependency.
