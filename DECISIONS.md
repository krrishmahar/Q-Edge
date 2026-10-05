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

## Algorithm
- **Ancilla is qubit 0 (least-significant bit).** With `|c> (x) |+>`, a cyclic decrement over
  all `n + 1` qubits and a Hadamard on the ancilla, the ancilla-|1> amplitudes equal
  `(c_i - c_{i+1}) / 2`; multiplying by `2 * norm` recovers the raw difference.
- **Sign in shot mode.** Measurement only yields `|c_i - c_{i+1}|`; this is sufficient because
  the edge magnitude is `sqrt(h^2 + v^2)`.
- **Wrap-around terms are discarded.** The cyclic shift also produces row-crossing and
  last-to-first differences; these are masked out after the circuit.

## Tiling
- **Stride `T - 1` with a 1-pixel halo.** Each `T x T` tile yields a `(T - 1) x (T - 1)` core;
  cores tile the image exactly, so stitching is a reshape and the result equals the untiled
  forward-difference gradient (verified by tests). The image is edge-padded on the bottom and
  right, so the last row and column use edge replication.
- **Normalisation.** The stitched magnitude is divided by its global maximum, then thresholded.
- **4K measurement (wave 2).** A synthetic 3840x2160 image with 8x8 tiles (169,641 tiles) runs
  end to end through the fast path in about 1.7 s on the 8-core development machine, single
  process.
