---
name: api-design
description: >-
  Design, review, or refactor Kirara's public API and internal runtime
  boundaries. Use when changing model integration, inputs, scheduling, state,
  compilation, attention, sampling, sharding, or package ownership.
---

# Kirara API Design

Keep the package-root API limited to `LLM` and `SamplingParams`. Put
extensibility behind a few runtime boundaries rather than adding public config
objects for each feature.

## Workflow

1. Read `README.md`, `kirara/api.py`, and the affected internal packages.
2. Identify which existing boundary owns the behavior before adding a type.
3. Update implementation, exports, tests, and README together.
4. Run `uv run python -m unittest discover -s tests -v`.

## Python API Rules

- Use PEP 585 built-in generics and absolute Kirara imports.
- Reject invalid public input with specific exceptions.
- Do not catch broad exceptions to guess calling conventions.
- Accept repository strings and compatible model instances through `LLM`.
- Resolve `model_impl` through `ModelRegistry`; do not branch on integration
  names in `LLM`.
- Keep runtime choices in `LLM` and request choices in `SamplingParams`.
- Pass Qwix configuration through unchanged; do not invent a Kirara
  quantization ontology.

## Runtime Boundaries

- `InputProcessor` owns text and structured multimodal normalization.
- `ModelLoader` finishes loading during initialization.
- `Adapter` exposes capabilities and one `Adapter(Batch, State)` call.
- `Scheduler` decides what work occurs and must not call `jax.jit`.
- `StateManager` owns allocation, slots, page tables, and state recycling.
- `Runner` owns every compiled executable and bounded bucket mapping.
- `AttentionBackend` owns interchangeable attention computation.
- `Sampler` converts logits to tokens with request-specific parameters.
- sharding uses JAX `Mesh`, logical axes, `PartitionSpec`, and
  `NamedSharding`; do not introduce worker/rank executor abstractions.

## Correctness Invariants

- Keep paged KV as the sole KV source of truth.
- Preserve immutable request slot IDs and exact greedy behavior when
  `temperature=0`.
- Padding and inactive slots must not write state, advance positions, or
  select logits.
- Runtime sequence lengths, block tables, physical IDs, and occupancy must
  not define new executable identities.
- Do not reconstruct complete dense KV sequences or call models per row.
- Prefill and decode are execution patterns, not separate adapter methods.

## Scope Discipline

- Add MaxText, Qwix, Pallas, prefix caching, and other integrations only when
  their implementation is real.
- Fail explicitly when a requested integration cannot support a supplied
  option.
- Standardize runtime boundaries, not neural-network architectures.
