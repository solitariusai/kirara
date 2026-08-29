# Kirara

Kirara is a JAX-native inference runtime. It owns scheduling, persistent model
state, bounded compilation, paged attention, sampling, and JAX sharding while
model implementations remain in TakTiny, MaxText, or custom integrations.

## Public API

```python
from kirara import LLM, SamplingParams

llm = LLM(
    "HuggingFaceTB/SmolLM2-135M",
    dtype="bfloat16",
    mesh={"tp": 4},
)

outputs = llm.generate(
    ["Hello!", "Explain JAX briefly."],
    SamplingParams(temperature=0.7, max_tokens=64),
)
```

Qwix rules are applied by the selected model integration while weights load:

```python
import jax.numpy as jnp
import qwix

rule = qwix.QuantizationRule(weight_qtype=jnp.int8)
llm = LLM("HuggingFaceTB/SmolLM2-135M", quantization=rule)
```

Kirara passes Qwix objects through unchanged. TakTiny performs the actual
weight transformation and also recognizes its Qwix dtype shortcuts such as
`"int8"` and `"int4"`. Existing model instances must already be quantized.

An existing model instance is also accepted when a registered integration can
adapt it. An object that already implements Kirara's adapter contract is
recognized directly and does not need to inherit from a Kirara class.

```python
llm = LLM(model_instance)
embeddings = llm.encode(["hello", "world"])
```

Only `LLM` and `SamplingParams` are exported from the package root. Model
loading, processing, scheduling, state allocation, and JIT bucket selection
remain internal.

Shared runtime annotations live in `kirara.types`. Domain types stay beside
their owners—for example input parts in `inputs`, adapter contracts in
`models`, and attention metadata in `attention`—so integrations can import
precise contracts without expanding the package-root API.

## Implemented runtime boundaries

- registry-driven TakTiny and custom-model integrations;
- unified `Adapter(Batch, State) -> ModelOutput` execution;
- text and structured multimodal input normalization;
- generation and encoder execution through one `Engine`;
- `State` with attribute, mapping, and JAX pytree behavior;
- authoritative paged KV state and lifecycle management;
- scheduler-owned request decisions and Runner-owned JIT executables;
- fixed decode compilation and bounded prefill/encoder buckets;
- budgeted chunked prefill over absolute prompt positions;
- reference-counted LRU prefix caching over authoritative KV pages;
- paged attention for MHA, GQA, and MQA;
- static multimodal tensor collation with per-slot modality masks;
- persistent recurrent/convolution state with inactive-slot protection;
- native Qwix load-time quantization through the TakTiny integration;
- JAX `Mesh`, logical-axis, and `NamedSharding` helpers;
- request-specific sampling parameters.

Model-specific processors remain responsible for turning images, audio, or
video into fixed-shape JAX tensors. Recurrent adapters declare their persistent
per-slot arrays through `StateSpec`. Prefix caching currently applies to pure
paged-KV state; Kirara rejects recurrent/KV hybrid prefix caching rather than
restoring incomplete state. MaxText and optimized Pallas kernels remain future
integrations; unsupported requested features fail explicitly.

## Test

```console
uv run python -m unittest discover -s tests -v
```
