# Kirara

Kirara is a JAX inference runtime with continuous batching and one
authoritative paged KV cache. Model architectures are implemented with
TakTiny, while cache and attention kernels are owned by Kirara.

## Native Kirara model

Pass a TakTiny-backed model directly to `LLM`. Generation uses Kirara's
scheduler, paged cache, and kernels.

```python
from kirara import LLM, SamplingParams

llm = LLM(model, tokenizer=tokenizer)
outputs = llm(
    ["Hello", "Write a tiny Python function"],
    SamplingParams(max_new_tokens=32),
)
```

## External model

Wrap an external implementation in `XLLM`, then pass the adapter to the same
`LLM` interface. This path delegates to the external generator and does not
use Kirara's scheduler or kernels.

```python
from kirara import LLM, SamplingParams, XLLM

llm = LLM(XLLM(external_llm), tokenizer=tokenizer)
outputs = llm("Hello", SamplingParams(max_new_tokens=32))
```

The default external contract is:

```python
external_llm.generate(
    input_ids: list[list[int]],
    sampling_params: SamplingParams,
) -> list[list[int]]
```

Pass `XLLM(external_llm, generate_fn=adapter)` when a library uses another
calling convention. The adapter uses the same input and output contract.

Pretokenized prompts are accepted as `list[int]` or `list[list[int]]`. Text
prompts require a tokenizer or a `tokenize_fn` that accepts one string and
returns one token-ID list.

See [`api-graph/public-api.md`](api-graph/public-api.md) for module ownership
and execution paths.

## Test

```console
uv run python -m unittest discover -s tests -v
```
