from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kirara.api.schemas import Output, OutputSampling, SamplingParams
from kirara.engine.scheduler import Scheduler
from kirara.models.external import XLLM

type Prompt = str | list[int]
type PromptInput = str | list[str] | list[int] | list[list[int]]
type TokenizeFunction = Callable[[str], Any]


class LLM:
    """Unified generation API for native Kirara and external models.

    A native model runs through Kirara's scheduler, paged cache, and kernels.
    An :class:`XLLM` delegates to the wrapped external generator instead.
    """

    def __init__(
        self,
        model: Any | XLLM,
        tokenizer: Any = None,
        tokenize_fn: TokenizeFunction | None = None,
        max_batch_size: int = 4,
        max_seq_len: int = 2048,
        prefill_token_budget: int | None = None,
        prefill_buckets: list[int] | None = None,
        block_size: int = 16,
        num_kv_blocks: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenize_fn = tokenize_fn
        self.external_model = model if isinstance(model, XLLM) else None
        self.scheduler = None

        if self.external_model is None:
            self.scheduler = Scheduler(
                model=model,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
                prefill_token_budget=prefill_token_budget,
                prefill_buckets=prefill_buckets,
                block_size=block_size,
                num_kv_blocks=num_kv_blocks,
            )

    @property
    def uses_external_model(self) -> bool:
        """Whether generation delegates to an external LLM."""
        return self.external_model is not None

    def __call__(
        self,
        messages: PromptInput,
        sampling: SamplingParams | None = None,
    ) -> list[Output]:
        """Generate one continuation for each prompt."""
        prompts = self._normalize_prompts(messages)
        prompt_ids = [self._tokenize(prompt) for prompt in prompts]
        parameters = sampling or SamplingParams()
        generated = self._generate_token_ids(prompt_ids, parameters)

        return [
            Output(
                prompt=prompt if isinstance(prompt, str) else "",
                prompt_ids=ids,
                output=[
                    OutputSampling(
                        text=self._decode(token_ids),
                        token_ids=token_ids,
                        num_tokens=len(token_ids),
                        stop_reason=self._stop_reason(token_ids, parameters),
                    )
                ],
            )
            for prompt, ids, token_ids in zip(
                prompts,
                prompt_ids,
                generated,
                strict=True,
            )
        ]

    def generate(
        self,
        messages: PromptInput,
        sampling: SamplingParams | None = None,
    ) -> list[Output]:
        """Named equivalent of calling the instance directly."""
        return self(messages, sampling)

    def _generate_token_ids(
        self,
        prompt_ids: list[list[int]],
        sampling: SamplingParams,
    ) -> list[list[int]]:
        if self.external_model is not None:
            return self.external_model.generate(prompt_ids, sampling)
        if self.scheduler is None:
            raise RuntimeError("native scheduler was not initialized")
        return self.scheduler.generate(prompt_ids, sampling)

    @staticmethod
    def _normalize_prompts(messages: PromptInput) -> list[Prompt]:
        if isinstance(messages, str):
            return [messages]
        if not isinstance(messages, list):
            raise TypeError("messages must be text or token IDs")
        if not messages:
            return []
        if all(isinstance(item, int) for item in messages):
            return [list(messages)]
        if all(isinstance(item, str) for item in messages):
            return list(messages)
        if all(
            isinstance(item, list)
            and all(isinstance(token, int) for token in item)
            for item in messages
        ):
            return [list(item) for item in messages]
        raise TypeError(
            "messages must be str, list[str], list[int], or list[list[int]]"
        )

    def _tokenize(self, prompt: Prompt) -> list[int]:
        if isinstance(prompt, list):
            if not prompt:
                raise ValueError("prompts must contain at least one token")
            return list(prompt)

        if self.tokenize_fn is not None:
            result = self.tokenize_fn(prompt)
        elif self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            result = self.tokenizer.encode(prompt)
        elif callable(self.tokenizer):
            result = self.tokenizer(prompt)
        else:
            raise ValueError(
                "text prompts require a tokenizer or tokenize_fn"
            )
        return self._extract_token_ids(result)

    @staticmethod
    def _extract_token_ids(result: Any) -> list[int]:
        input_ids = getattr(result, "input_ids", None)
        if input_ids is None and isinstance(result, dict):
            input_ids = result.get("input_ids")
        if input_ids is None:
            input_ids = result
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if (
            isinstance(input_ids, list)
            and len(input_ids) == 1
            and isinstance(input_ids[0], list)
        ):
            input_ids = input_ids[0]
        if not isinstance(input_ids, list) or not all(
            isinstance(token, int) for token in input_ids
        ):
            raise TypeError("tokenizer must return a one-dimensional token list")
        if not input_ids:
            raise ValueError("prompts must contain at least one token")
        return list(input_ids)

    def _decode(self, token_ids: list[int]) -> str:
        if self.tokenizer is None or not hasattr(self.tokenizer, "decode"):
            return ""
        return str(self.tokenizer.decode(token_ids))

    def _stop_reason(
        self,
        token_ids: list[int],
        sampling: SamplingParams,
    ) -> str:
        eos_ids = sampling.eos_token_ids
        if eos_ids is None and self.tokenizer is not None:
            eos_ids = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if token_ids and eos_ids is not None and token_ids[-1] in eos_ids:
            return "eos"
        return "length"


__all__ = ["LLM"]
