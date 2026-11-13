"""Minimal chat-completion client over any OpenAI-compatible endpoint
(vLLM, OpenAI, Gemini/Anthropic compatibility endpoints).

Every call returns the full request that was sent so it can be stored verbatim.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import openai


@dataclass
class ModelSpec:
    name: str                       # display / file-safe name
    model_id: str                   # id sent to the API
    provider: str = "openai"        # any OpenAI-compatible chat endpoint (vLLM)
    base_url: str | None = None     # None -> OpenAI default
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 256
    seed: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout: float = 120.0
    max_retries: int = 5
    concurrency: int = 8
    system_role: str = "system"     # some models want "developer" or no system role
    max_tokens_extra: int = 0       # added to each trial's max_tokens (thinking models: reasoning
                                    # tokens share the budget with the visible reply)
    judge_only: bool = False
    serve: str = ""                 # extra vllm flags (used by scripts/run_models_thalys.sh)
    gpu_mem: float = 0.92           # vllm --gpu-memory-utilization


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    request: dict[str, Any]
    error: str | None = None
    attempts: int = 1
    logprobs: list | None = None      # per content token: {"token": str, "top": {tok: logprob}}


class ChatClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        key = os.environ.get(spec.api_key_env) or ("EMPTY" if spec.base_url else None)
        if key is None:
            raise RuntimeError(f"{spec.api_key_env} is not set for model {spec.name}")
        self._client = openai.OpenAI(api_key=key, base_url=spec.base_url, timeout=spec.timeout,
                                     max_retries=0)

    def build_request(self, system: str | None, user: str, **overrides) -> dict[str, Any]:
        msgs = []
        if system:
            msgs.append({"role": self.spec.system_role, "content": system})
        msgs.append({"role": "user", "content": user})
        req: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": msgs,
            "temperature": overrides.get("temperature", self.spec.temperature),
            "max_tokens": overrides.get("max_tokens", self.spec.max_tokens) + self.spec.max_tokens_extra,
        }
        if overrides.get("logprobs"):
            req["logprobs"] = True
            req["top_logprobs"] = int(overrides.get("top_logprobs", 10))
        seed = overrides.get("seed", self.spec.seed)
        if seed is not None:
            req["seed"] = seed
        if self.spec.extra_body:
            req["extra_body"] = dict(self.spec.extra_body)
        return req

    def complete(self, system: str | None, user: str, **overrides) -> Completion:
        req = self.build_request(system, user, **overrides)
        last_err = None
        for attempt in range(1, self.spec.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**req)
                ch = resp.choices[0]
                usage = resp.usage
                lps = None
                if getattr(ch, "logprobs", None) and ch.logprobs and ch.logprobs.content:
                    lps = [{"token": t.token, "top": {x.token: x.logprob for x in (t.top_logprobs or [])}}
                           for t in ch.logprobs.content]
                return Completion(
                    text=(ch.message.content or ""),
                    finish_reason=ch.finish_reason,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    request=req,
                    attempts=attempt,
                    logprobs=lps,
                )
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(60, 2 ** attempt))
            except openai.APIStatusError as e:  # 4xx: do not retry
                return Completion("", None, None, None, (time.perf_counter() - t0) * 1000, req,
                                  error=f"{type(e).__name__}: {e}", attempts=attempt)
        return Completion("", None, None, None, 0.0, req, error=last_err or "unknown",
                          attempts=self.spec.max_retries)

