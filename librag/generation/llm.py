"""Minimal Ollama client with deterministic decoding and precise timing.

Ollama reports durations in nanoseconds; model loading time is returned
separately, so cold-start cost is excluded from per-query latency and
reported on its own.
"""
from __future__ import annotations

import time


class OllamaClient:
    def __init__(self, url: str, model: str, options: dict | None = None, keep_alive: str = "10m",
                 timeout: int = 300):
        self.url = url.rstrip("/")
        self.model = model
        self.options = dict(options or {})
        self.keep_alive = keep_alive
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> dict:
        import requests

        for attempt in range(3):
            try:
                r = requests.post(f"{self.url}{endpoint}", json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("unreachable")

    def chat(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        t0 = time.perf_counter()
        data = self._post("/api/chat", payload)
        wall = time.perf_counter() - t0
        ns = 1e9
        eval_s = data.get("eval_duration", 0) / ns
        return {
            "text": data.get("message", {}).get("content", ""),
            "wall_s": round(wall, 4),
            "total_s": round(data.get("total_duration", 0) / ns, 4),
            "load_s": round(data.get("load_duration", 0) / ns, 4),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "prompt_eval_s": round(data.get("prompt_eval_duration", 0) / ns, 4),
            "output_tokens": data.get("eval_count", 0),
            "eval_s": round(eval_s, 4),
            "tokens_per_s": round(data.get("eval_count", 0) / eval_s, 2) if eval_s else None,
            "done_reason": data.get("done_reason"),
        }

    def available_models(self) -> list[str]:
        import requests

        r = requests.get(f"{self.url}/api/tags", timeout=30)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def model_info(self) -> dict:
        info = self._post("/api/show", {"model": self.model})
        details = info.get("details", {})
        return {"model": self.model, "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
                "digest": info.get("digest") or info.get("modified_at")}

    def warm_up(self) -> float:
        """Load the model once and return the load time (reported as cold-start cost)."""
        return self.chat("Reply with OK.", "OK")["load_s"]

    def unload(self) -> None:
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})
