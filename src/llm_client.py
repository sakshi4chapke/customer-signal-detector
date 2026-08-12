"""
Thin wrapper around the Gemini API.

Everything that makes LLM calls survivable lives here so the agents can stay
focused on prompts and signals:

  * caching     - identical prompts never hit the API twice
  * retry       - transient failures and malformed JSON get a second chance
  * backoff     - exponential wait on rate limits (the free tier is strict)
  * kill switch - USE_LLM=false runs the whole pipeline with zero API calls

The cache is not a nicety. During development you will run the pipeline
dozens of times over the same 169 customers; without it you would exhaust
the free-tier quota before lunch and each run would take five minutes.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(".cache/llm")
MAX_RETRIES = 2
BASE_BACKOFF = 4.0          # seconds; doubles each attempt

# Free-tier Gemini allows roughly 15 requests/minute. Firing as fast as the
# loop can go guarantees rate-limiting, so we space calls out deliberately.
MIN_CALL_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "4.5"))

# Errors that will NEVER succeed on retry. Retrying these just burns quota
# and hides the real problem - a retired model name, a bad key, a typo.
FATAL_MARKERS = ("404", "NOT_FOUND", "no longer available",
                 "403", "PERMISSION_DENIED", "API key not valid",
                 "API_KEY_INVALID", "400", "INVALID_ARGUMENT")

# Errors worth waiting out.
TRANSIENT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate",
                     "500", "503", "UNAVAILABLE", "DEADLINE_EXCEEDED")

# Gemini's free tier caps requests PER DAY PER MODEL (20/day for some models).
# Waiting cannot help - but another model has its own separate bucket, so we
# fail over rather than stop. Order matters: best quality first.
MODEL_CHAIN = [m.strip() for m in os.getenv(
    "GEMINI_MODEL_CHAIN",
    "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.5-flash-lite"
).split(",") if m.strip()]


def _is_daily_quota(exc):
    """Distinguish 'wait a minute' from 'this model is done for today'."""
    text = str(exc)
    return "429" in text and ("PerDay" in text or "RequestsPerDay" in text)


class LLMUnavailable(Exception):
    """Raised when the model cannot be reached after all retries."""


class LLMFatalError(LLMUnavailable):
    """Configuration is wrong - retrying cannot help. Abort the run."""


def _classify(exc):
    text = str(exc)
    if any(m in text for m in FATAL_MARKERS):
        return "fatal"
    if any(m in text for m in TRANSIENT_MARKERS):
        return "transient"
    return "unknown"


def _strip_fences(text):
    """LLMs wrap JSON in markdown fences despite being told not to.

    Handles ```json ... ``` and bare ``` ... ``` and leading prose.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Fall back to the outermost JSON object/array if prose surrounds it.
    if not text.startswith(("{", "[")):
        match = re.search(r"([\[{].*[\]}])", text, re.DOTALL)
        if match:
            text = match.group(1)
    return text.strip()


class LLMClient:
    def __init__(self, model=None, use_llm=None, cache=True):
        preferred = model or os.getenv("GEMINI_MODEL")
        # Build the fallback chain, preferred model first, no duplicates.
        chain = ([preferred] if preferred else []) + MODEL_CHAIN
        self.model_chain = list(dict.fromkeys(chain))
        self.model = self.model_chain[0]
        self.exhausted = set()
        env_flag = os.getenv("USE_LLM", "true").lower() == "true"
        self.use_llm = env_flag if use_llm is None else use_llm
        self.cache = cache
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "failures": 0}
        self.last_error = None
        self._last_call_at = 0.0
        self.stats["model_switches"] = 0

        self._client = None
        if self.use_llm:
            key = os.getenv("GEMINI_API_KEY")
            if not key or key.startswith("your_"):
                raise LLMUnavailable(
                    "GEMINI_API_KEY missing from .env. Set it, or run with "
                    "USE_LLM=false for a rules-only pipeline."
                )
            from google import genai
            self._client = genai.Client(api_key=key)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- caching

    def _next_model(self):
        """Move to the next model whose daily quota is still intact."""
        for candidate in self.model_chain:
            if candidate not in self.exhausted:
                if candidate != self.model:
                    print(f"    [quota] {self.model} exhausted for today -> "
                          f"switching to {candidate}")
                    self.stats["model_switches"] += 1
                self.model = candidate
                return True
        return False

    def _cache_key(self, prompt):
        # Deliberately NOT keyed on model. A cached analysis of the same
        # transcript is equally valid whichever model produced it, and
        # re-keying on model would discard the cache on every failover.
        raw = f"signal-extraction-v1::{prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def _cache_read(self, key):
        path = CACHE_DIR / f"{key}.json"
        if self.cache and path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
        return None

    def _cache_write(self, key, value):
        if self.cache:
            (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))

    # ----------------------------------------------------------------- call

    def _throttle(self):
        """Space calls out to stay under the per-minute limit."""
        wait = MIN_CALL_INTERVAL - (time.time() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.time()

    def healthcheck(self):
        """Prove at least one model in the chain is usable BEFORE starting a
        long run. Fails fast instead of failing once per customer."""
        if not self.use_llm:
            return False, "USE_LLM is false"
        errors = []
        for candidate in self.model_chain:
            try:
                self._client.models.generate_content(
                    model=candidate,
                    contents='Return only: {"ok":true}',
                    config={"temperature": 0,
                            "response_mime_type": "application/json"},
                )
                self.model = candidate
                usable = [m for m in self.model_chain
                          if m not in self.exhausted]
                return True, (f"{candidate} responding "
                              f"(fallback chain: {' -> '.join(usable)})")
            except Exception as exc:                       # noqa: BLE001
                if _is_daily_quota(exc):
                    self.exhausted.add(candidate)
                errors.append(f"{candidate}: {str(exc)[:80]}")
        return False, "no usable model:\n    " + "\n    ".join(errors)

    def generate_json(self, prompt):
        """Send a prompt, return parsed JSON. Raises LLMUnavailable on
        exhaustion so the calling agent can degrade rather than crash."""
        if not self.use_llm:
            raise LLMUnavailable("USE_LLM is false - running rules-only")

        key = self._cache_key(prompt)
        cached = self._cache_read(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        last_error = None
        for attempt in range(MAX_RETRIES + len(self.model_chain)):
            try:
                self._throttle()
                self.stats["calls"] += 1
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        # Low temperature: we want judgement, not creativity.
                        # The same transcript should classify the same way.
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    },
                )
                parsed = json.loads(_strip_fences(response.text))
                self._cache_write(key, parsed)
                return parsed

            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON: {exc}"
                self.last_error = last_error
                self.stats["retries"] += 1

            except Exception as exc:                       # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                self.last_error = last_error
                kind = _classify(exc)

                # A retired model or bad key will never work. Stop the whole
                # run immediately rather than repeating the mistake 169 times.
                if kind == "fatal":
                    self.stats["failures"] += 1
                    raise LLMFatalError(last_error) from exc

                # Daily quota gone: waiting is pointless, switch models.
                if _is_daily_quota(exc):
                    self.exhausted.add(self.model)
                    if self._next_model():
                        continue                  # retry immediately, new model
                    self.stats["failures"] += 1
                    raise LLMFatalError(
                        "All models exhausted their daily free-tier quota. "
                        "Run with USE_LLM=false, reduce LLM_BUDGET, or wait "
                        "for the quota to reset (midnight Pacific)."
                    ) from exc

                self.stats["retries"] += 1
                if kind == "transient":
                    time.sleep(BASE_BACKOFF * (2 ** attempt))
                else:
                    time.sleep(1.0)

        self.stats["failures"] += 1
        raise LLMUnavailable(f"failed after {MAX_RETRIES} attempts: {last_error}")

    def report(self):
        s = self.stats
        return (f"model: {self.model} · calls: {s['calls']} · "
                f"cache hits: {s['cache_hits']} · retries: {s['retries']} · "
                f"failures: {s['failures']}")