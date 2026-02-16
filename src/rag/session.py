# src/rag/session.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _pick_int(d: Dict[str, Any], keys: list[str]) -> Optional[int]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, int):
            return v
    return None


@dataclass
class TurnUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class SessionStats:
    turns: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    total_tokens_total: int = 0

    # Optional (set these if you want cost estimates)
    input_price_per_1k: Optional[float] = None
    output_price_per_1k: Optional[float] = None

    def update(self, usage: TurnUsage) -> None:
        self.turns += 1
        self.input_tokens_total += usage.input_tokens
        self.output_tokens_total += usage.output_tokens
        self.total_tokens_total += usage.total_tokens

    def estimate_cost(self) -> Optional[float]:
        if self.input_price_per_1k is None or self.output_price_per_1k is None:
            return None
        return (
            (self.input_tokens_total / 1000.0) * self.input_price_per_1k
            + (self.output_tokens_total / 1000.0) * self.output_price_per_1k
        )

    def format_summary(self, turn_usage: TurnUsage) -> str:
        base = (
            f"📌 This turn tokens: input={turn_usage.input_tokens}, output={turn_usage.output_tokens}, "
            f"total={turn_usage.total_tokens}\n"
            f"📊 Session so far ({self.turns} turns): input={self.input_tokens_total}, output={self.output_tokens_total}, "
            f"total={self.total_tokens_total}"
        )
        cost = self.estimate_cost()
        if cost is not None:
            base += f"\n💰 Estimated session cost: ${cost:.6f}"
        return base


def usage_from_metadata(meta: Dict[str, Any]) -> TurnUsage:
    """
    Normalize usage metadata into TurnUsage.

    Supports common shapes:
    - OpenAI: {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    - Some LangChain variants: {"input_tokens": X, "output_tokens": Y, "total_tokens": Z}
    - Sometimes nested under "usage" or "token_usage"
    """
    if not meta:
        return TurnUsage()

    usage = meta
    if isinstance(meta.get("usage"), dict):
        usage = meta["usage"]
    if isinstance(meta.get("token_usage"), dict):
        usage = meta["token_usage"]

    input_tokens = _pick_int(usage, ["prompt_tokens", "input_tokens"]) or 0
    output_tokens = _pick_int(usage, ["completion_tokens", "output_tokens"]) or 0
    total_tokens = _pick_int(usage, ["total_tokens"]) or (input_tokens + output_tokens)

    return TurnUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)