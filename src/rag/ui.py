# src/rag/ui.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

@dataclass
class Turn:
    user: str
    assistant: str = ""


def render_chat(turns: List[Turn], max_turns: int = 8) -> Panel:
    body = Text()
    recent = turns[-max_turns:]

    if not recent:
        body.append("Welcome! Ask a question.\n", style="bold")
        body.append("Commands: /exit, /clear\n")

    for t in recent:
        body.append("You\n", style="bold cyan")
        body.append(t.user.strip() + "\n\n")
        body.append("Assistant\n", style="bold green")
        body.append((t.assistant or "").rstrip() + "\n")
        body.append("─" * 70 + "\n")

    return Panel(body, title="PDF Q&A", subtitle="Type below • /exit • /clear")