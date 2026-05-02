"""Small text helpers for verification prompts (avoid importing main)."""


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n... [truncated]"
