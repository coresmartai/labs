"""System instructions + prompt utilities."""
from pathlib import Path

_HERE = Path(__file__).parent


def load_system_instruction() -> str:
    """Return the teacher-persona system instruction used by both training
    and inference. Prompts live in .txt files so they can be Git-versioned."""
    return (_HERE / "system_instruction.txt").read_text(encoding="utf-8").strip()
