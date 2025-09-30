"""Timing utilities"""
import time
from contextlib import contextmanager
from typing import Generator
from rich.console import Console

console = Console()


class Timer:
    """Simple timer for profiling code sections"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start_time
        console.print(f"[cyan]{self.name}[/cyan] completed in [bold]{self.elapsed:.2f}s[/bold]")


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    """Context manager for timing code blocks"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        console.print(f"[cyan]{name}[/cyan] completed in [bold]{elapsed:.2f}s[/bold]")