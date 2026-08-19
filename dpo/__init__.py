"""DPO dataset builder — converts completed RunState objects into preference pairs."""

from .builder import build_pair
from .storage import append_pair

__all__ = ["build_pair", "append_pair"]
