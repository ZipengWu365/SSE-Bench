"""Dataset adapters that map raw sources into the shared Event schema."""

from .uci_news import UciNewsConfig, prepare_dataset

__all__ = ["UciNewsConfig", "prepare_dataset"]
