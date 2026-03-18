"""Dataset adapters that map raw sources into the shared Event schema."""

from .infopath import InfoPathConfig, prepare_dataset as prepare_infopath_dataset
from .synthetic_sse import SyntheticSseConfig, prepare_dataset as prepare_synthetic_sse_dataset
from .uci_news import UciNewsConfig, prepare_dataset

__all__ = [
    "InfoPathConfig",
    "SyntheticSseConfig",
    "UciNewsConfig",
    "prepare_dataset",
    "prepare_infopath_dataset",
    "prepare_synthetic_sse_dataset",
]
