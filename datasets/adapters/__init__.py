"""Dataset adapters that map raw sources into the shared Event schema."""

from .infopath import InfoPathConfig, prepare_dataset as prepare_infopath_dataset
from .uci_news import UciNewsConfig, prepare_dataset

__all__ = ["InfoPathConfig", "UciNewsConfig", "prepare_dataset", "prepare_infopath_dataset"]
