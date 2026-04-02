"""Model handle data class for multi-model management."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer


@dataclass
class ModelHandle:
    """
    A handle containing all objects associated with a single loaded model.

    This dataclass groups together the model, tokenizer, extractor,
    simulator, and visualizer for easy management and access.
    """

    id: str
    name: str
    model: Any
    tokenizer: Any
    config: Dict[str, Any]
    extractor: KVCacheExtractor
    simulator: KVCacheSimulator
    visualizer: KVCacheVisualizer
    device: str = "cpu"

    @property
    def num_layers(self) -> int:
        """Get number of layers from config."""
        return self.config.get("num_layers", 12)

    @property
    def num_heads(self) -> int:
        """Get number of attention heads from config."""
        return self.config.get("num_heads", 12)

    @property
    def head_dim(self) -> int:
        """Get head dimension from config."""
        return self.config.get("head_dim", 64)

    @property
    def num_kv_heads(self) -> int:
        """Get number of KV heads (for GQA) from config."""
        return self.config.get("num_kv_heads", self.num_heads)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary dict of the model configuration."""
        return {
            "id": self.id,
            "name": self.name,
            "device": self.device,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.config.get("hidden_size", "N/A"),
        }