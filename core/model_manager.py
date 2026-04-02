"""Basic model manager for single model support (Phase 1)."""

import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from core.model_base import ModelHandle
from model_loader import ModelLoader
from kvcache_extractor import KVCacheExtractor
from kvcache_simulator import KVCacheSimulator
from visualizer import KVCacheVisualizer
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model to be loaded."""
    model_type: str  # "huggingface" or "pytorch"
    model_name: str = "gpt2"
    checkpoint_path: str = ""
    device: str = "auto"
    quantization: str = None
    max_history_length: int = 100  # KV Cache 历史记录最大长度


class ModelManager:
    """
    Manages model lifecycle (Phase 1: single model, Phase 3: multi-model).

    This class handles loading, unloading, and accessing models.
    """

    def __init__(self):
        self._active: Optional[ModelHandle] = None
        self._loader: Optional[ModelLoader] = None

    @property
    def active_model(self) -> Optional[ModelHandle]:
        """Get the currently active model handle."""
        return self._active

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._active is not None

    def load(self, config: ModelConfig) -> Tuple[bool, str]:
        """
        Load a model according to the given configuration.

        Args:
            config: ModelConfig with model type and parameters

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Loading model: {config.model_name}")

            # Create appropriate loader
            if config.model_type == "huggingface":
                self._loader = ModelLoader.create(
                    "huggingface",
                    model_name=config.model_name,
                    quantization=config.quantization
                )
            else:
                self._loader = ModelLoader.create(
                    "pytorch",
                    checkpoint_path=config.checkpoint_path
                )

            # Load model
            model, tokenizer, model_config = self._loader.load(device=config.device)

            # Create model handle
            num_layers = model_config.get('num_layers', 12)
            num_heads = model_config.get('num_heads', 12)
            num_kv_heads = model_config.get('num_kv_heads', num_heads)
            head_dim = model_config.get('head_dim', 64)

            extractor = KVCacheExtractor(
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim
            )

            simulator = KVCacheSimulator(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_history_length=config.max_history_length
            )

            visualizer = KVCacheVisualizer(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim
            )

            device_str = str(self._loader.current_device) if hasattr(self._loader, 'current_device') else config.device

            self._active = ModelHandle(
                id="model_a",
                name=config.model_name,
                model=model,
                tokenizer=tokenizer,
                config=model_config,
                extractor=extractor,
                simulator=simulator,
                visualizer=visualizer,
                device=device_str
            )

            logger.info(f"Model loaded successfully: {num_layers} layers, {num_heads} heads")
            return True, f"模型加载成功 ({num_layers} 层, {num_heads} 头)"

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False, f"加载失败: {str(e)}"

    def unload(self):
        """Unload the current model and free resources."""
        if self._active is not None:
            logger.info(f"Unloading model: {self._active.name}")
            self._active = None
            self._loader = None

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_active_handle(self) -> Optional[ModelHandle]:
        """Get the active model handle (alias for active_model property)."""
        return self._active
