"""Model manager for multi-model support (Phase 3)."""

import torch
from typing import Optional, Dict, Any, Tuple, List
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
    Manages multiple models lifecycle (Phase 3: multi-model support).

    This class handles loading, unloading, and accessing multiple models.
    """

    def __init__(self):
        self._models: Dict[str, ModelHandle] = {}  # {id: handle}
        self._active_id: Optional[str] = "model_a"  # Default primary model ID
        self._loaders: Dict[str, ModelLoader] = {}

    @property
    def active_model(self) -> Optional[ModelHandle]:
        """Get the currently active model handle."""
        return self._models.get(self._active_id)

    @property
    def active_id(self) -> Optional[str]:
        """Get the active model ID."""
        return self._active_id

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._active_id in self._models and self._models[self._active_id] is not None

    @property
    def model_ids(self) -> List[str]:
        """Get list of loaded model IDs."""
        return list(self._models.keys())

    @property
    def num_models(self) -> int:
        """Get number of loaded models."""
        return len(self._models)

    def load(self, config: ModelConfig, model_id: str = None) -> Tuple[bool, str]:
        """
        Load a model according to the given configuration.

        Args:
            config: ModelConfig with model type and parameters
            model_id: Optional ID for the model (default: "model_a")

        Returns:
            Tuple of (success: bool, message: str)
        """
        if model_id is None:
            model_id = self._active_id or "model_a"

        try:
            logger.info(f"Loading model: {config.model_name} (id: {model_id})")

            # Create appropriate loader
            if config.model_type == "huggingface":
                loader = ModelLoader.create(
                    "huggingface",
                    model_name=config.model_name,
                    quantization=config.quantization
                )
            else:
                loader = ModelLoader.create(
                    "pytorch",
                    checkpoint_path=config.checkpoint_path
                )

            # Load model
            model, tokenizer, model_config = loader.load(device=config.device)

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

            device_str = str(loader.current_device) if hasattr(loader, 'current_device') else config.device

            handle = ModelHandle(
                id=model_id,
                name=config.model_name,
                model=model,
                tokenizer=tokenizer,
                config=model_config,
                extractor=extractor,
                simulator=simulator,
                visualizer=visualizer,
                device=device_str
            )

            self._models[model_id] = handle
            self._loaders[model_id] = loader

            # Set as active if first model
            if self._active_id is None:
                self._active_id = model_id

            logger.info(f"Model loaded successfully: {num_layers} layers, {num_heads} heads")
            return True, f"模型加载成功 ({num_layers} 层, {num_heads} 头)"

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False, f"加载失败: {str(e)}"

    def unload(self, model_id: str = None):
        """
        Unload a specific model or the active model.

        Args:
            model_id: Model ID to unload (default: active model)
        """
        if model_id is None:
            model_id = self._active_id

        if model_id in self._models:
            logger.info(f"Unloading model: {self._models[model_id].name}")
            del self._models[model_id]
            if model_id in self._loaders:
                del self._loaders[model_id]

            # If unloading active model, switch to another
            if self._active_id == model_id:
                self._active_id = next(iter(self._models.keys())) if self._models else None

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def set_active(self, model_id: str):
        """Set the active model by ID."""
        if model_id in self._models:
            self._active_id = model_id
            logger.info(f"Switched to model: {model_id}")
        else:
            logger.warning(f"Model not found: {model_id}")

    def get_model(self, model_id: str) -> Optional[ModelHandle]:
        """Get a model handle by ID."""
        return self._models.get(model_id)

    def get_active_handle(self) -> Optional[ModelHandle]:
        """Get the active model handle (alias for active_model property)."""
        return self.active_model

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded models."""
        return {
            "active_id": self._active_id,
            "num_models": len(self._models),
            "models": {
                id: handle.get_summary()
                for id, handle in self._models.items()
            }
        }
