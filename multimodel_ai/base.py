"""
Base classes for multi-model AI framework.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, List
from typing_extensions import TypedDict


class ModelType(Enum):
    """Enumeration of supported model types."""
    TEXT_GENERATION = "text_generation"
    IMAGE_ANALYSIS = "image_analysis"
    EMBEDDINGS = "embeddings"
    CLASSIFICATION = "classification"
    CUSTOM = "custom"


class ModelConfig(TypedDict, total=False):
    """Configuration for a model."""
    name: str
    model_type: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseModel(ABC):
    """
    Abstract base class for all AI models.
    
    All models in the multi-model framework should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, name: str, model_type: ModelType, config: Optional[ModelConfig] = None):
        """
        Initialize the base model.
        
        Args:
            name: Name of the model
            model_type: Type of the model
            config: Optional configuration dictionary
        """
        self.name = name
        self.model_type = model_type
        self.config = config or {}
        self._is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """
        Make a prediction using the model.
        
        Args:
            input_data: Input data for prediction
            **kwargs: Additional parameters
            
        Returns:
            Model prediction
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.name,
            "type": self.model_type.value,
            "loaded": self._is_loaded,
            "config": self.config,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.model_type.value})"
