"""
Model registry for managing multiple AI models.
"""

from typing import Dict, List, Optional
from multimodel_ai.base import BaseModel, ModelType


class ModelRegistry:
    """
    Registry for managing multiple AI models.
    
    This class provides a centralized way to register, retrieve, and manage
    multiple AI models in the framework.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, BaseModel] = {}
        self._models_by_type: Dict[ModelType, List[str]] = {
            model_type: [] for model_type in ModelType
        }
    
    def register(self, model: BaseModel) -> None:
        """
        Register a model in the registry.
        
        Args:
            model: The model to register
            
        Raises:
            ValueError: If a model with the same name already exists
        """
        if model.name in self._models:
            raise ValueError(f"Model with name '{model.name}' already registered")
        
        self._models[model.name] = model
        self._models_by_type[model.model_type].append(model.name)
    
    def unregister(self, name: str) -> None:
        """
        Unregister a model from the registry.
        
        Args:
            name: Name of the model to unregister
            
        Raises:
            KeyError: If the model doesn't exist
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        
        model = self._models[name]
        if model.is_loaded():
            model.unload()
        
        self._models_by_type[model.model_type].remove(name)
        del self._models[name]
    
    def get(self, name: str) -> Optional[BaseModel]:
        """
        Get a model by name.
        
        Args:
            name: Name of the model
            
        Returns:
            The model if found, None otherwise
        """
        return self._models.get(name)
    
    def get_by_type(self, model_type: ModelType) -> List[BaseModel]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Type of models to retrieve
            
        Returns:
            List of models of the specified type
        """
        model_names = self._models_by_type.get(model_type, [])
        return [self._models[name] for name in model_names]
    
    def list_models(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())
    
    def count(self) -> int:
        """
        Get the number of registered models.
        
        Returns:
            Number of registered models
        """
        return len(self._models)
    
    def clear(self) -> None:
        """Unregister all models."""
        for name in list(self._models.keys()):
            self.unregister(name)
    
    def __contains__(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._models
    
    def __len__(self) -> int:
        """Get the number of registered models."""
        return len(self._models)
