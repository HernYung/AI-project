"""
Model factory for creating model instances.
"""

from typing import Type, Dict, Any, Optional
from multimodel_ai.base import BaseModel, ModelType, ModelConfig


class ModelFactory:
    """
    Factory for creating model instances.
    
    This class provides a centralized way to create different types of models
    based on configuration.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self._model_classes: Dict[str, Type[BaseModel]] = {}
    
    def register_model_class(self, model_class: Type[BaseModel], class_name: Optional[str] = None) -> None:
        """
        Register a model class with the factory.
        
        Args:
            model_class: The model class to register
            class_name: Optional name for the class (defaults to class.__name__)
        """
        name = class_name or model_class.__name__
        self._model_classes[name] = model_class
    
    def create_model(
        self,
        class_name: str,
        name: str,
        config: Optional[ModelConfig] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            class_name: Name of the registered model class
            name: Name for the model instance
            config: Optional configuration
            **kwargs: Additional keyword arguments to pass to the model constructor
            
        Returns:
            Created model instance
            
        Raises:
            ValueError: If the model class is not registered
        """
        if class_name not in self._model_classes:
            raise ValueError(f"Model class '{class_name}' not registered")
        
        model_class = self._model_classes[class_name]
        return model_class(name=name, config=config, **kwargs)
    
    def list_model_classes(self) -> list:
        """
        List all registered model class names.
        
        Returns:
            List of model class names
        """
        return list(self._model_classes.keys())
