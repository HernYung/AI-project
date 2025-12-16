"""
Multi-model AI Framework
========================

A flexible framework for managing and using multiple AI models.
"""

__version__ = "0.1.0"

from multimodel_ai.base import BaseModel, ModelType
from multimodel_ai.registry import ModelRegistry
from multimodel_ai.factory import ModelFactory

__all__ = [
    "BaseModel",
    "ModelType",
    "ModelRegistry",
    "ModelFactory",
]
