"""Tests for the base model classes."""

import pytest
from multimodel_ai.base import BaseModel, ModelType, ModelConfig


class DummyModel(BaseModel):
    """Dummy model for testing."""
    
    def load(self):
        self._is_loaded = True
    
    def predict(self, input_data, **kwargs):
        return f"Prediction: {input_data}"
    
    def unload(self):
        self._is_loaded = False


class TestBaseModel:
    """Tests for BaseModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        assert model.name == "test"
        assert model.model_type == ModelType.TEXT_GENERATION
        assert not model.is_loaded()
    
    def test_initialization_with_config(self):
        """Test model initialization with config."""
        config = {"parameters": {"key": "value"}}
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION, config=config)
        assert model.config == config
    
    def test_load(self):
        """Test model loading."""
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        assert not model.is_loaded()
        model.load()
        assert model.is_loaded()
    
    def test_unload(self):
        """Test model unloading."""
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        model.load()
        assert model.is_loaded()
        model.unload()
        assert not model.is_loaded()
    
    def test_predict(self):
        """Test prediction."""
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        result = model.predict("input")
        assert result == "Prediction: input"
    
    def test_get_info(self):
        """Test getting model info."""
        config = {"parameters": {"key": "value"}}
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION, config=config)
        info = model.get_info()
        
        assert info["name"] == "test"
        assert info["type"] == "text_generation"
        assert info["loaded"] is False
        assert info["config"] == config
    
    def test_repr(self):
        """Test string representation."""
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        repr_str = repr(model)
        assert "DummyModel" in repr_str
        assert "test" in repr_str
        assert "text_generation" in repr_str


class TestModelType:
    """Tests for ModelType enum."""
    
    def test_model_types(self):
        """Test that all expected model types exist."""
        assert ModelType.TEXT_GENERATION.value == "text_generation"
        assert ModelType.IMAGE_ANALYSIS.value == "image_analysis"
        assert ModelType.EMBEDDINGS.value == "embeddings"
        assert ModelType.CLASSIFICATION.value == "classification"
        assert ModelType.CUSTOM.value == "custom"
