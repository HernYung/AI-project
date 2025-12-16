"""Tests for the model registry."""

import pytest
from multimodel_ai.base import BaseModel, ModelType
from multimodel_ai.registry import ModelRegistry


class DummyModel(BaseModel):
    """Dummy model for testing."""
    
    def load(self):
        self._is_loaded = True
    
    def predict(self, input_data, **kwargs):
        return f"Prediction: {input_data}"
    
    def unload(self):
        self._is_loaded = False


class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    def test_initialization(self):
        """Test registry initialization."""
        registry = ModelRegistry()
        assert registry.count() == 0
        assert registry.list_models() == []
    
    def test_register(self):
        """Test registering a model."""
        registry = ModelRegistry()
        model = DummyModel(name="test1", model_type=ModelType.TEXT_GENERATION)
        
        registry.register(model)
        assert registry.count() == 1
        assert "test1" in registry
    
    def test_register_duplicate(self):
        """Test that registering duplicate model raises error."""
        registry = ModelRegistry()
        model1 = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        model2 = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        
        registry.register(model1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(model2)
    
    def test_unregister(self):
        """Test unregistering a model."""
        registry = ModelRegistry()
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        
        registry.register(model)
        assert registry.count() == 1
        
        registry.unregister("test")
        assert registry.count() == 0
        assert "test" not in registry
    
    def test_unregister_loaded_model(self):
        """Test that unregistering a loaded model unloads it."""
        registry = ModelRegistry()
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        model.load()
        
        registry.register(model)
        assert model.is_loaded()
        
        registry.unregister("test")
        assert not model.is_loaded()
    
    def test_unregister_nonexistent(self):
        """Test that unregistering nonexistent model raises error."""
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.unregister("nonexistent")
    
    def test_get(self):
        """Test getting a model by name."""
        registry = ModelRegistry()
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        
        registry.register(model)
        retrieved = registry.get("test")
        assert retrieved is model
    
    def test_get_nonexistent(self):
        """Test getting a nonexistent model returns None."""
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None
    
    def test_get_by_type(self):
        """Test getting models by type."""
        registry = ModelRegistry()
        model1 = DummyModel(name="text1", model_type=ModelType.TEXT_GENERATION)
        model2 = DummyModel(name="text2", model_type=ModelType.TEXT_GENERATION)
        model3 = DummyModel(name="image1", model_type=ModelType.IMAGE_ANALYSIS)
        
        registry.register(model1)
        registry.register(model2)
        registry.register(model3)
        
        text_models = registry.get_by_type(ModelType.TEXT_GENERATION)
        assert len(text_models) == 2
        assert model1 in text_models
        assert model2 in text_models
        
        image_models = registry.get_by_type(ModelType.IMAGE_ANALYSIS)
        assert len(image_models) == 1
        assert model3 in image_models
    
    def test_list_models(self):
        """Test listing all model names."""
        registry = ModelRegistry()
        model1 = DummyModel(name="test1", model_type=ModelType.TEXT_GENERATION)
        model2 = DummyModel(name="test2", model_type=ModelType.IMAGE_ANALYSIS)
        
        registry.register(model1)
        registry.register(model2)
        
        names = registry.list_models()
        assert len(names) == 2
        assert "test1" in names
        assert "test2" in names
    
    def test_clear(self):
        """Test clearing all models."""
        registry = ModelRegistry()
        model1 = DummyModel(name="test1", model_type=ModelType.TEXT_GENERATION)
        model2 = DummyModel(name="test2", model_type=ModelType.IMAGE_ANALYSIS)
        
        registry.register(model1)
        registry.register(model2)
        assert registry.count() == 2
        
        registry.clear()
        assert registry.count() == 0
    
    def test_contains(self):
        """Test checking if model exists in registry."""
        registry = ModelRegistry()
        model = DummyModel(name="test", model_type=ModelType.TEXT_GENERATION)
        
        assert "test" not in registry
        registry.register(model)
        assert "test" in registry
    
    def test_len(self):
        """Test length of registry."""
        registry = ModelRegistry()
        assert len(registry) == 0
        
        model1 = DummyModel(name="test1", model_type=ModelType.TEXT_GENERATION)
        registry.register(model1)
        assert len(registry) == 1
        
        model2 = DummyModel(name="test2", model_type=ModelType.IMAGE_ANALYSIS)
        registry.register(model2)
        assert len(registry) == 2
