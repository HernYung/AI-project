"""Tests for the model factory."""

import pytest
from multimodel_ai.base import BaseModel, ModelType
from multimodel_ai.factory import ModelFactory


class DummyModel(BaseModel):
    """Dummy model for testing."""
    
    def load(self):
        self._is_loaded = True
    
    def predict(self, input_data, **kwargs):
        return f"Prediction: {input_data}"
    
    def unload(self):
        self._is_loaded = False


class TestModelFactory:
    """Tests for ModelFactory class."""
    
    def test_initialization(self):
        """Test factory initialization."""
        factory = ModelFactory()
        assert factory.list_model_classes() == []
    
    def test_register_model_class(self):
        """Test registering a model class."""
        factory = ModelFactory()
        factory.register_model_class(DummyModel)
        
        classes = factory.list_model_classes()
        assert len(classes) == 1
        assert "DummyModel" in classes
    
    def test_register_model_class_with_custom_name(self):
        """Test registering a model class with custom name."""
        factory = ModelFactory()
        factory.register_model_class(DummyModel, class_name="CustomName")
        
        classes = factory.list_model_classes()
        assert "CustomName" in classes
    
    def test_create_model(self):
        """Test creating a model."""
        factory = ModelFactory()
        factory.register_model_class(DummyModel)
        
        model = factory.create_model(
            class_name="DummyModel",
            name="test",
            model_type=ModelType.TEXT_GENERATION
        )
        
        assert isinstance(model, DummyModel)
        assert model.name == "test"
        assert model.model_type == ModelType.TEXT_GENERATION
    
    def test_create_model_with_config(self):
        """Test creating a model with config."""
        factory = ModelFactory()
        factory.register_model_class(DummyModel)
        
        config = {"parameters": {"key": "value"}}
        model = factory.create_model(
            class_name="DummyModel",
            name="test",
            model_type=ModelType.TEXT_GENERATION,
            config=config
        )
        
        assert model.config == config
    
    def test_create_model_unregistered_class(self):
        """Test creating a model with unregistered class raises error."""
        factory = ModelFactory()
        
        with pytest.raises(ValueError, match="not registered"):
            factory.create_model(
                class_name="NonexistentModel",
                name="test"
            )
    
    def test_list_model_classes(self):
        """Test listing model classes."""
        factory = ModelFactory()
        
        assert factory.list_model_classes() == []
        
        factory.register_model_class(DummyModel, class_name="Model1")
        factory.register_model_class(DummyModel, class_name="Model2")
        
        classes = factory.list_model_classes()
        assert len(classes) == 2
        assert "Model1" in classes
        assert "Model2" in classes
