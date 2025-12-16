"""Tests for the model implementations."""

import pytest
import numpy as np
from multimodel_ai.models import TextGenerationModel, ImageAnalysisModel, EmbeddingModel
from multimodel_ai.base import ModelType


class TestTextGenerationModel:
    """Tests for TextGenerationModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = TextGenerationModel(name="test")
        assert model.name == "test"
        assert model.model_type == ModelType.TEXT_GENERATION
        assert not model.is_loaded()
    
    def test_initialization_with_config(self):
        """Test model initialization with config."""
        config = {"parameters": {"max_length": 200, "temperature": 0.9}}
        model = TextGenerationModel(name="test", config=config)
        assert model.max_length == 200
        assert model.temperature == 0.9
    
    def test_load_unload(self):
        """Test loading and unloading."""
        model = TextGenerationModel(name="test")
        assert not model.is_loaded()
        
        model.load()
        assert model.is_loaded()
        
        model.unload()
        assert not model.is_loaded()
    
    def test_predict(self):
        """Test text prediction."""
        model = TextGenerationModel(name="test")
        model.load()
        
        result = model.predict("Hello")
        assert isinstance(result, str)
        assert "test" in result
        assert "Hello" in result
    
    def test_predict_unloaded(self):
        """Test that prediction fails if model not loaded."""
        model = TextGenerationModel(name="test")
        
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict("Hello")


class TestImageAnalysisModel:
    """Tests for ImageAnalysisModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ImageAnalysisModel(name="test")
        assert model.name == "test"
        assert model.model_type == ModelType.IMAGE_ANALYSIS
        assert not model.is_loaded()
    
    def test_load_unload(self):
        """Test loading and unloading."""
        model = ImageAnalysisModel(name="test")
        model.load()
        assert model.is_loaded()
        
        model.unload()
        assert not model.is_loaded()
    
    def test_predict_with_numpy_array(self):
        """Test image prediction with numpy array."""
        model = ImageAnalysisModel(name="test")
        model.load()
        
        image = np.random.rand(224, 224, 3)
        result = model.predict(image)
        
        assert isinstance(result, dict)
        assert "model" in result
        assert "image_shape" in result
        assert result["image_shape"] == (224, 224, 3)
    
    def test_predict_with_path(self):
        """Test image prediction with path."""
        model = ImageAnalysisModel(name="test")
        model.load()
        
        result = model.predict("/path/to/image.jpg")
        
        assert isinstance(result, dict)
        assert "model" in result
        assert "image_path" in result
    
    def test_predict_unloaded(self):
        """Test that prediction fails if model not loaded."""
        model = ImageAnalysisModel(name="test")
        
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(np.random.rand(224, 224, 3))


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = EmbeddingModel(name="test")
        assert model.name == "test"
        assert model.model_type == ModelType.EMBEDDINGS
        assert model.embedding_dim == 768
    
    def test_initialization_with_config(self):
        """Test model initialization with config."""
        config = {"parameters": {"embedding_dim": 512}}
        model = EmbeddingModel(name="test", config=config)
        assert model.embedding_dim == 512
    
    def test_load_unload(self):
        """Test loading and unloading."""
        model = EmbeddingModel(name="test")
        model.load()
        assert model.is_loaded()
        
        model.unload()
        assert not model.is_loaded()
    
    def test_predict_single_text(self):
        """Test embedding prediction for single text."""
        model = EmbeddingModel(name="test")
        model.load()
        
        embedding = model.predict("Hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
    
    def test_predict_batch(self):
        """Test embedding prediction for batch of texts."""
        model = EmbeddingModel(name="test")
        model.load()
        
        texts = ["Hello", "World", "Test"]
        embeddings = model.predict(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)
    
    def test_predict_unloaded(self):
        """Test that prediction fails if model not loaded."""
        model = EmbeddingModel(name="test")
        
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict("Hello")
