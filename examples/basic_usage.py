"""
Basic usage example of the multi-model AI framework.

This example demonstrates:
1. Creating and registering multiple models
2. Using different model types
3. Managing models through the registry
"""

import numpy as np
from multimodel_ai import ModelRegistry, ModelType
from multimodel_ai.models import TextGenerationModel, ImageAnalysisModel, EmbeddingModel


def main():
    print("=" * 60)
    print("Multi-Model AI Framework - Basic Usage Example")
    print("=" * 60)
    print()
    
    # Create a model registry
    registry = ModelRegistry()
    print("✓ Created model registry")
    print()
    
    # Create and register different models
    print("Registering models...")
    print("-" * 60)
    
    # Text generation model
    text_model = TextGenerationModel(
        name="gpt-demo",
        config={"parameters": {"max_length": 150, "temperature": 0.8}}
    )
    registry.register(text_model)
    print(f"✓ Registered: {text_model}")
    
    # Image analysis model
    image_model = ImageAnalysisModel(
        name="vision-demo",
        config={"parameters": {"classes": ["cat", "dog", "bird"]}}
    )
    registry.register(image_model)
    print(f"✓ Registered: {image_model}")
    
    # Embedding model
    embedding_model = EmbeddingModel(
        name="embedding-demo",
        config={"parameters": {"embedding_dim": 512}}
    )
    registry.register(embedding_model)
    print(f"✓ Registered: {embedding_model}")
    print()
    
    # List all models
    print(f"Total models registered: {registry.count()}")
    print(f"Model names: {registry.list_models()}")
    print()
    
    # Use text generation model
    print("Using Text Generation Model")
    print("-" * 60)
    text_model.load()
    output = text_model.predict("Hello, this is a multi-model AI system")
    print(f"Input: 'Hello, this is a multi-model AI system'")
    print(f"Output: {output}")
    print()
    
    # Use image analysis model
    print("Using Image Analysis Model")
    print("-" * 60)
    image_model.load()
    # Create a dummy image (random numpy array)
    dummy_image = np.random.rand(224, 224, 3)
    result = image_model.predict(dummy_image)
    print(f"Image shape: {dummy_image.shape}")
    print(f"Analysis result: {result}")
    print()
    
    # Use embedding model
    print("Using Embedding Model")
    print("-" * 60)
    embedding_model.load()
    texts = ["Hello world", "Multi-model AI", "Example text"]
    embeddings = embedding_model.predict(texts)
    print(f"Input texts: {texts}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (truncated): {embeddings[0][:5]}...")
    print()
    
    # Get models by type
    print("Models by Type")
    print("-" * 60)
    for model_type in ModelType:
        models = registry.get_by_type(model_type)
        if models:
            print(f"{model_type.value}: {[m.name for m in models]}")
    print()
    
    # Cleanup
    print("Cleaning up...")
    print("-" * 60)
    text_model.unload()
    image_model.unload()
    embedding_model.unload()
    print("✓ All models unloaded")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
