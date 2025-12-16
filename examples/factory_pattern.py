"""
Advanced usage example showing the model factory pattern.

This example demonstrates:
1. Using ModelFactory to create models dynamically
2. Registering custom model classes
3. Creating models from configuration
"""

from multimodel_ai import ModelRegistry, ModelFactory, ModelType
from multimodel_ai.models import TextGenerationModel, ImageAnalysisModel, EmbeddingModel


def main():
    print("=" * 60)
    print("Multi-Model AI Framework - Factory Pattern Example")
    print("=" * 60)
    print()
    
    # Create factory and registry
    factory = ModelFactory()
    registry = ModelRegistry()
    print("✓ Created factory and registry")
    print()
    
    # Register model classes with factory
    print("Registering model classes with factory...")
    print("-" * 60)
    factory.register_model_class(TextGenerationModel)
    factory.register_model_class(ImageAnalysisModel)
    factory.register_model_class(EmbeddingModel)
    print(f"✓ Registered classes: {factory.list_model_classes()}")
    print()
    
    # Create models using factory
    print("Creating models using factory...")
    print("-" * 60)
    
    # Create text model
    text_model = factory.create_model(
        class_name="TextGenerationModel",
        name="dynamic-text-model",
        config={"parameters": {"max_length": 200, "temperature": 0.9}}
    )
    registry.register(text_model)
    print(f"✓ Created and registered: {text_model}")
    
    # Create image model
    image_model = factory.create_model(
        class_name="ImageAnalysisModel",
        name="dynamic-image-model",
        config={"parameters": {"classes": ["person", "vehicle", "animal"]}}
    )
    registry.register(image_model)
    print(f"✓ Created and registered: {image_model}")
    
    # Create embedding model
    embedding_model = factory.create_model(
        class_name="EmbeddingModel",
        name="dynamic-embedding-model",
        config={"parameters": {"embedding_dim": 1024}}
    )
    registry.register(embedding_model)
    print(f"✓ Created and registered: {embedding_model}")
    print()
    
    # Use the dynamically created models
    print("Using dynamically created models...")
    print("-" * 60)
    
    # Load and use text model
    text_model.load()
    result = text_model.predict("Testing factory pattern")
    print(f"Text model output: {result}")
    
    # Load and use embedding model
    embedding_model.load()
    embeddings = embedding_model.predict("Factory pattern example")
    print(f"Embedding shape: {embeddings.shape}")
    print()
    
    # Show registry info
    print("Registry Information")
    print("-" * 60)
    print(f"Total models: {len(registry)}")
    for name in registry.list_models():
        model = registry.get(name)
        info = model.get_info()
        print(f"  - {info['name']}: type={info['type']}, loaded={info['loaded']}")
    print()
    
    # Cleanup
    print("Cleaning up...")
    registry.clear()
    print("✓ All models unregistered")
    print()
    
    print("=" * 60)
    print("Factory pattern example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
