# Multi-Model AI Framework (멀티모델 인공지능)

A flexible and extensible Python framework for managing and using multiple AI models in a unified system.

## Features

- 🚀 **Multi-Model Support**: Manage multiple AI models of different types simultaneously
- 🔧 **Flexible Architecture**: Easy-to-extend base classes for custom model implementations
- 📦 **Model Registry**: Centralized registry for organizing and accessing models
- 🏭 **Factory Pattern**: Dynamic model creation with configuration support
- 🎯 **Type Safety**: Built-in model type enumeration and type checking
- 📊 **Example Models**: Pre-built implementations for text generation, image analysis, and embeddings

## Installation

### From Source

```bash
git clone https://github.com/HernYung/sub.git
cd sub
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Usage

```python
from multimodel_ai import ModelRegistry
from multimodel_ai.models import TextGenerationModel, ImageAnalysisModel

# Create a model registry
registry = ModelRegistry()

# Create and register models
text_model = TextGenerationModel(name="gpt-demo")
registry.register(text_model)

image_model = ImageAnalysisModel(name="vision-demo")
registry.register(image_model)

# Use the models
text_model.load()
output = text_model.predict("Hello, AI!")
print(output)

# List all models
print(f"Registered models: {registry.list_models()}")
```

### Using the Factory Pattern

```python
from multimodel_ai import ModelFactory
from multimodel_ai.models import TextGenerationModel

# Create a factory
factory = ModelFactory()
factory.register_model_class(TextGenerationModel)

# Create models dynamically
model = factory.create_model(
    class_name="TextGenerationModel",
    name="dynamic-model",
    config={"parameters": {"max_length": 200}}
)

model.load()
result = model.predict("Factory pattern example")
```

## Architecture

### Core Components

1. **BaseModel**: Abstract base class that all models inherit from
   - Defines the interface for model loading, prediction, and unloading
   - Provides common functionality for model management

2. **ModelRegistry**: Central registry for managing multiple models
   - Register and unregister models
   - Retrieve models by name or type
   - Automatic model lifecycle management

3. **ModelFactory**: Factory for creating model instances
   - Register model classes
   - Create models from configuration
   - Dynamic model instantiation

4. **ModelType**: Enumeration of supported model types
   - TEXT_GENERATION
   - IMAGE_ANALYSIS
   - EMBEDDINGS
   - CLASSIFICATION
   - CUSTOM

### Model Types

The framework includes three example model implementations:

- **TextGenerationModel**: For text generation tasks
- **ImageAnalysisModel**: For image analysis and computer vision tasks
- **EmbeddingModel**: For generating text embeddings

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic multi-model usage
- `factory_pattern.py`: Advanced factory pattern usage

Run examples:
```bash
python examples/basic_usage.py
python examples/factory_pattern.py
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest --cov=multimodel_ai tests/
```

## Creating Custom Models

To create your own model, inherit from `BaseModel`:

```python
from multimodel_ai.base import BaseModel, ModelType

class MyCustomModel(BaseModel):
    def __init__(self, name, config=None):
        super().__init__(name=name, model_type=ModelType.CUSTOM, config=config)
    
    def load(self):
        # Load your model
        self._is_loaded = True
    
    def predict(self, input_data, **kwargs):
        # Make predictions
        return your_prediction
    
    def unload(self):
        # Clean up resources
        self._is_loaded = False
```

## Project Structure

```
sub/
├── multimodel_ai/          # Main package
│   ├── __init__.py         # Package initialization
│   ├── base.py             # Base classes and types
│   ├── registry.py         # Model registry
│   ├── factory.py          # Model factory
│   └── models/             # Example model implementations
│       ├── __init__.py
│       ├── text_model.py
│       ├── image_model.py
│       └── embedding_model.py
├── examples/               # Usage examples
│   ├── basic_usage.py
│   └── factory_pattern.py
├── tests/                  # Test suite
│   ├── test_base.py
│   ├── test_registry.py
│   ├── test_factory.py
│   └── test_models.py
├── requirements.txt        # Dependencies
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- typing-extensions >= 4.0.0

## License

This project is open source.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- Integration with popular AI frameworks (TensorFlow, PyTorch, Hugging Face)
- Model versioning support
- Asynchronous model loading and prediction
- Model performance monitoring
- Configuration file support (YAML/JSON)
- CLI interface for model management