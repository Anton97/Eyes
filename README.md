# ResNet Model Library

This repository contains a simple library for loading a trained ResNet34 model and making predictions from image files.

## Installation

Clone the repository and make sure you have the following Python packages:

```bash
pip install torch torchvision pillow
```

## Usage

```python
from resnet_model.model import load_model
from resnet_model.utils import preprocess_image, predict

model = load_model("ResNet_best_model.pth")
image_tensor = preprocess_image("your_image.jpg")
class_id, probabilities = predict(model, image_tensor)

print(f"Predicted class: {class_id}")
print(f"Probabilities: {probabilities}")
```
