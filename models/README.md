# Place your trained PyTorch model file here
# The model should be named 'best_model.pth' and placed in this directory
# This matches the MODEL_PATH configuration in .env.example

# To use your trained model:
# 1. Copy your best_model.pth file to this folder
# 2. The model will be automatically loaded when the API starts
# 3. Make sure the model was trained with the same architecture (EfficientNet-B3 with GeM pooling)

# If no model file is found, the service will create a dummy model for development/testing