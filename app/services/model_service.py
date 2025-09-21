import torch
import numpy as np
import os
from typing import List, Optional
from app.models.schemas import PredictionResult, RetinalCondition
from app.models.pytorch_model import RetinalModel
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ModelService:
    """
    Singleton service for loading and using the PyTorch retinal analysis model
    """
    _instance: Optional['ModelService'] = None
    _model = None
    _model_loaded = False
    
    def __init__(self):
        if ModelService._instance is not None:
            raise RuntimeError("ModelService is a singleton. Use get_instance() method.")
        
        self.model_path = os.getenv("MODEL_PATH", "./models/best_model.pth")
        self.input_size = int(os.getenv("MODEL_INPUT_SIZE", "300"))
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        
        # Set device (CPU/CUDA)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)
        
        # Class labels for diabetic retinopathy (0-4 scale)
        self.class_labels = [
            RetinalCondition.NORMAL,                    # 0 - No DR
            RetinalCondition.DIABETIC_RETINOPATHY,      # 1 - Mild
            RetinalCondition.DIABETIC_RETINOPATHY,      # 2 - Moderate  
            RetinalCondition.DIABETIC_RETINOPATHY,      # 3 - Severe
            RetinalCondition.DIABETIC_RETINOPATHY       # 4 - Proliferative
        ]
        
        # Detailed class names for better reporting
        self.detailed_class_names = [
            "No Diabetic Retinopathy",
            "Mild Diabetic Retinopathy", 
            "Moderate Diabetic Retinopathy",
            "Severe Diabetic Retinopathy",
            "Proliferative Diabetic Retinopathy"
        ]
        
        self._load_model()
    
    @classmethod
    def get_instance(cls) -> 'ModelService':
        """Get singleton instance of ModelService"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_model(self):
        """Load the trained PyTorch model"""
        try:
            if os.path.exists(self.model_path):
                logger.info("Loading PyTorch model from %s", self.model_path)
                
                # Initialize model architecture (must match training)
                self._model = RetinalModel(
                    model_name='efficientnet_b3',
                    num_classes=1,  # Regression output
                    pretrained=False  # We're loading trained weights
                )
                
                # Load trained weights
                state_dict = torch.load(self.model_path, map_location=self.device)
                self._model.load_state_dict(state_dict)
                
                # Move model to device and set to evaluation mode
                self._model.to(self.device)
                self._model.eval()
                
                self._model_loaded = True
                logger.info("PyTorch model loaded successfully")
                
                # Log model info
                total_params = sum(p.numel() for p in self._model.parameters())
                logger.info("Model parameters: %s", f"{total_params:,}")
                
            else:
                logger.warning("Model file not found at %s", self.model_path)
                logger.info("Creating a dummy model for development")
                self._create_dummy_model()
                
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error loading model: %s", str(e))
            logger.info("Creating a dummy model for development")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for development/testing"""
        logger.info("Creating dummy PyTorch model for development")
        
        # Create a simple dummy model with same architecture
        self._model = RetinalModel(
            model_name='efficientnet_b3',
            num_classes=1,
            pretrained=True  # Use pretrained for dummy
        )
        
        self._model.to(self.device)
        self._model.eval()
        self._model_loaded = True
        logger.info("Dummy PyTorch model created successfully")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded and self._model is not None
    
    def predict(self, image: np.ndarray) -> List[PredictionResult]:
        """
        Make prediction on preprocessed image
        
        Args:
            image: Preprocessed image as numpy array (H, W, C)
            
        Returns:
            List of prediction results sorted by confidence
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert numpy array to torch tensor
            if len(image.shape) == 3:
                # Add batch dimension: (H, W, C) -> (1, C, H, W)
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            elif len(image.shape) == 4:
                # Already has batch dimension: (1, H, W, C) -> (1, C, H, W)
                image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).float()
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self._model(image_tensor)
                prediction_value = output.cpu().numpy()[0, 0]  # Get scalar prediction
            
            # Convert regression output to class probabilities
            # Your model outputs a single regression value (0-4 scale)
            predicted_class = int(round(np.clip(prediction_value, 0, 4)))
            
            # Create more realistic probabilities using Gaussian distribution around the prediction
            import scipy.stats as stats
            
            class_centers = [0, 1, 2, 3, 4]
            probabilities = []
            
            # Use Gaussian distribution centered at prediction_value with appropriate sigma
            sigma = 0.5  # Controls how spread out the probabilities are
            
            for center in class_centers:
                # Calculate probability using Gaussian distribution
                prob = stats.norm.pdf(center, loc=prediction_value, scale=sigma)
                probabilities.append(prob)
            
            # Normalize probabilities to sum to 1
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                probabilities = [0.2] * 5  # Uniform distribution as fallback
            
            # Create prediction results with proper condition mapping
            results = []
            for i, (prob, condition) in enumerate(zip(probabilities, self.class_labels)):
                # Map to detailed condition names for better understanding
                detailed_condition = self.detailed_class_names[i] if i < len(self.detailed_class_names) else condition.value
                
                result = PredictionResult(
                    condition=detailed_condition,  # Use detailed condition name instead of enum
                    confidence=float(prob),
                    probability=float(prob)
                )
                results.append(result)
            
            # Sort by confidence (descending)
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info("Prediction completed. Raw output: %.3f, "
                       "Predicted class: %d, "
                       "Top result: %s (%.3f)", 
                       prediction_value, predicted_class, 
                       results[0].condition, results[0].confidence)
            
            return results
            
        except Exception as e:
            logger.error("Prediction error: %s", str(e))
            raise RuntimeError(f"Prediction failed: {str(e)}") from e
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.is_model_loaded():
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self._model.parameters()) if self._model else 0
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad) if self._model else 0
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "model_type": "PyTorch",
            "architecture": "EfficientNet-B3 with GeM pooling",
            "input_size": self.input_size,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": len(self.class_labels),
            "class_labels": [label.value for label in self.class_labels],
            "detailed_class_names": self.detailed_class_names,
            "confidence_threshold": self.confidence_threshold
        }