from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Example timestamp constant to avoid duplication
EXAMPLE_TIMESTAMP = "2025-09-21T10:30:00Z"

class PredictionStatus(str, Enum):
    """Status of prediction"""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"

class RetinalCondition(str, Enum):
    """Possible retinal conditions"""
    NORMAL = "normal"
    DIABETIC_RETINOPATHY = "diabetic_retinopathy"
    GLAUCOMA = "glaucoma"
    MACULAR_DEGENERATION = "macular_degeneration"
    HYPERTENSIVE_RETINOPATHY = "hypertensive_retinopathy"

class PredictionRequest(BaseModel):
    """Request model for retinal image prediction"""
    patient_id: str = Field(..., description="Patient ID")
    patient_name: str = Field(..., description="Patient name")
    image_base64: str = Field(..., description="Base64 encoded retinal image")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "patient_name": "John Doe",
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
            }
        }

class PredictionResult(BaseModel):
    """Individual prediction result"""
    condition: str = Field(..., description="Predicted condition name (e.g., 'No Diabetic Retinopathy', 'Mild Diabetic Retinopathy')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability score (0-1)")

class PredictionResponse(BaseModel):
    """Response model for retinal image prediction"""
    status: PredictionStatus = Field(..., description="Prediction status")
    patient_id: str = Field(..., description="Patient ID")
    patient_name: str = Field(..., description="Patient name")
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    primary_diagnosis: str = Field(..., description="Most likely diagnosis name (e.g., 'No Diabetic Retinopathy', 'Mild Diabetic Retinopathy')")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "patient_id": "P001",
                "patient_name": "John Doe",
                "predictions": [
                    {
                        "condition": "normal",
                        "confidence": 0.85,
                        "probability": 0.85
                    },
                    {
                        "condition": "diabetic_retinopathy",
                        "confidence": 0.15,
                        "probability": 0.15
                    }
                ],
                "primary_diagnosis": "normal",
                "confidence_score": 0.85,
                "processing_time": 2.3,
                "timestamp": EXAMPLE_TIMESTAMP,
                "metadata": {
                    "model_version": "1.0.0",
                    "image_size": [224, 224]
                }
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    status: PredictionStatus = Field(default=PredictionStatus.ERROR, description="Error status")
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error_code": "INVALID_IMAGE",
                "message": "Invalid image format or corrupted image data",
                "details": {
                    "supported_formats": ["JPEG", "PNG", "TIFF"]
                },
                "timestamp": EXAMPLE_TIMESTAMP
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "timestamp": EXAMPLE_TIMESTAMP,
                "uptime": 3600.5
            }
        }