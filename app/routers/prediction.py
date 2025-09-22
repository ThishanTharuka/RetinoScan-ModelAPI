from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import base64
from app.models.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    PredictionStatus
)
from app.services.model_service import ModelService
from app.services.image_service import ImageService
from app.services.interpretability import GradCAM, overlay_heatmap_on_image
from app.utils.logger import get_logger
import time

router = APIRouter()
logger = get_logger(__name__)

@router.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns detailed information about the current model including
    architecture, parameters, and configuration.
    """
    try:
        model_service = ModelService.get_instance()
        model_info = model_service.get_model_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": model_info,
                "message": "Model information retrieved successfully"
            }
        )
    except Exception as e:
        logger.error("Error getting model info: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        ) from e

@router.post("/predict", response_model=PredictionResponse)
async def predict_retinal_condition(request: PredictionRequest):
    """
    Predict retinal condition from base64 encoded image
    
    Takes a base64 encoded retinal image and returns predictions
    for diabetic retinopathy levels (0-4 scale).
    
    The model predicts:
    - 0: No Diabetic Retinopathy
    - 1: Mild Diabetic Retinopathy
    - 2: Moderate Diabetic Retinopathy  
    - 3: Severe Diabetic Retinopathy
    - 4: Proliferative Diabetic Retinopathy
    """
    start_time = time.time()
    
    try:
        # Get services
        model_service = ModelService.get_instance()
        image_service = ImageService()
        
        # Validate model is loaded
        if not model_service.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later."
            )
        
        # Process image using the exact same pipeline as training
        try:
            image = image_service.decode_base64_image(request.image_base64)
            
            # Validate image
            if not image_service.validate_image(image):
                raise ValueError("Image validation failed")
            
            processed_image = image_service.preprocess_image(image)
        except Exception as e:
            logger.error("Image processing error: %s", str(e))
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            ) from e
        
        # Make prediction
        predictions = model_service.predict(processed_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = PredictionResponse(
            status=PredictionStatus.SUCCESS,
            patient_id=request.patient_id,
            patient_name=request.patient_name,
            predictions=predictions,
            primary_diagnosis=predictions[0].condition if predictions else "normal",
            confidence_score=predictions[0].confidence if predictions else 0.0,
            processing_time=processing_time,
            metadata={
                "model_version": "1.0.0",
                "model_architecture": "EfficientNet-B3 with GeM pooling",
                "preprocessing": "Crop black background + Resize to 300x300 + ImageNet normalization",
                "image_size": processed_image.shape[:2] if hasattr(processed_image, 'shape') else None
            }
        )
        
        logger.info("Prediction completed for patient %s in %.2fs", request.patient_id, processing_time)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@router.post("/predict/upload", response_model=PredictionResponse)
async def predict_from_upload(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    patient_name: str = Form(...)
):
    """
    Predict retinal condition from uploaded image file
    
    Takes an uploaded image file and returns predictions
    for diabetic retinopathy levels (0-4 scale).
    
    Supported formats: JPG, PNG, TIFF
    Maximum file size: 10MB
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPG, PNG, TIFF)"
            )
        
        # Get services
        model_service = ModelService.get_instance()
        image_service = ImageService()
        
        # Validate model is loaded
        if not model_service.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later."
            )
        
        # Read and process image using the exact same pipeline as training
        try:
            image_data = await file.read()
            image = image_service.decode_image_bytes(image_data)
            
            # Validate image
            if not image_service.validate_image(image):
                raise ValueError("Image validation failed")
            
            processed_image = image_service.preprocess_image(image)
        except Exception as e:
            logger.error("Image processing error: %s", str(e))
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            ) from e
        
        # Make prediction
        predictions = model_service.predict(processed_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = PredictionResponse(
            status=PredictionStatus.SUCCESS,
            patient_id=patient_id,
            patient_name=patient_name,
            predictions=predictions,
            primary_diagnosis=predictions[0].condition if predictions else "normal",
            confidence_score=predictions[0].confidence if predictions else 0.0,
            processing_time=processing_time,
            metadata={
                "model_version": "1.0.0",
                "model_architecture": "EfficientNet-B3 with GeM pooling",
                "preprocessing": "Crop black background + Resize to 300x300 + ImageNet normalization",
                "image_size": processed_image.shape[:2] if hasattr(processed_image, 'shape') else None,
                "file_name": file.filename,
                "file_size": len(image_data)
            }
        )
        
        logger.info("Prediction completed for patient %s in %.2fs", patient_id, processing_time)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/interpret/gradcam")
async def interpret_gradcam(request: PredictionRequest | None = None, file: UploadFile | None = None):
    """
    Generate a Grad-CAM heatmap overlay for a provided retinal image.

    Accepts either a JSON body with base64 `image_base64` (PredictionRequest)
    or a multipart upload file. Returns a base64 PNG overlay.
    """
    try:
        model_service = ModelService.get_instance()
        image_service = ImageService()

        if not model_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Load image from either request body or uploaded file
        if file is not None:
            image_bytes = await file.read()
            image = image_service.decode_image_bytes(image_bytes)
        elif request is not None and getattr(request, 'image_base64', None):
            image = image_service.decode_base64_image(request.image_base64)
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        # Validate and preprocess (we want original sized image for overlay, but use preprocess for model)
        if not image_service.validate_image(image):
            raise HTTPException(status_code=400, detail="Invalid image data")

        preprocessed = image_service.preprocess_image(image)

        # Instantiate GradCAM with underlying PyTorch model
        gradcam = GradCAM(model_service._model)

        # Generate CAM on the preprocessed image (note: preprocessed is normalized)
        # convert back to RGB uint8 expected by GradCAM helper
        try:
            # If preprocessed values are normalized floats (C,H,W converted to H,W,C), rescale to 0-255
            img_for_cam = (preprocessed * 255.0).astype('uint8') if preprocessed.max() <= 1.0 else preprocessed.astype('uint8')
        except Exception:
            img_for_cam = preprocessed.astype('uint8')

        cam = gradcam.generate_cam(img_for_cam)

        # Overlay heatmap on the resized image used for model (makes visual alignment easier)
        overlay_b64 = overlay_heatmap_on_image(img_for_cam, cam, alpha=0.5)

        return JSONResponse(status_code=200, content={"status": "success", "heatmap": overlay_b64})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("GradCAM endpoint error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"GradCAM failed: {str(e)}") from e


@router.get("/interpret/heatmap/latest")
async def get_latest_heatmap():
    """
    Return the most recently generated heatmap image (if present) as a data URL.
    This is useful for quick frontend previewing when a heatmap was generated on the server
    and saved to a known path (heatmap_output.png in the project root).
    """
    try:
        # Save path is relative to the repository root (model-api/heatmap_output.png)
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        heatmap_path = os.path.join(repo_root, 'heatmap_output.png')
        if not os.path.exists(heatmap_path):
            raise HTTPException(status_code=404, detail="No heatmap found")

        # Read file in threadpool to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_running_loop()
        def _read():
            with open(heatmap_path, 'rb') as fh:
                return fh.read()

        data = await loop.run_in_executor(None, _read)
        b64 = base64.b64encode(data).decode('ascii')
        return JSONResponse(status_code=200, content={"status": "success", "heatmap": f"data:image/png;base64,{b64}"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error returning latest heatmap: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to return latest heatmap: {str(e)}") from e