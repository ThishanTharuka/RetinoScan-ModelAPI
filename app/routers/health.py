from fastapi import APIRouter
from app.models.schemas import HealthResponse
import time

router = APIRouter()

# Store startup time for uptime calculation
startup_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns the current status of the API service and model.
    """
    from app.services.model_service import ModelService
    
    model_service = ModelService.get_instance()
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_service.is_model_loaded(),
        uptime=uptime
    )

@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/Docker health checks
    """
    from app.services.model_service import ModelService
    
    model_service = ModelService.get_instance()
    
    if not model_service.is_model_loaded():
        return {"status": "not ready", "reason": "model not loaded"}, 503
    
    return {"status": "ready"}

@router.get("/health/live")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes/Docker health checks
    """
    return {"status": "alive"}