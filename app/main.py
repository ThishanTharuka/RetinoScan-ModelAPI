from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import os
from dotenv import load_dotenv

from app.routers import prediction, health
from app.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger()

# Create FastAPI app
app = FastAPI(
    title="RetinoScan Model API",
    description="API for retinal image analysis using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get allowed origins from environment
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
if not allowed_origins or allowed_origins == [""]:
    allowed_origins = ["*"]  # Allow all origins in development

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure properly for production
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting RetinoScan Model API...")
    # Initialize model loading here
    from app.services.model_service import ModelService
    ModelService.get_instance()
    logger.info("Model loaded successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RetinoScan Model API...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RetinoScan Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )