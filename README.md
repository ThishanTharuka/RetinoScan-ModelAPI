# RetinoScan Model API

A FastAPI-based machine learning service for diabetic retinopathy detection and classification using EfficientNet-B3 with GeM pooling.

## 🎯 Overview

This API provides real-time diabetic retinopathy classification across 5 severity levels:
- **Level 0**: No Diabetic Retinopathy
- **Level 1**: Mild Diabetic Retinopathy  
- **Level 2**: Moderate Diabetic Retinopathy
- **Level 3**: Severe Diabetic Retinopathy
- **Level 4**: Proliferative Diabetic Retinopathy

## 🏗️ Architecture

- **Framework**: FastAPI 0.110.0
- **ML Model**: EfficientNet-B3 with GeM pooling
- **Deep Learning**: PyTorch 2.1.0+
- **Image Processing**: OpenCV, PIL, Albumentations
- **Deployment**: Uvicorn ASGI server

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM (8GB recommended)
- GPU optional (CUDA support)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/RetinoScan-ModelAPI.git
   cd RetinoScan-ModelAPI
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Download model file**
   - Place your trained model (`best_model.pth`) in the `models/` directory
   - See `models/README.md` for model requirements

6. **Start the server**
   ```bash
   python start.py
   ```

The API will be available at:
- **API**: http://localhost:8001
- **Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/api/v1/health

## 📡 API Endpoints

### Health Check
```http
GET /api/v1/health
```

### Prediction (Base64)
```http
POST /api/v1/predict
Content-Type: application/json

{
  "patient_id": "P001",
  "patient_name": "John Doe",
  "image_base64": "data:image/jpeg;base64,..."
}
```

### Prediction (File Upload)
```http
POST /api/v1/predict/upload
Content-Type: multipart/form-data

file: (image file)
patient_id: P001
patient_name: John Doe
```

### Model Information
```http
GET /api/v1/model/info
```

## 📊 Response Format

```json
{
  "status": "success",
  "patient_id": "P001",
  "patient_name": "John Doe",
  "predictions": [
    {
      "condition": "No Diabetic Retinopathy",
      "confidence": 0.7371,
      "probability": 0.7371
    },
    {
      "condition": "Mild Diabetic Retinopathy",
      "confidence": 0.2612,
      "probability": 0.2612
    }
  ],
  "primary_diagnosis": "No Diabetic Retinopathy",
  "confidence_score": 0.7371,
  "processing_time": 0.75,
  "timestamp": "2025-09-22T01:33:37.365372",
  "metadata": {
    "model_version": "1.0.0",
    "model_architecture": "EfficientNet-B3 with GeM pooling",
    "preprocessing": "Crop black background + Resize to 300x300 + ImageNet normalization",
    "image_size": [300, 300],
    "file_name": "retinal_image.png",
    "file_size": 2220460
  }
}
```

## ⚙️ Configuration

Environment variables in `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
DEBUG=True

# Model Configuration
MODEL_PATH=./models/best_model.pth
MODEL_INPUT_SIZE=300
CONFIDENCE_THRESHOLD=0.5

# CORS Settings
ALLOWED_ORIGINS=http://localhost:4200,http://localhost:3000

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
```

## 🧪 Testing

Run the test suite:
```bash
python test_fix.py          # Test prediction accuracy
python test_health.py       # Test health endpoint
python test_imports.py      # Test dependencies
```

## 📁 Project Structure

```
RetinoScan-ModelAPI/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pytorch_model.py # PyTorch model definition
│   │   └── schemas.py       # Pydantic schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py        # Health check endpoints
│   │   └── prediction.py    # Prediction endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_service.py # Image preprocessing
│   │   └── model_service.py # ML model service
│   └── utils/
│       ├── __init__.py
│       └── logger.py        # Logging configuration
├── models/
│   ├── README.md           # Model file documentation
│   └── best_model.pth      # Trained model (not in git)
├── logs/                   # Application logs
├── tests/                  # Test files
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── start.py             # Server startup script
└── README.md           # This file
```

## 🔬 Model Details

- **Architecture**: EfficientNet-B3 backbone with Global Error Max (GeM) pooling
- **Input Size**: 300x300 RGB images
- **Preprocessing**: 
  - Black background cropping
  - Resize to 300x300
  - ImageNet normalization
- **Output**: 5-class probability distribution using Gaussian distribution
- **Performance**: ~0.75s processing time on CPU

## 🌐 CORS Configuration

The API supports cross-origin requests from configured domains. Update `ALLOWED_ORIGINS` in `.env` for production deployment.

## 📝 Logging

Logs are written to:
- Console (development)
- `logs/retinoscan_api.log` (file)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## 🔒 Security Features

- Input validation with Pydantic
- File type and size validation
- CORS protection
- Request rate limiting (configurable)
- Secure headers

## 🚀 Deployment

### Docker (Recommended)
```bash
docker build -t retinoscan-model-api .
docker run -p 8001:8001 retinoscan-model-api
```

### Production
- Use a reverse proxy (nginx)
- Enable HTTPS
- Configure proper CORS origins
- Set up monitoring and logging
- Use a process manager (PM2, systemd)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- GitHub Issues: [RetinoScan-ModelAPI Issues](https://github.com/YourUsername/RetinoScan-ModelAPI/issues)
- Documentation: [API Docs](http://localhost:8001/docs)

## 🏥 Medical Disclaimer

This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.