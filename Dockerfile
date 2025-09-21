# Dockerfile for deploying RetinoScan Model API on Hugging Face Spaces (Docker runtime)
# Uses Uvicorn + FastAPI entrypoint at app.main:app

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system deps required by some packages (opencv, pillow, torch wheels may need gcc libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libgl1 \
  git \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Optional: install Gradio for an interactive demo UI (Space will be able to serve a demo)
RUN pip install --no-cache-dir gradio

# Copy the application code
COPY . /app

# Expose the port HF Spaces expects
EXPOSE ${PORT}

# Default command - run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
