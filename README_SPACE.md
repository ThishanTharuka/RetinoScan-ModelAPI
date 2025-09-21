Deploying RetinoScan Model API to Hugging Face Spaces

Two options to host on HF Spaces:

A) Docker-based Space (recommended for full control)
- Add the provided `Dockerfile` to the repo root (already included).
- Create a new Space on Hugging Face and choose "Docker" as the runtime. Push this repo to the Space.
- The Dockerfile exposes port 8080 and runs the FastAPI app at `app.main:app`.

B) Gradio-based Space (quick demo)
- Use `space_app.py` (Gradio wrapper) and create a new Space with the Gradio runtime.
- You can set `MODEL_API_URL` environment variable in Space to point to the internal URL (if using a Docker Space) or an externally hosted model API.

Notes and limitations
- Hugging Face Spaces free tier provides CPU-only machines for most users. No free GPUs.
- Long-running heavy models may run out of memory or be slow; consider model quantization or a lighter model for Spaces.
- If model weights are large, keep them in `models/` and reference them from within the app. Large files may increase build time.
- For private model files, consider using HF Hub artifacts (requires auth) or download them at runtime from a secure storage bucket.

Quick deploy steps (Docker Space):
1. Push the `model-api` repo (with Dockerfile) to a new Hugging Face Space (Docker runtime).
2. Set any secrets in the Space settings (ENV vars) like `ALLOWED_ORIGINS` or API keys.
3. Build logs are visible in the Space — watch for large wheel downloads (torch).
4. After successful build, the Space will be live. Use the Space URL in your frontend as the API endpoint.

If you want, I can:
- Add a small `hf_space` script that automates building a lightweight demo (download smaller pretrained weights).
- Create a `space` directory with minimal files needed by the Spaces UI.
- Walk through creating the Space on the Hugging Face website and map the repo.
