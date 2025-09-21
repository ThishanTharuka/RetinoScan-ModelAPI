# Optional lightweight Gradio demo for Hugging Face Spaces
# This file is optional — Spaces can run the Dockerfile. If you prefer the "Gradio" app
# runtime on Spaces, use `app.py` or `space_app.py` as the entrypoint and omit Docker.

import gradio as gr
import base64
import requests
import os

API_URL = os.getenv('MODEL_API_URL', 'http://127.0.0.1:8080/api/v1')

def predict_from_file(image):
    # Gradio provides a PIL image. Convert to bytes and send to /predict/upload
    import io
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    files = {'file': ('upload.png', buf, 'image/png')}
    data = {'patient_id': 'demo', 'patient_name': 'demo'}
    try:
        r = requests.post(f"{API_URL}/predict/upload", files=files, data=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks() as demo:
    gr.Markdown("# RetinoScan - Demo\nUpload a retinal image and get predictions from the hosted model API.")
    img = gr.Image(type='pil')
    btn = gr.Button('Predict')
    out = gr.JSON()

    btn.click(predict_from_file, inputs=img, outputs=out)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 8080)))
