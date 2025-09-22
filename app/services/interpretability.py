import io
import base64
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """
    Robust Grad-CAM implementation that hooks the last Conv2d layer
    of the model (preferred) and computes gradients via autograd.grad.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.activations = None  # will hold activation tensor (with grad_fn)

        # Find the last Conv2d module to hook
        last_conv = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_conv_name = name
                break

        if last_conv is not None:
            try:
                last_conv.register_forward_hook(self._forward_hook)
                logger.info("GradCAM: hooks registered on last Conv2d: %s", last_conv_name)
            except Exception as e:
                logger.error("GradCAM: failed to register forward hook on %s: %s", last_conv_name, str(e))
                # Fallback: hook entire model
                self.model.register_forward_hook(self._forward_hook)
                logger.info("GradCAM: fallback hook registered on model")
        else:
            # Fallback: hook the model
            self.model.register_forward_hook(self._forward_hook)
            logger.info("GradCAM: no Conv2d found; hook registered on model (fallback)")

    def _forward_hook(self, module, input, output):
        # Store activations WITHOUT detaching so we can compute grads w.r.t. them
        self.activations = output

    def _preprocess_to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        # Expect input image_np in (H, W, C) with pixel range 0-255 or 0-1
        img = image_np.astype(np.float32)
        if img.max() > 2.0:
            img = img / 255.0

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        return tensor

    def generate_cam(self, input_image: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the provided input image.
        """
        try:
            tensor = self._preprocess_to_tensor(input_image)

            # Ensure gradients are tracked for inputs so activations have grad_fn
            tensor.requires_grad_(True)

            # Forward pass with gradient tracking enabled
            with torch.set_grad_enabled(True):
                output = self.model(tensor)

            if self.activations is None:
                raise RuntimeError("GradCAM hooks did not capture activations")

            # Determine target scalar for backprop
            if class_idx is None:
                if output.numel() == 1 or (output.dim() == 1 or output.shape[-1] == 1):
                    target = output.squeeze()
                else:
                    pred = output.argmax(dim=1)
                    class_idx = int(pred.item())
                    target = output[0, class_idx]
            else:
                if output.numel() == 1 or (output.dim() == 1 or output.shape[-1] == 1):
                    target = output.squeeze()
                else:
                    target = output[0, class_idx]

            # Compute gradients of the target w.r.t. the activations
            # activations shape: (B, C, H, W)
            grads = torch.autograd.grad(outputs=target, inputs=self.activations, retain_graph=True, allow_unused=True)[0]

            if grads is None:
                raise RuntimeError("GradCAM could not compute gradients for activations")

            # Global-average pool the gradients to obtain weights
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

            # Weighted combination of activations
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (B, 1, H, W)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(input_image.shape[0], input_image.shape[1]), mode='bilinear', align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()

            cam -= cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_uint8 = (cam * 255).astype('uint8')
            return cam_uint8

        except Exception as e:
            logger.error("Error generating Grad-CAM: %s", str(e))
            raise


def overlay_heatmap_on_image(
    image_np: np.ndarray,
    heatmap_np: np.ndarray,
    alpha: float = 0.5,
    mark_spots: bool = True,
    threshold: float = 0.5,
    min_area: int = 50,
) -> str:
    """
    Overlay a heatmap (H,W uint8) onto the original RGB image and return base64 PNG.

    If `mark_spots` is True the function will attempt to find high-activation regions
    and draw red circles around them to highlight abnormal spots detected by the
    model. Detection prefers OpenCV contour-based grouping; when OpenCV is not
    available a simple greedy clustering on thresholded pixels is used.
    """
    try:
        # Ensure heatmap is single-channel uint8
        if heatmap_np.dtype != np.uint8:
            heatmap_np = np.clip(heatmap_np, 0, 255).astype('uint8')
        # Prefer OpenCV colormap (avoids matplotlib dependency)
        try:
            import cv2

            # Resize heatmap to match image if needed
            if heatmap_np.shape[:2] != image_np.shape[:2]:
                heatmap_np = cv2.resize(heatmap_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)

            # applyColorMap expects single-channel 8-bit
            colored_bgr = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
            # Convert BGR -> RGB
            colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e_cv:
            # Fallback: if cv2 not available, create a simple grayscale 3-channel map
            logger.warning("OpenCV colormap unavailable, falling back to grayscale: %s", str(e_cv))
            if heatmap_np.shape[:2] != image_np.shape[:2]:
                # Use PIL for resizing if cv2 missing
                im_hm = Image.fromarray(heatmap_np)
                im_hm = im_hm.resize((image_np.shape[1], image_np.shape[0]), Image.BILINEAR)
                colored = np.stack([np.array(im_hm)] * 3, axis=-1)
            else:
                colored = np.stack([heatmap_np] * 3, axis=-1)

        # Blend overlay
        overlay = ((1 - alpha) * image_np.astype('float32') + alpha * colored.astype('float32')).astype('uint8')

        # Marker detection and drawing were removed per user request.
        # The function will only produce the colored overlay and return it as a base64 PNG.

        # Return blended overlay as PNG base64 (no markers)
        im = Image.fromarray(overlay)
        buffer = io.BytesIO()
        im.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        logger.error("Error overlaying heatmap: %s", str(e))
        raise
