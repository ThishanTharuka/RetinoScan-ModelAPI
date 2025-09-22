import base64
import numpy as np
from PIL import Image
import cv2
import io
import os
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ImageService:
    """
    Service for image processing operations matching the training preprocessing pipeline
    """
    
    def __init__(self):
        self.target_size = int(os.getenv("MODEL_INPUT_SIZE", "300"))
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
        # Albumentations transform for inference (matching training)
        self.transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], p=1.0)
    
    def crop_black_background(self, image, tol=7):
        """
        Crop black background from retinal images
        This matches the preprocessing used in training
        """
        if image.ndim == 2:
            mask = image > tol
            return image[np.ix_(mask.any(1), mask.any(0))]
        elif image.ndim == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            if image[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
                return image
            else:
                image1 = image[:,:,0][np.ix_(mask.any(1), mask.any(0))]
                image2 = image[:,:,1][np.ix_(mask.any(1), mask.any(0))]
                image3 = image[:,:,2][np.ix_(mask.any(1), mask.any(0))]
                return np.stack([image1, image2, image3], axis=-1)
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Decode base64 string to image array
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL Image object
        """
        try:
            # Handle data URL format (data:image/jpeg;base64,...)
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Check file size
            if len(image_data) > self.max_file_size:
                raise ValueError(f"Image too large. Maximum size: {self.max_file_size} bytes")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))

            # Auto-resize/convert large or problematic images to acceptable limits
            image = self._ensure_acceptable_image(image)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            logger.info(f"Decoded image: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise ValueError(f"Invalid base64 image data: {str(e)}")
    
    def decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode image bytes to image array
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image as numpy array
        """
        try:
            # Check file size
            if len(image_bytes) > self.max_file_size:
                raise ValueError(f"Image too large. Maximum size: {self.max_file_size} bytes")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Auto-resize/convert large or problematic images to acceptable limits
            image = self._ensure_acceptable_image(image)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            logger.info(f"Decoded image: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error decoding image bytes: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model prediction using the exact same pipeline as training
        
        Args:
            image: Raw image as numpy array
            
        Returns:
            Preprocessed image ready for model (as numpy array, not tensor)
        """
        try:
            # Step 1: Crop black background (matching training preprocessing)
            cropped_image = self.crop_black_background(image.copy())
            logger.info(f"After cropping: {cropped_image.shape}")
            
            # Step 2: Resize to target size (matching training)
            resized_image = cv2.resize(cropped_image, (self.target_size, self.target_size))
            logger.info(f"After resizing: {resized_image.shape}")
            
            # Step 3: Apply Albumentations transform (normalization + any other transforms)
            transformed = self.transform(image=resized_image)
            processed_image = transformed['image']
            
            # Convert tensor back to numpy array for consistency with model service
            if hasattr(processed_image, 'numpy'):
                processed_image = processed_image.numpy()
            
            # Ensure image is in (H, W, C) format for the model service
            if len(processed_image.shape) == 3 and processed_image.shape[0] == 3:
                # Convert from (C, H, W) to (H, W, C)
                processed_image = processed_image.transpose(1, 2, 0)
            
            logger.info(f"Final preprocessed image: {processed_image.shape}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size using OpenCV (matching training)
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        # Use OpenCV for high-quality resizing (matching training code)
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Ensure RGB format (OpenCV uses BGR by default, but we're already in RGB)
        return resized
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate image meets requirements
        
        Args:
            image: Image to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check dimensions
            if len(image.shape) != 3:
                logger.error("Image must be 3-dimensional (height, width, channels)")
                return False
            
            # Check channels
            if image.shape[2] != 3:
                logger.error("Image must have 3 channels (RGB)")
                return False
            
            # Check size constraints
            height, width = image.shape[:2]
            if height < 32 or width < 32:
                logger.error("Image too small (minimum 32x32 pixels)")
                return False
            
            if height > 4096 or width > 4096:
                logger.error("Image too large (maximum 4096x4096 pixels)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation error: {str(e)}")
            return False
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Get information about an image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image information
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size_mb": image.nbytes / (1024 * 1024),
            "min_value": float(np.min(image)),
            "max_value": float(np.max(image)),
            "mean_value": float(np.mean(image))
        }

    def _ensure_acceptable_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Ensure a PIL Image is within file size and dimension constraints.

        If the image dimensions exceed the maximum allowed by the service
        or the in-memory JPEG representation is too large, downscale the
        image and re-encode to JPEG in-memory with reasonable quality.

        Returns a PIL.Image (RGB).
        """
        try:
            # Ensure RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Quick dimension check
            w, h = pil_image.size
            max_dim = 3000  # chosen conservative max side to keep quality
            if max(w, h) > max_dim:
                ratio = max_dim / float(max(w, h))
                new_size = (int(w * ratio), int(h * ratio))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)

            # Re-encode to JPEG in-memory and check size
            buf = io.BytesIO()
            pil_image.save(buf, format='JPEG', quality=92)
            size_bytes = buf.tell()

            # If still too large, progressively reduce quality
            quality = 92
            while size_bytes > self.max_file_size and quality >= 50:
                buf = io.BytesIO()
                quality -= 8
                pil_image.save(buf, format='JPEG', quality=quality)
                size_bytes = buf.tell()

            # If we've reduced quality below threshold and still too large,
            # scale down further until it fits or minimal quality reached
            while size_bytes > self.max_file_size:
                w, h = pil_image.size
                new_size = (int(w * 0.85), int(h * 0.85))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                pil_image.save(buf, format='JPEG', quality=quality)
                size_bytes = buf.tell()

            # Return PIL Image (already RGB)
            return pil_image
        except Exception as e:
            logger.warning(f"_ensure_acceptable_image failed: {str(e)}; returning original image")
            try:
                return pil_image.convert('RGB')
            except Exception:
                return pil_image