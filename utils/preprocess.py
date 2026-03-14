"""
Image Preprocessing Pipeline — Forest Fire Recognition
========================================================

Optimised for forest / aerial / drone imagery:
  1. CLAHE contrast enhancement — balances dark undergrowth with
     bright sky / flames so the CNN sees detail in both regions.
  2. Light denoising — removes sensor noise without blurring smoke.
  3. Normalization — [0, 1] float32 for InceptionV3.
"""

import cv2
import numpy as np

TARGET_SIZE = (224, 224)


def load_and_preprocess(image_path: str):
    """
    Full preprocessing pipeline for a single image file.
    Returns (model_input, display_image) or (None, None) on failure.
    """
    raw = cv2.imread(image_path)
    if raw is None:
        return None, None
    return _pipeline(raw)


def preprocess_frame(frame: np.ndarray):
    """
    Preprocess a camera frame (already in BGR).
    Returns (model_input, display_image) or (None, None) on failure.
    """
    if frame is None or frame.size == 0:
        return None, None
    return _pipeline(frame)


def _pipeline(bgr_image: np.ndarray):
    """
    Core preprocessing pipeline — forest-optimised.

    Steps:
      BGR → resize → CLAHE contrast enhancement → light denoise →
      RGB → normalize → expand_dims.
    """
    resized = cv2.resize(bgr_image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # CLAHE on the L channel of LAB colour space.
    # This greatly improves visibility in forest images where
    # dark green canopy and bright sky / flames coexist.
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Light denoising — preserve smoke texture (lower strength than before)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 4, 4, 7, 21)

    # Convert to RGB for the model
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batch = np.expand_dims(normalized, axis=0)

    display = resized.copy()  # keep original colours for UI
    return batch, display
