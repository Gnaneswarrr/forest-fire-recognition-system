"""
Forest Fire Recognition — Prediction Engine
=============================================

Purpose-built for detecting fire and smoke in FOREST scenes.

Three-layer analysis:
  1) CNN (InceptionV3) — primary deep-learning classifier.
  2) Forest-aware HSV colour analysis — tuned for fire/smoke against
     green canopy & woodland tones.
  3) Texture & spatial analysis — smoke reduces edge clarity; fire
     tends to appear at specific heights in forest scenes.

When the scene contains vegetation (green canopy, trees), the system
automatically tightens fire/smoke thresholds — because even a small
amount of flame/smoke in a forest image is significant.
"""

import os
import cv2
import numpy as np

_model = None
_CLASS_NAMES = ["Fire", "No Fire", "Smoke"]


def _get_model():
    """Lazy-load the Keras model on first call."""
    global _model
    if _model is not None:
        return _model

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "fire_detection_model.h5"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    from tensorflow.keras.models import load_model
    _model = load_model(model_path)
    return _model


# =====================================================================
# Layer 2 — Forest-aware colour analysis
# =====================================================================

def _detect_vegetation(hsv: np.ndarray, total_px: int) -> float:
    """
    Measure how much of the image is green vegetation / forest canopy.

    Returns a ratio [0.0 .. 1.0].  Values above ~0.10 indicate a
    forest-like scene (trees, grass, woodland).
    """
    # Broad green range covering healthy & dry vegetation
    green_lush = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # Darker forest greens (shade / undergrowth)
    green_dark = cv2.inRange(hsv, (35, 20, 20), (85, 255, 100))
    # Dry/autumn vegetation (yellowish-green)
    green_dry  = cv2.inRange(hsv, (20, 30, 50), (38, 200, 200))

    veg_mask = green_lush | green_dark | green_dry
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg_clean = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kern)
    return np.count_nonzero(veg_clean) / total_px


def _analyze_colors(image_bgr: np.ndarray):
    """
    Forest-fire-tuned HSV colour analysis.

    Returns (fire_score, smoke_score, vegetation_ratio).
    fire_score & smoke_score are each [0.0 .. 1.0].
    vegetation_ratio is the fraction of green/forest pixels.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = image_bgr.shape[:2]
    total = h * w

    # ── Vegetation detection ──
    veg_ratio = _detect_vegetation(hsv, total)
    is_forest = veg_ratio > 0.08

    # ── Fire masks ──
    # Core flame colours: red, orange, bright yellow
    fire_red1   = cv2.inRange(hsv, (0,   100, 180), (12,  255, 255))
    fire_red2   = cv2.inRange(hsv, (168, 100, 180), (180, 255, 255))
    fire_orange = cv2.inRange(hsv, (12,  100, 180), (25,  255, 255))
    fire_yellow = cv2.inRange(hsv, (22,  80,  200), (38,  255, 255))

    # Forest-fire specific: ember glow (dimmer, in undergrowth)
    ember_glow = cv2.inRange(hsv, (0, 80, 120), (18, 200, 200))

    # Hot white core of intense fire
    hot_core = cv2.inRange(hsv, (0, 0, 240), (30, 60, 255))

    fire_mask = fire_red1 | fire_red2 | fire_orange | fire_yellow | ember_glow | hot_core

    # Clean noise
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_clean = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kern)
    fire_clean = cv2.morphologyEx(fire_clean, cv2.MORPH_CLOSE, kern)
    fire_ratio = np.count_nonzero(fire_clean) / total

    # In a forest scene, even 1-2% fire pixels is a strong signal.
    # In a non-forest scene, require more.
    if is_forest:
        fire_score = min(fire_ratio / 0.015, 1.0)  # 1.5% → 1.0
    else:
        fire_score = min(fire_ratio / 0.05, 1.0)    # 5% → 1.0

    # ── Check fire-adjacent-to-green (forest fire signature) ──
    # Dilate the fire region and check overlap with vegetation
    if fire_ratio > 0.005 and is_forest:
        dilated_fire = cv2.dilate(fire_clean, kern, iterations=3)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        overlap = cv2.bitwise_and(dilated_fire, green_mask)
        overlap_ratio = np.count_nonzero(overlap) / total
        if overlap_ratio > 0.005:
            # Fire is right next to green vegetation — strong forest fire signal
            fire_score = min(fire_score * 1.5, 1.0)

    # ── Smoke masks — tuned for forest smoke ──
    # Wildfire smoke: greyish, low saturation, moderate brightness
    smoke_grey = cv2.inRange(hsv, (0, 0, 60), (180, 40, 210))
    # Blueish smoke haze (common in forest fire smoke against sky)
    smoke_blue = cv2.inRange(hsv, (90, 10, 100), (130, 60, 220))
    # Brownish smoke (burning vegetation)
    smoke_brown = cv2.inRange(hsv, (8, 20, 80), (25, 80, 180))

    smoke_mask = smoke_grey | smoke_blue | smoke_brown
    smoke_clean = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kern)
    smoke_ratio = np.count_nonzero(smoke_clean) / total

    smoke_score = 0.0

    # For forest scenes, smoke thresholds are much lower — any haze
    # above the canopy matters
    if is_forest:
        min_smoke_ratio = 0.12   # 12% of image
    else:
        min_smoke_ratio = 0.30   # 30% for non-forest

    if smoke_ratio > min_smoke_ratio:
        # Verify smoke has uniform texture (not just grey objects)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        smoke_vals = gray[smoke_clean > 0]
        if len(smoke_vals) > 100:
            std = np.std(smoke_vals)
            # Real smoke is somewhat uniform (std < 55)
            if std < 55:
                if is_forest:
                    smoke_score = min((smoke_ratio - 0.08) * 2.5, 0.9)
                else:
                    smoke_score = min((smoke_ratio - 0.25) * 2.0, 0.7)

    # Forest smoke can also appear as thin wisps — check upper part
    if is_forest and smoke_score < 0.3:
        upper_third = smoke_clean[:h // 3, :]
        upper_ratio = np.count_nonzero(upper_third) / (h // 3 * w + 1)
        if upper_ratio > 0.15:
            # Smoke concentrated above treeline
            smoke_score = max(smoke_score, min(upper_ratio * 2.0, 0.6))

    return max(0.0, fire_score), max(0.0, smoke_score), veg_ratio


# =====================================================================
# Layer 3 — Texture & spatial analysis
# =====================================================================

def _analyze_texture(image_bgr: np.ndarray) -> float:
    """
    Smoke reduces edge clarity in an image.  Compute edge density —
    low edge density suggests haze/smoke obscuring the scene.

    Returns a "smoke evidence" score [0.0 .. 0.5].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # Typical outdoor/forest image: 4-12% edges.
    # Smoke-filled: often < 3%.  Very clear: > 10%.
    if edge_density < 0.025:
        return 0.4                   # very hazy — strong smoke evidence
    elif edge_density < 0.04:
        return 0.2                   # moderately hazy
    return 0.0                       # normal clarity


def _analyze_spatial(fire_mask_binary: np.ndarray, h: int) -> float:
    """
    Check WHERE fire pixels appear in the image.
    Forest ground fires: lower 60% of image.
    Crown fires: upper 40%.
    If fire is in the lower half → stronger forest fire signal.

    Returns a boost factor [1.0 .. 1.3].
    """
    if fire_mask_binary is None or np.count_nonzero(fire_mask_binary) == 0:
        return 1.0

    fire_rows = np.where(fire_mask_binary > 0)[0]
    if len(fire_rows) == 0:
        return 1.0

    avg_y = np.mean(fire_rows) / h
    # Fire in lower-middle portion (0.3–0.8) is typical of forest fire
    if 0.25 <= avg_y <= 0.85:
        return 1.2
    return 1.0


# =====================================================================
# CNN reliability
# =====================================================================

def _cnn_reliability(preds: np.ndarray) -> float:
    """Shannon-entropy reliability: 0.0 (random) … 1.0 (confident)."""
    eps = 1e-10
    entropy = -float(np.sum(preds * np.log(preds + eps)))
    max_ent = float(np.log(len(preds)))
    return 1.0 - (entropy / max_ent) if max_ent > 0 else 0.0


# =====================================================================
# Main prediction entry point
# =====================================================================

def predict(preprocessed_batch: np.ndarray, image_path: str = None) -> dict:
    """
    Forest-fire-focused hybrid inference.

    Combines:
      1) CNN classifier (InceptionV3)
      2) Forest-aware HSV colour analysis
      3) Texture (edge-density) analysis
      4) Spatial fire-position analysis
    """
    model = _get_model()
    cnn_preds = model.predict(preprocessed_batch, verbose=0)[0]
    n = len(cnn_preds)

    # Model class order (alphabetical from flow_from_directory):
    # Index 0 = Smoke, Index 1 = fire, Index 2 = non fire
    cnn_smoke  = float(cnn_preds[0]) if n >= 1 else 0.0
    cnn_fire   = float(cnn_preds[1]) if n >= 2 else 0.0
    cnn_nofire = float(cnn_preds[2]) if n >= 3 else 1.0

    # ---------- Colour + vegetation + texture analysis ----------
    color_fire, color_smoke, veg_ratio = 0.0, 0.0, 0.0
    texture_smoke = 0.0
    spatial_boost = 1.0
    is_forest = False

    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            color_fire, color_smoke, veg_ratio = _analyze_colors(img)
            texture_smoke = _analyze_texture(img)
            is_forest = veg_ratio > 0.08

            # Spatial analysis — re-create fire mask for position check
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            fm = (cv2.inRange(hsv, (0, 100, 180), (12, 255, 255)) |
                  cv2.inRange(hsv, (168, 100, 180), (180, 255, 255)) |
                  cv2.inRange(hsv, (12, 100, 180), (25, 255, 255)))
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN, kern)
            spatial_boost = _analyze_spatial(fm, img.shape[0])

    # ---------- Fusion: blending CNN + visual analysis ----------
    rel = _cnn_reliability(cnn_preds)
    peak = float(np.max(cnn_preds))

    if peak < 0.45 and rel < 0.15:
        w_cnn = 0.20       # CNN truly uncertain — lean on colour
    elif is_forest:
        w_cnn = 0.80       # Forest scene — slightly more colour weight
    else:
        w_cnn = 0.90       # Non-forest — trust CNN more
    w_col = 1.0 - w_cnn

    # Combine colour + texture for smoke evidence
    combined_smoke = color_smoke + texture_smoke * 0.3
    combined_smoke = min(combined_smoke, 1.0)

    color_nofire = max(0.0, 1.0 - color_fire - combined_smoke)

    f_fire   = w_cnn * cnn_fire   + w_col * color_fire
    f_smoke  = w_cnn * cnn_smoke  + w_col * combined_smoke
    f_nofire = w_cnn * cnn_nofire + w_col * color_nofire

    # Apply spatial boost to fire score
    f_fire *= spatial_boost

    # Forest context boost: if vegetation is present and CNN says fire,
    # this is a genuine forest fire — boost confidence.
    if is_forest and cnn_fire > 0.35 and color_fire > 0.1:
        f_fire *= 1.15      # 15% confidence boost for confirmed forest fire

    # If forest scene with smoke AND low edge density, strengthen smoke
    if is_forest and texture_smoke > 0.15 and cnn_smoke > 0.25:
        f_smoke *= 1.1

    # Normalise to sum = 1
    total = f_fire + f_smoke + f_nofire
    if total > 0:
        f_fire   /= total
        f_smoke  /= total
        f_nofire /= total

    # ---------- Decision logic — forest-aware ----------
    no_color_evidence = (color_fire < 0.03 and color_smoke < 0.03)

    # Forest scenes: lower thresholds (even small fire in forest is critical)
    if is_forest:
        fire_thresh  = 0.35
        smoke_thresh_with_color = 0.28
        smoke_thresh_no_color   = 0.55
    else:
        fire_thresh  = 0.42
        smoke_thresh_with_color = 0.35
        smoke_thresh_no_color   = 0.65

    if f_fire >= fire_thresh and f_fire > f_nofire:
        detailed_label, label = "Fire", "FIRE DETECTED"
        confidence = f_fire * 100
    elif f_smoke > f_nofire:
        smoke_thresh = smoke_thresh_with_color if not no_color_evidence else smoke_thresh_no_color
        if f_smoke >= smoke_thresh:
            detailed_label, label = "Smoke", "FIRE DETECTED"
            confidence = f_smoke * 100
        else:
            detailed_label, label = "No Fire", "NO FIRE"
            confidence = f_nofire * 100
    else:
        detailed_label, label = "No Fire", "NO FIRE"
        confidence = f_nofire * 100

    risk = _assess_risk(detailed_label, confidence, is_forest, veg_ratio)

    return {
        "label": label,
        "detailed_label": detailed_label,
        "confidence": round(confidence, 2),
        "risk_level": risk["level"],
        "risk_color": risk["color"],
        "probabilities": {
            "fire":    round(f_fire   * 100, 2),
            "no_fire": round(f_nofire * 100, 2),
            "smoke":   round(f_smoke  * 100, 2),
        },
        "recommendation": risk["recommendation"],
    }


# =====================================================================
# Forest-specific risk assessment
# =====================================================================

def _assess_risk(detail: str, conf: float, is_forest: bool = False,
                 veg_ratio: float = 0.0) -> dict:
    """
    Map detection result + confidence to risk level.
    In forest scenes the recommendations are forest-specific.
    """
    if detail == "Fire":
        if conf > 85:
            if is_forest:
                return {"level": "Critical", "color": "#D32F2F",
                        "recommendation": (
                            "CRITICAL FOREST FIRE: Immediate response needed! "
                            "Alert the nearest forest ranger station & fire department. "
                            "Evacuate all personnel from the area. "
                            "Establish firebreaks if trained to do so."
                        )}
            return {"level": "Critical", "color": "#D32F2F",
                    "recommendation": "CRITICAL: Immediate evacuation required. Contact emergency services (911) now."}
        if conf > 60:
            if is_forest:
                return {"level": "High", "color": "#F44336",
                        "recommendation": (
                            "HIGH RISK — FOREST FIRE: Active fire detected in woodland area. "
                            "Report coordinates to forest fire control center. "
                            "Begin controlled evacuation of nearby forest zones."
                        )}
            return {"level": "High", "color": "#F44336",
                    "recommendation": "HIGH RISK: Active fire detected. Begin evacuation and alert authorities."}
        if is_forest:
            return {"level": "Moderate", "color": "#FF9800",
                    "recommendation": (
                        "MODERATE — POSSIBLE FOREST FIRE: Fire activity detected in forest area. "
                        "Dispatch ranger patrol to verify. Prepare water tankers & aerial support standby."
                    )}
        return {"level": "Moderate", "color": "#FF9800",
                "recommendation": "MODERATE: Possible fire activity. Monitor closely and prepare to evacuate."}

    if detail == "Smoke":
        if conf > 75:
            if is_forest:
                return {"level": "High", "color": "#FF5722",
                        "recommendation": (
                            "HIGH RISK — FOREST SMOKE: Dense smoke detected over woodland. "
                            "Active fire likely nearby or underground (smoldering). "
                            "Deploy ground crew & aerial reconnaissance to locate the fire source. "
                            "Close forest trails in the area."
                        )}
            return {"level": "High", "color": "#FF5722",
                    "recommendation": "HIGH RISK: Dense smoke detected — fire likely nearby. Investigate immediately."}
        if is_forest:
            return {"level": "Moderate", "color": "#FF9800",
                    "recommendation": (
                        "MODERATE — FOREST SMOKE: Smoke detected above tree canopy. "
                        "Could indicate a developing wildfire or controlled burn. "
                        "Verify with ranger station — deploy lookout if unscheduled."
                    )}
        return {"level": "Moderate", "color": "#FF9800",
                "recommendation": "MODERATE: Smoke visible. Could indicate a developing fire — stay alert."}

    # No fire
    if is_forest and conf > 70:
        return {"level": "Safe", "color": "#2E7D32",
                "recommendation": (
                    "SAFE — FOREST CLEAR: No fire or smoke detected in this forest area. "
                    f"Vegetation coverage: {veg_ratio*100:.0f}%. Canopy appears healthy."
                )}
    if conf > 80:
        return {"level": "Safe", "color": "#2E7D32",
                "recommendation": "SAFE: No fire or smoke detected. Area appears clear."}
    return {"level": "Low", "color": "#4CAF50",
            "recommendation": "LOW RISK: Likely safe, but confidence is moderate. Re-check if uncertain."}
