"""
model_loader.py
---------------
Optional ML model integration for deepfake detection.

This module provides a plug-in interface for loading a pre-trained classifier
(e.g., trained on ASVspoof 2019 dataset) that can replace or augment the
rule-based Trust Score in deepfake_detector.py.

If no model is found, the system gracefully falls back to the rule-based approach.

Recommended model approach for ASVspoof 2019:
- Features: 40 MFCCs + pitch jitter + spectral features (flattened to 1D vector)
- Model: sklearn GaussianNB, LogisticRegression, or a lightweight MLP
- Training: PA (physical access) or LA (logical access) subset
- Label: 'bonafide' (real=1) vs 'spoof' (fake=0)
"""

import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "asv_classifier.pkl")

_model = None
_model_loaded = False


def load_model():
    """
    Attempt to load a pre-trained sklearn model from disk.
    Returns (model, True) if successful, (None, False) otherwise.
    """
    global _model, _model_loaded

    if _model_loaded:
        return _model, _model is not None

    if not os.path.exists(MODEL_PATH):
        _model_loaded = True
        return None, False

    try:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _model_loaded = True
        print(f"[ModelLoader] Loaded pre-trained model from {MODEL_PATH}")
        return _model, True
    except Exception as e:
        print(f"[ModelLoader] Failed to load model: {e}")
        _model_loaded = True
        return None, False


def build_feature_vector(features: dict) -> np.ndarray:
    """
    Build a flat feature vector from extracted features for ML inference.
    Matches the format expected by a model trained on ASVspoof 2019 features.
    """
    mfcc_mean = features["mfcc"]["mfcc_mean"]
    mfcc_std = features["mfcc"]["mfcc_std"]

    scalar_features = np.array([
        features["mfcc"]["mfcc_var_mean"],
        features["pitch"]["pitch_jitter"],
        features["pitch"]["pitch_std"],
        features["pitch"]["voiced_ratio"],
        features["spectral"]["spectral_centroid_mean"],
        features["spectral"]["spectral_flatness_mean"],
        features["spectral"]["spectral_bandwidth_mean"],
        features["spectral"]["zcr_std"],
        features["energy"]["rms_var"],
        features["energy"]["rms_std"],
    ])

    return np.concatenate([mfcc_mean, mfcc_std, scalar_features])


def predict_with_model(features: dict) -> dict:
    """
    Run ML model inference if a model is available.
    Returns probability of being 'bonafide' (real human) as a score 0–100.
    Falls back to None if no model is loaded.
    """
    model, available = load_model()

    if not available or model is None:
        return {
            "ml_available": False,
            "ml_trust_score": None,
            "ml_verdict": None,
        }

    try:
        feature_vec = build_feature_vector(features).reshape(1, -1)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feature_vec)[0]
            classes = list(model.classes_)
            real_idx = classes.index(1) if 1 in classes else -1
            score = float(proba[real_idx] * 100) if real_idx >= 0 else float(proba[1] * 100)
        else:
            prediction = model.predict(feature_vec)[0]
            score = 85.0 if prediction == 1 else 20.0

        return {
            "ml_available": True,
            "ml_trust_score": round(score, 1),
            "ml_verdict": "LIKELY HUMAN" if score >= 60 else "LIKELY AI / DEEPFAKE",
        }

    except Exception as e:
        return {
            "ml_available": False,
            "ml_trust_score": None,
            "ml_verdict": None,
            "ml_error": str(e),
        }
