import numpy as np
import librosa
from core.feature_extractor import extract_all_features


def score_mfcc(mfcc_var_mean: float) -> float:
    """
    Non-monotonic MFCC variance scorer.
    Human speech sits in a natural range (80–320).
    Too LOW = AI (unnaturally smooth/flat synthesis).
    Too HIGH = AI (over-expressive neural synthesis, e.g. ElevenLabs).
    Returns 100 in the human range, drops off outside it.
    """
    optimal_low = 80.0
    optimal_high = 320.0

    if mfcc_var_mean < optimal_low:
        return max(0.0, (mfcc_var_mean / optimal_low) * 100.0)
    elif mfcc_var_mean <= optimal_high:
        return 100.0
    else:
        excess = mfcc_var_mean - optimal_high
        penalty_range = optimal_high * 3.0
        return max(0.0, 100.0 * (1.0 - excess / penalty_range))


def score_pitch_jitter(jitter: float, voiced_ratio: float) -> float:
    """
    Score based on pitch jitter and voiced ratio.
    Human jitter typically 0.003–0.030. Very low = robotic; very high = may indicate
    artificially added jitter in synthesis. Voiced ratio < 0.15 = insufficient data.
    """
    if voiced_ratio < 0.15:
        return 50.0

    low, high = 0.002, 0.030
    clamped = max(low, min(jitter, high))
    return ((clamped - low) / (high - low)) * 100.0


def score_spectral_flatness(flatness_mean: float) -> float:
    """
    Score based on spectral flatness.
    High flatness = noise-like / unnatural. Low flatness = tonal voiced speech.
    AI voices often show very uniform spectral flatness.
    Typical range: 0.001–0.12.
    """
    low, high = 0.001, 0.12
    clamped = max(low, min(flatness_mean, high))
    normalized = (clamped - low) / (high - low)
    return (1.0 - normalized) * 100.0


def score_energy_variation(rms_var: float) -> float:
    """
    Score based on RMS energy variation.
    AI speech often has overly flat or robotic energy curves.
    Human speech has natural energy bursts. Typical variance: 0.0005–0.025.
    Very low energy variance is a strong AI indicator.
    """
    low, high = 0.0003, 0.020
    clamped = max(low, min(rms_var, high))
    return ((clamped - low) / (high - low)) * 100.0


def score_zcr_variation(zcr_std: float) -> float:
    """
    Score based on zero-crossing rate standard deviation.
    Human speech fluctuates naturally in ZCR. AI speech is more uniform.
    """
    low, high = 0.01, 0.10
    clamped = max(low, min(zcr_std, high))
    return ((clamped - low) / (high - low)) * 100.0


def score_mfcc_delta2(delta2_var: float) -> float:
    """
    Score based on MFCC acceleration (delta-delta) variance.
    Measures how erratically the spectral change-of-change behaves.
    Human speech production involves complex muscular coordination → high acceleration variance.
    AI synthesis tends to produce unnaturally smooth temporal evolution → low delta2 variance.
    Human typical: 10–60. AI (high-quality TTS): 3–10.
    """
    low, high = 2.0, 30.0
    clamped = max(low, min(delta2_var, high))
    return ((clamped - low) / (high - low)) * 100.0


def score_spectral_flux(flux_std: float) -> float:
    """
    Score based on spectral flux standard deviation.
    Spectral flux measures frame-to-frame spectral change.
    High variance in flux = unpredictable spectral evolution = human-like.
    Low variance in flux = too-uniform spectral changes = AI-like.
    Human typical: 5–20. High-quality AI: 5–12 (moderate but smooth).
    """
    low, high = 3.0, 18.0
    clamped = max(low, min(flux_std, high))
    return ((clamped - low) / (high - low)) * 100.0


def _compute_extra_features(y: np.ndarray, sr: int, mfcc: np.ndarray) -> dict:
    """Compute additional features not in the base extractor."""
    S = np.abs(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    delta2 = librosa.feature.delta(mfcc, order=2)

    return {
        "spectral_flux_std": float(np.std(flux)),
        "mfcc_delta2_var": float(np.mean(np.var(delta2, axis=1))),
    }


def compute_trust_score(features: dict) -> dict:
    """
    Combine all feature scores into a final Trust Score (0–100).
    100 = very likely human. 0 = very likely AI / deepfake.

    Weights (must sum to 1.0):
      MFCC variance (non-monotonic)  30%  — catches both too-smooth and over-expressive AI
      Energy variation               20%  — flat energy envelope = strong AI signal
      MFCC delta2 variance           15%  — temporal acceleration; AI is too smooth
      Pitch jitter                   15%  — robotic steadiness vs. human irregularity
      Spectral flux variability      10%  — frame-to-frame change uniformity
      Spectral flatness               7%  — tonal vs. noise-like spectrum
      ZCR variation                   3%  — zero-crossing rate variance
    """
    y = features["y"]
    sr = features["sr"]
    mfcc = features["mfcc"]["mfcc"]

    extra = _compute_extra_features(y, sr, mfcc)

    mfcc_score = score_mfcc(features["mfcc"]["mfcc_var_mean"])
    pitch_score = score_pitch_jitter(
        features["pitch"]["pitch_jitter"],
        features["pitch"]["voiced_ratio"],
    )
    flatness_score = score_spectral_flatness(
        features["spectral"]["spectral_flatness_mean"]
    )
    energy_score = score_energy_variation(features["energy"]["rms_var"])
    zcr_score = score_zcr_variation(features["spectral"]["zcr_std"])
    delta2_score = score_mfcc_delta2(extra["mfcc_delta2_var"])
    flux_score = score_spectral_flux(extra["spectral_flux_std"])

    weights = {
        "mfcc": 0.30,
        "energy": 0.20,
        "delta2": 0.15,
        "pitch": 0.15,
        "flux": 0.10,
        "flatness": 0.07,
        "zcr": 0.03,
    }

    trust_score = (
        weights["mfcc"] * mfcc_score
        + weights["energy"] * energy_score
        + weights["delta2"] * delta2_score
        + weights["pitch"] * pitch_score
        + weights["flux"] * flux_score
        + weights["flatness"] * flatness_score
        + weights["zcr"] * zcr_score
    )

    trust_score = float(np.clip(trust_score, 0, 100))

    if trust_score >= 70:
        verdict = "LIKELY HUMAN"
        risk = "Low Risk"
        color = "green"
    elif trust_score >= 45:
        verdict = "UNCERTAIN"
        risk = "Medium Risk"
        color = "orange"
    else:
        verdict = "LIKELY AI / DEEPFAKE"
        risk = "High Risk"
        color = "red"

    return {
        "trust_score": trust_score,
        "verdict": verdict,
        "risk": risk,
        "color": color,
        "component_scores": {
            "MFCC Variance": round(mfcc_score, 1),
            "Energy Variation": round(energy_score, 1),
            "MFCC Acceleration": round(delta2_score, 1),
            "Pitch Jitter": round(pitch_score, 1),
            "Spectral Flux": round(flux_score, 1),
            "Spectral Flatness": round(flatness_score, 1),
            "ZCR Variation": round(zcr_score, 1),
        },
        "raw": {
            "mfcc_var_mean": round(features["mfcc"]["mfcc_var_mean"], 4),
            "pitch_jitter": round(features["pitch"]["pitch_jitter"], 5),
            "voiced_ratio": round(features["pitch"]["voiced_ratio"], 3),
            "pitch_mean": round(features["pitch"]["pitch_mean"], 2),
            "spectral_flatness_mean": round(
                features["spectral"]["spectral_flatness_mean"], 5
            ),
            "zcr_std": round(features["spectral"]["zcr_std"], 5),
            "rms_var": round(features["energy"]["rms_var"], 6),
            "mfcc_delta2_var": round(extra["mfcc_delta2_var"], 4),
            "spectral_flux_std": round(extra["spectral_flux_std"], 4),
            "duration": round(features["duration"], 2),
        },
    }


def analyze_audio_file(file_path: str) -> dict:
    """End-to-end analysis: extract features → compute trust score."""
    features = extract_all_features(file_path)
    result = compute_trust_score(features)
    result["features"] = features
    return result
