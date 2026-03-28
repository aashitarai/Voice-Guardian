import numpy as np
import librosa


def detect_background_loop(y: np.ndarray, sr: int, segment_duration: float = 0.5) -> dict:
    """
    Detect if the background noise has a looped/synthetic pattern.
    AI recordings often have digitally looped ambient sounds or near-silent backgrounds.

    Strategy:
    - Split audio into small segments
    - Compute RMS energy of each segment
    - Detect near-silence or abnormally repetitive patterns
    """
    segment_samples = int(segment_duration * sr)
    segments = [
        y[i: i + segment_samples]
        for i in range(0, len(y) - segment_samples, segment_samples)
    ]

    if len(segments) < 3:
        return {
            "loop_detected": False,
            "confidence": 0.0,
            "notes": "Audio too short to analyze background.",
        }

    rms_values = np.array([np.sqrt(np.mean(s ** 2)) for s in segments])
    rms_cv = float(np.std(rms_values) / (np.mean(rms_values) + 1e-8))

    silence_threshold = 0.002
    silence_ratio = float(np.sum(rms_values < silence_threshold) / len(rms_values))

    mfcc_segments = [
        librosa.feature.mfcc(y=s, sr=sr, n_mfcc=13).mean(axis=1)
        for s in segments[:min(len(segments), 20)]
    ]

    correlations = []
    for i in range(len(mfcc_segments) - 1):
        corr = float(
            np.corrcoef(mfcc_segments[i], mfcc_segments[i + 1])[0, 1]
        )
        correlations.append(corr)

    avg_correlation = float(np.mean(correlations)) if correlations else 0.0

    loop_score = 0.0
    loop_reasons = []

    if rms_cv < 0.15:
        loop_score += 40
        loop_reasons.append("Abnormally uniform energy levels across segments")

    if silence_ratio > 0.6:
        loop_score += 30
        loop_reasons.append(f"High silence ratio ({silence_ratio:.0%}) — digitally silent background")

    if avg_correlation > 0.92:
        loop_score += 30
        loop_reasons.append(f"Very high segment similarity ({avg_correlation:.2f}) — possible looped audio")

    loop_detected = loop_score >= 50

    return {
        "loop_detected": loop_detected,
        "loop_score": round(loop_score, 1),
        "confidence": round(loop_score / 100.0, 2),
        "rms_cv": round(rms_cv, 4),
        "silence_ratio": round(silence_ratio, 3),
        "avg_segment_correlation": round(avg_correlation, 4),
        "reasons": loop_reasons if loop_reasons else ["No suspicious background patterns detected"],
        "notes": (
            "Looped or synthetic background noise detected."
            if loop_detected
            else "Background appears naturally random — consistent with real environments."
        ),
    }
