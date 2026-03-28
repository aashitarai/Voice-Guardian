import numpy as np
import librosa


def load_audio(file_path: str, sr: int = 22050):
    """Load an audio file and return the signal and sample rate."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr


def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 40) -> dict:
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
    AI voices tend to have unnaturally smooth, low-variance MFCCs.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    return {
        "mfcc": mfcc,
        "mfcc_delta": mfcc_delta,
        "mfcc_delta2": mfcc_delta2,
        "mfcc_mean": np.mean(mfcc, axis=1),
        "mfcc_std": np.std(mfcc, axis=1),
        "mfcc_var_mean": float(np.mean(np.var(mfcc, axis=1))),
    }


def extract_pitch_features(y: np.ndarray, sr: int) -> dict:
    """
    Extract pitch (F0) and jitter features.
    Human pitch is naturally irregular; AI speech has robotic steadiness.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )

    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) < 2:
        return {
            "f0": f0,
            "voiced_flag": voiced_flag,
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_jitter": 0.0,
            "voiced_ratio": 0.0,
        }

    pitch_diffs = np.abs(np.diff(voiced_f0))
    jitter = float(np.mean(pitch_diffs) / (np.mean(voiced_f0) + 1e-6))

    return {
        "f0": f0,
        "voiced_flag": voiced_flag,
        "pitch_mean": float(np.mean(voiced_f0)),
        "pitch_std": float(np.std(voiced_f0)),
        "pitch_jitter": jitter,
        "voiced_ratio": float(np.sum(voiced_flag) / len(voiced_flag)),
    }


def extract_spectral_features(y: np.ndarray, sr: int) -> dict:
    """
    Extract spectral features.
    AI voices often show unnaturally flat/clean spectral characteristics.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "spectral_flatness_std": float(np.std(spectral_flatness)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }


def extract_rms_energy(y: np.ndarray) -> dict:
    """Extract RMS energy contour (AI voices may have too-perfect energy envelopes)."""
    rms = librosa.feature.rms(y=y)[0]
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_var": float(np.var(rms)),
        "rms": rms,
    }


def extract_all_features(file_path: str) -> dict:
    """Master function: loads audio and extracts all features."""
    y, sr = load_audio(file_path)

    mfcc_feats = extract_mfcc(y, sr)
    pitch_feats = extract_pitch_features(y, sr)
    spectral_feats = extract_spectral_features(y, sr)
    energy_feats = extract_rms_energy(y)

    return {
        "y": y,
        "sr": sr,
        "duration": float(len(y) / sr),
        "mfcc": mfcc_feats,
        "pitch": pitch_feats,
        "spectral": spectral_feats,
        "energy": energy_feats,
    }
