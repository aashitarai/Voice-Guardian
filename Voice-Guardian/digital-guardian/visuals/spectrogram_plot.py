import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
import io

matplotlib.use("Agg")


def plot_mel_spectrogram(y: np.ndarray, sr: int, title: str = "Mel Spectrogram") -> bytes:
    """
    Generate a mel spectrogram image and return as PNG bytes.
    AI voices often show very clean, evenly-spaced horizontal bands.
    Human voices show irregular, noisy frequency distributions.
    """
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=8000,
        ax=ax,
        cmap="magma",
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_mfcc(mfcc: np.ndarray, sr: int, title: str = "MFCC Features") -> bytes:
    """
    Generate MFCC heatmap.
    Smooth, horizontal stripes → AI. Noisy, irregular patterns → Human.
    """
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        ax=ax,
        cmap="coolwarm",
        sr=sr,
    )
    fig.colorbar(img, ax=ax)
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_pitch_contour(f0: np.ndarray, sr: int, hop_length: int = 512) -> bytes:
    """
    Plot pitch (F0) contour over time.
    AI voices: flat, straight lines.
    Human voices: natural undulating curve with micro-variations.
    """
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    valid = ~np.isnan(f0)
    ax.scatter(
        times[valid], f0[valid],
        s=4, color="#00d4ff", alpha=0.8, label="Voiced pitch"
    )
    ax.plot(
        times[valid], f0[valid],
        color="#00d4ff", alpha=0.4, linewidth=0.8
    )

    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Frequency (Hz)", color="white")
    ax.set_title("Pitch (F0) Contour — Smooth = AI, Irregular = Human",
                 color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_rms_energy(rms: np.ndarray, sr: int, hop_length: int = 512) -> bytes:
    """
    Plot RMS energy contour.
    Flat energy = AI (too consistent), variable energy = human (natural breathing, emphasis).
    """
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    ax.fill_between(times, rms, alpha=0.5, color="#ff7f50")
    ax.plot(times, rms, color="#ff7f50", linewidth=1.2)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("RMS Energy", color="white")
    ax.set_title("Energy Envelope — Flat = AI, Variable = Human",
                 color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
