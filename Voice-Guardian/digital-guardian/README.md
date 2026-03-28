# 🛡️ The Digital Guardian
## Deepfake Voice Authenticity Detector

**AI for Impact Hackathon — PS1: Digital Safety & Financial Inclusion**

---

## Overview

The Digital Guardian is a real-time acoustic fingerprinting tool that detects AI-generated (deepfake) voices. It analyzes audio files using signal processing techniques inspired by the **ASVspoof 2019** anti-spoofing challenge.

---

## How to Run (PyCharm)

### 1. Set up a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Project Structure

```
digital-guardian/
│
├── app.py                        ← Main Streamlit UI
│
├── core/
│   ├── feature_extractor.py      ← MFCC, pitch, spectral feature extraction (librosa)
│   ├── deepfake_detector.py      ← Trust Score computation from features
│   └── noise_analyzer.py         ← Background loop/synthetic noise detection
│
├── models/
│   └── model_loader.py           ← Optional: plug in a pre-trained ASVspoof model
│
├── visuals/
│   ├── spectrogram_plot.py       ← Mel spectrogram, MFCC, pitch & energy plots
│   └── score_gauge.py            ← Plotly gauge + feature bar chart
│
├── data/sample_audio/            ← Place test .wav files here
│
├── requirements.txt
└── README.md
```

---

## Features

### MVP (Must-Haves)
- [x] Upload `.wav`, `.mp3`, `.flac` audio files
- [x] MFCC feature extraction via librosa
- [x] Pitch variance & jitter analysis
- [x] Trust Score (0–100%) with color-coded verdict

### Good-to-Haves (Winning Edge)
- [x] Mel spectrogram visualization
- [x] MFCC heatmap — smooth stripes = AI, noisy = human
- [x] Pitch contour plot — flat = AI, undulating = human
- [x] Energy envelope plot
- [x] Background noise loop detection (synthetic ambient audio detection)
- [x] Optional ML model plug-in (ASVspoof 2019 model)

---

## How the Trust Score Works

| Feature | Weight | AI Signature |
|---|---|---|
| MFCC Variance | 30% | Too smooth, unnaturally low variance |
| Pitch Jitter | 30% | Too steady, lacks human micro-wobble |
| Spectral Flatness | 20% | Overly flat/clean frequency distribution |
| Energy Variation | 10% | Too uniform energy envelope |
| ZCR Variation | 10% | Robotic zero-crossing pattern |

**Score interpretation:**
- 🟢 **70–100%** = Likely Human
- 🟠 **45–69%** = Uncertain (borderline)
- 🔴 **0–44%** = Likely AI / Deepfake

---

## Adding a Pre-trained Model (ASVspoof 2019)

1. Train a classifier on the [ASVspoof 2019 dataset](https://www.asvspoof.org/)
2. Export the sklearn model: `pickle.dump(model, open("models/asv_classifier.pkl", "wb"))`
3. The app will automatically detect and use it, blending ML + acoustic scores

Feature vector format is defined in `models/model_loader.py`.

---

## Dataset Reference

**ASVspoof 2019** — Logical Access (LA) subset recommended
- Labels: `bonafide` (real) vs `spoof` (AI-generated)
- Attacks include: neural TTS, voice conversion, waveform concat
- Download: https://www.asvspoof.org/database

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | Web UI |
| librosa | Audio feature extraction |
| numpy / scipy | Signal math |
| matplotlib | Spectrogram rendering |
| plotly | Interactive gauge & charts |
| scikit-learn | Optional ML classifier |

---

*Built for AI for Impact Hackathon · Problem Statement 1 · Digital Safety & Financial Inclusion*
