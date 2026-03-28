import os
import tempfile
import streamlit as st
import numpy as np

from core.deepfake_detector import analyze_audio_file
from core.noise_analyzer import detect_background_loop
from core.feature_extractor import load_audio
from models.model_loader import predict_with_model
from visuals.spectrogram_plot import (
    plot_mel_spectrogram,
    plot_mfcc,
    plot_pitch_contour,
    plot_rms_energy,
)
from visuals.score_gauge import render_trust_gauge, render_component_bars

st.set_page_config(
    page_title="The Digital Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 32px;
        border: 1px solid #1e3a4a;
    }
    .main-header h1 { color: #00d4ff; font-size: 2.8rem; margin: 0; letter-spacing: -1px; }
    .main-header p { color: #94a3b8; font-size: 1.1rem; margin-top: 8px; }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 4px 2px;
    }
    .badge-blue { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
    .badge-green { background: #14291f; color: #4ade80; border: 1px solid #16a34a; }

    .verdict-box {
        border-radius: 12px; padding: 20px 28px; margin: 16px 0;
        font-size: 1.4rem; font-weight: 700; text-align: center;
        letter-spacing: 1px;
    }
    .verdict-human { background: #052e16; border: 2px solid #22c55e; color: #4ade80; }
    .verdict-uncertain { background: #1c1007; border: 2px solid #f97316; color: #fb923c; }
    .verdict-fake { background: #1a0000; border: 2px solid #ef4444; color: #f87171; }

    .metric-card {
        background: #1e2130; border-radius: 10px; padding: 16px;
        border: 1px solid #2d3348; text-align: center;
    }
    .metric-card .val { font-size: 1.5rem; font-weight: 700; color: #00d4ff; }
    .metric-card .label { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }

    .noise-box {
        border-radius: 10px; padding: 16px 20px; margin-top: 12px;
        font-size: 0.95rem;
    }
    .noise-clean { background: #052e16; border: 1px solid #22c55e; color: #4ade80; }
    .noise-suspicious { background: #1a0000; border: 1px solid #ef4444; color: #f87171; }

    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin: 24px 0 12px;
        border-bottom: 1px solid #1e2130; padding-bottom: 8px;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #2d3348; border-radius: 12px; padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ The Digital Guardian</h1>
        <p>Real-time Acoustic Fingerprinting for Deepfake Voice Detection</p>
        <span class="badge badge-blue">AI Safety Tool</span>
        <span class="badge badge-blue">MFCC Analysis</span>
        <span class="badge badge-blue">Pitch Jitter</span>
        <span class="badge badge-green">ASVspoof 2019 Inspired</span>
    </div>
    """,
    unsafe_allow_html=True,
)

col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown('<div class="section-title">Upload Audio File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Drop a .wav or .mp3 file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Supports WAV, MP3, FLAC, OGG. Minimum 1 second recommended.",
    )

with col_info:
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    The Guardian analyzes 5 acoustic fingerprints:

    | Feature | What It Detects |
    |---|---|
    | **MFCC Variance** | AI speech is unnaturally smooth |
    | **Pitch Jitter** | AI pitch lacks natural micro-wobble |
    | **Spectral Flatness** | AI has overly clean frequency bands |
    | **Energy Variation** | AI energy envelope is too consistent |
    | **ZCR Variation** | AI zero-crossings lack natural rhythm |
    """)

if uploaded_file is not None:
    st.divider()

    with tempfile.NamedTemporaryFile(
        suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("🔍 Analyzing audio fingerprint..."):
        try:
            result = analyze_audio_file(tmp_path)
            features = result["features"]
            y, sr = features["y"], features["sr"]

            noise_result = detect_background_loop(y, sr)
            ml_result = predict_with_model(features)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()
        finally:
            os.unlink(tmp_path)

    trust_score = result["trust_score"]
    verdict = result["verdict"]
    risk = result["risk"]
    color = result["color"]

    if verdict == "LIKELY HUMAN":
        verdict_class = "verdict-human"
    elif verdict == "UNCERTAIN":
        verdict_class = "verdict-uncertain"
    else:
        verdict_class = "verdict-fake"

    st.markdown(
        f'<div class="verdict-box {verdict_class}">⚡ {verdict} &nbsp;|&nbsp; {risk}</div>',
        unsafe_allow_html=True,
    )

    col_gauge, col_bars = st.columns([1, 1], gap="large")

    with col_gauge:
        gauge_fig = render_trust_gauge(trust_score, verdict, risk)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_bars:
        bars_fig = render_component_bars(result["component_scores"])
        st.plotly_chart(bars_fig, use_container_width=True)

    st.markdown('<div class="section-title">Raw Acoustic Metrics</div>', unsafe_allow_html=True)
    raw = result["raw"]
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{raw['pitch_jitter']:.4f}</div>
            <div class="label">Pitch Jitter</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{raw['mfcc_var_mean']:.1f}</div>
            <div class="label">MFCC Variance</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{raw['spectral_flatness_mean']:.4f}</div>
            <div class="label">Spectral Flatness</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{raw['voiced_ratio']:.0%}</div>
            <div class="label">Voiced Ratio</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{raw['duration']}s</div>
            <div class="label">Duration</div>
        </div>""", unsafe_allow_html=True)

    noise_class = "noise-suspicious" if noise_result["loop_detected"] else "noise-clean"
    noise_icon = "🔴" if noise_result["loop_detected"] else "🟢"
    noise_label = "Looped/Synthetic Background Detected" if noise_result["loop_detected"] else "Natural Background — No Looping Detected"
    noise_reasons = " &nbsp;·&nbsp; ".join(noise_result["reasons"])

    st.markdown(f"""
    <div class="noise-box {noise_class}">
        {noise_icon} <strong>Background Noise Analysis:</strong> {noise_label}<br>
        <small style="opacity:0.75">{noise_reasons}</small>
    </div>
    """, unsafe_allow_html=True)

    if ml_result.get("ml_available"):
        st.info(
            f"🤖 **ML Model Score:** {ml_result['ml_trust_score']}% — {ml_result['ml_verdict']} "
            f"(ensemble with acoustic analysis)"
        )

    st.markdown('<div class="section-title">Visual Spectrograms</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎵 Mel Spectrogram", "📊 MFCC Heatmap", "📈 Pitch Contour", "⚡ Energy Envelope"]
    )

    with tab1:
        mel_img = plot_mel_spectrogram(y, sr, title="Mel Spectrogram — AI voices show unnaturally clean bands")
        st.image(mel_img, use_column_width=True)
        st.caption("**Reading the spectrogram:** Human voices show irregular, noisy frequency patterns. AI voices have unusually clean, evenly-spaced horizontal lines.")

    with tab2:
        mfcc_img = plot_mfcc(features["mfcc"]["mfcc"], sr, title="MFCC Heatmap — Smooth stripes = AI, noisy patches = Human")
        st.image(mfcc_img, use_column_width=True)
        st.caption("**Reading the MFCC:** High variance (noisy pattern) = human. Uniform horizontal stripes = AI-generated.")

    with tab3:
        pitch_img = plot_pitch_contour(features["pitch"]["f0"], sr)
        st.image(pitch_img, use_column_width=True)
        st.caption("**Reading the pitch contour:** Human pitch undulates naturally. AI pitch appears as a flat or overly smooth line.")

    with tab4:
        energy_img = plot_rms_energy(features["energy"]["rms"], sr)
        st.image(energy_img, use_column_width=True)
        st.caption("**Reading the energy envelope:** Human speech has natural energy bursts (breathing, emphasis). AI speech may appear too uniform.")

    st.audio(uploaded_file)

    with st.expander("🔬 Full Technical Details"):
        st.json({
            "trust_score": trust_score,
            "verdict": verdict,
            "risk": risk,
            "component_scores": result["component_scores"],
            "raw_features": result["raw"],
            "background_noise_analysis": {
                k: v for k, v in noise_result.items() if k != "reasons"
            },
            "noise_reasons": noise_result["reasons"],
        })

else:
    st.info("⬆️  Upload a .wav or .mp3 audio file above to begin analysis.")
    st.markdown("""
    ---
    ### What this tool detects:
    - **Voice cloning attacks** (ElevenLabs, RVC, SV2TTS)
    - **TTS-generated speech** (text-to-speech synthesis)
    - **Voice conversion** (voice morphing/deepfakes)

    ### What makes AI voices detectable:
    - **Spectral Gaps** — synthesized voices miss the messy, random noise of real vocal cords
    - **Robotic Fluency** — AI pitch is mathematically precise; human pitch naturally wobbles (jitter)
    - **Perfect Energy** — real voices have breathing patterns and emphasis variation
    - **Looped Backgrounds** — AI recordings often have digitally looped or silent backgrounds

    > **Note:** This tool is most effective on clean voice recordings of 3+ seconds.
    > It is designed for educational and hackathon use. For production anti-fraud use,
    > combine with a model trained on the full ASVspoof 2019 dataset.
    """)

st.markdown(
    """
    <div style="text-align:center; color:#4b5563; font-size:0.8rem; margin-top:40px; padding-top:20px; border-top:1px solid #1e2130">
        🛡️ The Digital Guardian · AI for Impact Hackathon · PS1: Digital Safety & Financial Inclusion<br>
        Built with Streamlit · librosa · plotly · ASVspoof 2019 Methodology
    </div>
    """,
    unsafe_allow_html=True,
)
