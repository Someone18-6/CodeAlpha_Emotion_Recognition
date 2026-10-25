"""
Speech Emotion AI
=================
A warm, editorial Streamlit interface for the XGBoost speech-emotion model
trained on the RAVDESS dataset.

What stays exactly as before
  * the trained model + label encoder loaded from MODEL_DIR
  * audio loaded at 22,050 Hz, mono
  * the 360-feature extraction (MFCC / chroma / mel, mean + std)
  * the eight emotion classes and the encoder's label mapping

Requires Streamlit 1.40+ (keyed containers). Plotly is used for the charts
when installed; the app degrades gracefully to plain HTML/SVG if it is not.
"""

import hashlib
import html
import inspect
import math
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

import joblib
import librosa
import numpy as np
import soundfile as sf
import streamlit as st

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # the app still works without Plotly
    go = None
    PLOTLY_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = "/content/drive/MyDrive/Emotion_Recognition"
MODEL_DIR = os.path.join(BASE_DIR, "model")

TARGET_SR = 22050  # must match training
SUPPORTED_TYPES = ["wav", "mp3", "m4a", "ogg", "flac"]
AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

# Your own reported figure. It is shown with a caveat, never as a guarantee.
REPORTED_TEST_ACCURACY = "68.06%"
MAX_HISTORY = 12

# ---- Palette (warm only: no blues, purples or cyans) ------------------------
INK = "#292724"
MUTED = "#6F675E"
GRID = "#EFE7DB"
ACCENT = "#C76543"
BAR_MUTED = "#DDCFBB"
FONT_STACK = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Each emotion gets a muted warm colour, a short description, and a small
# "voice-print" pattern (relative bar heights) used as its visual signature.
EMOTION_STYLE = {
    "angry": {
        "color": "#B5472F",
        "desc": "A sharp, forceful delivery with high vocal intensity.",
        "pattern": [0.5, 0.9, 0.4, 1.0, 0.35, 0.95, 0.45, 0.85, 0.5],
    },
    "calm": {
        "color": "#7F9470",
        "desc": "A soft, steady delivery with an unhurried, even tone.",
        "pattern": [0.35, 0.5, 0.65, 0.75, 0.8, 0.75, 0.65, 0.5, 0.35],
    },
    "disgust": {
        "color": "#8C8A3F",
        "desc": "A clipped, strained delivery with a tone of aversion.",
        "pattern": [0.7, 0.4, 0.6, 0.3, 0.5, 0.3, 0.6, 0.4, 0.7],
    },
    "fearful": {
        "color": "#8A6247",
        "desc": "A tense, unsteady delivery with a tremulous quality.",
        "pattern": [0.3, 0.6, 0.35, 0.75, 0.4, 0.8, 0.45, 0.65, 0.3],
    },
    "happy": {
        "color": "#D9A02F",
        "desc": "A bright, lively delivery with an upbeat, animated tone.",
        "pattern": [0.4, 0.7, 0.55, 0.9, 0.7, 1.0, 0.75, 0.55, 0.4],
    },
    "neutral": {
        "color": "#968B7E",
        "desc": "An even, unexpressive delivery without strong colouring.",
        "pattern": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    },
    "sad": {
        "color": "#A67B7B",
        "desc": "A low, subdued delivery with a heavier, slower feel.",
        "pattern": [0.85, 0.75, 0.62, 0.5, 0.4, 0.3, 0.24, 0.2, 0.15],
    },
    "surprised": {
        "color": "#E3874F",
        "desc": "A sudden, heightened delivery with abrupt shifts in pitch.",
        "pattern": [0.25, 0.3, 0.38, 0.55, 1.0, 0.55, 0.38, 0.3, 0.25],
    },
}
DEFAULT_STYLE = {
    "color": "#968B7E",
    "desc": "The model's closest match for this recording.",
    "pattern": [0.4, 0.55, 0.7, 0.55, 0.4, 0.55, 0.7, 0.55, 0.4],
}
LEGEND_ORDER = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

LOADING_STEPS = [
    ("Preparing your recording", "Staging a private temporary copy of the file"),
    ("Extracting acoustic features", "Decoding at 22.05 kHz mono and computing 360 features"),
    ("Estimating emotion", "Scoring the features with the XGBoost model"),
]


def style_for(emotion):
    return EMOTION_STYLE.get(str(emotion).lower(), DEFAULT_STYLE)


# =============================================================================
# Page setup
# =============================================================================
def make_favicon():
    """A small terracotta tile with voice bars. Falls back to an emoji."""
    try:
        from PIL import Image, ImageDraw

        size = 128
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=32, fill=ACCENT)
        bar_w, gap = 11, 9
        x = (size - (5 * bar_w + 4 * gap)) // 2
        for h in (30, 56, 80, 56, 30):
            draw.rounded_rectangle(
                (x, size // 2 - h // 2, x + bar_w, size // 2 + h // 2),
                radius=5,
                fill="#FAF7F2",
            )
            x += bar_w + gap
        return img
    except Exception:
        return "🎙️"


st.set_page_config(
    page_title="Speech Emotion AI",
    page_icon=make_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Newer Streamlit versions replace use_container_width with width="stretch".
PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
BUTTON_HAS_WIDTH = "width" in inspect.signature(st.button).parameters


def stretch_button(label, **kwargs):
    """A full-width st.button on both newer and older Streamlit versions."""
    if BUTTON_HAS_WIDTH:
        return st.button(label, width="stretch", **kwargs)
    return st.button(label, use_container_width=True, **kwargs)


# =============================================================================
# Styles
# =============================================================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:ital,wght@0,400;0,500;1,400&display=swap');

:root{
  --ivory:#FAF7F2; --ivory-2:#F5F0E8; --card:#FFFDFA;
  --sand:#E9DFD2; --sand-2:#D9CCBA; --line:#E7DDCF;
  --ink:#292724; --ink-2:#4B463F; --muted:#6F675E;
  --accent:#C76543; --accent-btn:#B8553A; --accent-btn-hover:#A44B32; --accent-ink:#A34A30;
  --shadow-sm:0 1px 2px rgba(74,52,30,.05),0 6px 16px -8px rgba(74,52,30,.12);
  --shadow-md:0 2px 4px rgba(74,52,30,.04),0 20px 44px -20px rgba(74,52,30,.24);
  --serif:'Newsreader','Iowan Old Style','Palatino Linotype',Georgia,serif;
  --sans:'Inter',-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
}
html{color-scheme:light}
[data-testid="stMarkdownContainer"],[data-testid="stFileUploader"],[data-testid="stFileUploaderDropzone"] *:not([data-testid="stIconMaterial"]),
[data-testid="stButton"] button,[data-testid="stAudio"]{font-family:var(--sans) !important}
::selection{background:#EBC9B8;color:var(--ink)}

/* ---------- App shell ---------- */
.stApp{
  background:radial-gradient(1100px 520px at 88% -8%,#F3E6D6 0%,rgba(250,247,242,0) 62%),var(--ivory);
  color:var(--ink);
  font-family:var(--sans) !important;
}
[data-testid="stHeader"]{background:transparent}
[data-testid="stDecoration"], [data-testid="stAppDeployButton"], .stAppDeployButton, footer, #MainMenu{display:none !important}
.block-container{max-width:1080px;padding:3rem 1.5rem 3rem !important}

/* ---------- Topbar ---------- */
.sea-topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.25rem 0 1.5rem}
.sea-brand{display:flex;align-items:center;gap:.7rem;font-family:var(--serif);font-size:1.35rem;font-weight:500;letter-spacing:-.01em;color:var(--ink)}
.sea-brand svg{width:32px;height:32px;flex:none}
.sea-chip{display:inline-flex;align-items:center;gap:.5rem;padding:.4rem .85rem;border:1px solid var(--line);border-radius:999px;background:var(--card);font-size:.78rem;color:var(--muted);white-space:nowrap}
.sea-chip i{width:6px;height:6px;border-radius:50%;background:var(--accent)}

/* ---------- Hero ---------- */
.sea-hero{display:grid;grid-template-columns:minmax(0,1.12fr) minmax(0,.88fr);gap:3.5rem;align-items:center;padding:1rem 0 2.5rem}
.sea-hero--compact{grid-template-columns:minmax(0,1fr);padding:.25rem 0 1.5rem}
.sea-hero--compact .sea-hero-art,.sea-hero--compact .sea-lede,.sea-hero--compact .sea-facts{display:none}
.sea-eyebrow{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent-ink);font-weight:600;margin-bottom:1rem}
.sea-title{font-family:var(--serif);font-weight:400;font-size:clamp(2.7rem,6.2vw,4.5rem);line-height:1.03;letter-spacing:-.028em;color:var(--ink);margin:0 0 1.35rem}
.sea-title em{font-style:italic;color:var(--accent-ink)}
.sea-hero--compact .sea-title{font-size:clamp(1.9rem,4vw,2.5rem);margin:0}
.sea-lede{font-size:1.08rem;line-height:1.65;color:var(--ink-2);max-width:52ch}
.sea-facts{display:flex;flex-wrap:wrap;gap:1.25rem 2.5rem;margin-top:2rem;padding-top:1.5rem;border-top:1px solid var(--line)}
.sea-fact strong{display:block;font-family:var(--serif);font-size:2rem;font-weight:500;line-height:1;color:var(--ink)}
.sea-fact span{display:block;margin-top:.35rem;font-size:.82rem;color:var(--muted)}
.sea-art{border-radius:32px;background:linear-gradient(180deg,#F4EBDE 0%,#EDE1D0 100%);border:1px solid var(--line);padding:1.1rem 1.1rem .9rem;box-shadow:var(--shadow-md)}
.sea-art svg{display:block;width:100%;height:auto}
.sea-legend{display:flex;flex-wrap:wrap;justify-content:center;gap:.35rem .95rem;padding:.35rem .5rem 0;font-size:.74rem;color:var(--ink-2)}
.sea-legend span{display:inline-flex;align-items:center;gap:.4rem}
.sea-legend i{width:8px;height:8px;border-radius:50%;background:var(--c)}
.sea-bar{transform-box:fill-box;transform-origin:center;animation:sea-pulse 2s ease-in-out infinite}
.sea-halo{transform-box:fill-box;transform-origin:center;animation:sea-halo 3.6s ease-in-out infinite}

/* ---------- Sections ---------- */
.sea-section{margin:2.75rem 0 1.25rem}
.sea-h2{font-family:var(--serif);font-size:clamp(1.8rem,3.6vw,2.3rem);font-weight:400;letter-spacing:-.02em;line-height:1.15;margin:0 0 .45rem;color:var(--ink)}
.sea-sub{color:var(--muted);font-size:.98rem;line-height:1.55;max-width:62ch}

/* ---------- Panels (keyed Streamlit containers) ---------- */
[class*="st-key-panel_"]{
  background:var(--card);border:1px solid var(--line);border-radius:26px;
  padding:1.75rem !important;box-shadow:var(--shadow-sm);
  transition:box-shadow .3s ease,border-color .3s ease;animation:sea-rise .5s ease both;
}
[class*="st-key-panel_"]:hover{box-shadow:var(--shadow-md);border-color:var(--sand-2)}
.sea-panel-head{display:flex;gap:1rem;align-items:flex-start;margin-bottom:.6rem}
.sea-panel-icon{width:46px;height:46px;border-radius:14px;background:#F6E7DE;display:grid;place-items:center;flex:none}
.sea-panel-icon svg{width:24px;height:24px}
.sea-panel-title{font-family:var(--serif);font-size:1.5rem;font-weight:500;letter-spacing:-.015em;line-height:1.2;color:var(--ink)}
.sea-panel-sub{margin-top:.3rem;font-size:.92rem;line-height:1.5;color:var(--muted)}

/* ---------- Uploader ---------- */
[data-testid="stFileUploaderDropzone"]{
  background:var(--ivory-2);border:1.5px dashed #CDBBA3;border-radius:20px;
  padding:2.1rem 1.5rem;transition:border-color .2s ease,background .2s ease,transform .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:focus-within{border-color:var(--accent);background:#FBF0E8}
[data-testid="stFileUploaderDropzone"] svg{color:var(--accent)}
[data-testid="stFileUploaderDropzoneInstructions"] span{color:var(--ink);font-weight:500}
[data-testid="stFileUploaderDropzoneInstructions"] small{color:var(--muted)}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]{
  border-radius:999px;border:1px solid var(--accent-btn);background:var(--card);
  color:var(--accent-ink);font-weight:500;padding:.5rem 1.4rem;transition:all .2s ease;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover{background:var(--accent-btn);color:#fff;border-color:var(--accent-btn)}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-minimal"],
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-borderlessIcon"]{
  background:transparent;border:0;border-radius:10px;color:var(--muted);transition:background .2s ease,color .2s ease;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-minimal"]:hover,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-borderlessIcon"]:hover{background:#F6E7DE;color:var(--accent-ink)}
[data-testid="stFileChip"]{background:var(--card);border:1px solid var(--line);border-radius:14px}
[data-testid="stFileChip"] > div:first-child{background:#F6E7DE;color:var(--accent-ink);border-radius:10px}

/* ---------- Buttons ---------- */
[data-testid="stButton"],[data-testid="stButton"] button,.st-key-analyze_btn,.st-key-clear_btn,.st-key-clear_history{width:100% !important}
[data-testid="stButton"] button{height:3.3rem;border-radius:16px;font-weight:500;letter-spacing:.01em;transition:transform .2s ease,box-shadow .2s ease,background .2s ease,border-color .2s ease}
[data-testid="stButton"] button p{color:inherit !important;margin:0;font-size:1.02rem;font-weight:500}
[data-testid="stBaseButton-primary"]{
  background:var(--accent-btn);border:1px solid var(--accent-btn);color:#fff;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 8px 18px -8px rgba(184,85,58,.65);
}
[data-testid="stBaseButton-primary"]:hover:not(:disabled){background:var(--accent-btn-hover);border-color:var(--accent-btn-hover);color:#fff;transform:translateY(-1px);box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 12px 22px -8px rgba(164,75,50,.7)}
[data-testid="stBaseButton-primary"]:active:not(:disabled){transform:translateY(0)}
[data-testid="stBaseButton-primary"]:disabled{background:var(--sand);border-color:var(--sand);color:#7C7266;box-shadow:none;cursor:not-allowed}
[data-testid="stButton"] [data-testid="stBaseButton-secondary"]{background:var(--card);border:1px solid var(--sand-2);color:var(--ink)}
[data-testid="stButton"] [data-testid="stBaseButton-secondary"]:hover:not(:disabled){border-color:var(--accent);color:var(--accent-ink);background:#FFF8F3}
[data-testid="stButton"] [data-testid="stBaseButton-secondary"]:disabled{background:transparent;color:#8B8175;border-color:var(--line)}
button:focus-visible,[data-testid="stFileUploaderDropzone"]:focus-within{outline:2px solid var(--accent);outline-offset:2px}

/* ---------- Audio player ---------- */
audio{width:100%;height:48px;border-radius:999px;filter:sepia(.28) saturate(.7) contrast(.97)}

/* ---------- File card ---------- */
.sea-file{display:flex;align-items:center;gap:1rem;padding:1rem 1.15rem;border:1px solid var(--line);border-radius:18px;background:var(--ivory-2);animation:sea-rise .4s ease both}
.sea-file-icon{width:46px;height:46px;border-radius:13px;background:var(--card);border:1px solid var(--line);display:grid;place-items:center;flex:none}
.sea-file-icon svg{width:22px;height:22px}
.sea-file-main{min-width:0;flex:1}
.sea-file-name{font-weight:600;font-size:.98rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--ink)}
.sea-file-meta{display:flex;flex-wrap:wrap;gap:.2rem 1.1rem;margin-top:.25rem;font-size:.84rem;color:var(--muted)}
.sea-badge{flex:none;padding:.3rem .75rem;border-radius:999px;background:#EEF0E4;color:#4A5A3C;font-size:.76rem;font-weight:600}
.sea-wavepreview{padding:.35rem .25rem 0}
.sea-wavepreview svg{display:block;width:100%;height:64px}
.sea-tips{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;font-size:.84rem;color:var(--muted)}
.sea-tips b{font-weight:500;margin-right:.25rem}
.sea-tips span.t{padding:.3rem .75rem;border:1px solid var(--line);border-radius:999px;background:var(--ivory-2);color:var(--ink-2)}

/* ---------- Notices ---------- */
.sea-notice{display:flex;gap:.8rem;padding:1rem 1.2rem;border-radius:16px;border:1px solid #EBC7BA;background:#FBEDE8;color:#7A2E1B;font-size:.93rem;line-height:1.55}
.sea-notice strong{display:block;font-weight:600;margin-bottom:.15rem}

/* ---------- Loading ---------- */
.sea-loading{padding:1.25rem 1.35rem;border-radius:18px;border:1px solid var(--line);background:var(--ivory-2);animation:sea-rise .3s ease both}
.sea-loading-row{display:flex;align-items:center;gap:1rem}
.sea-loading-wave{display:flex;align-items:center;gap:4px;height:32px;flex:none}
.sea-loading-wave i{display:block;width:4px;height:100%;border-radius:2px;background:var(--accent);transform-origin:center;animation:sea-pulse 1s ease-in-out infinite}
.sea-loading-wave i:nth-child(2){animation-delay:.12s}.sea-loading-wave i:nth-child(3){animation-delay:.24s}
.sea-loading-wave i:nth-child(4){animation-delay:.36s}.sea-loading-wave i:nth-child(5){animation-delay:.48s}
.sea-loading-title{font-weight:600;color:var(--ink);font-size:.98rem}
.sea-loading-sub{font-size:.85rem;color:var(--muted);margin-top:.15rem}
.sea-track{height:5px;border-radius:99px;background:var(--sand);margin-top:1rem;overflow:hidden}
.sea-track i{display:block;height:100%;border-radius:inherit;background:var(--accent)}

/* ---------- Result card ---------- */
.sea-result{--emo:#968B7E;--tint:rgba(150,139,126,.14);position:relative;overflow:hidden;border-radius:28px;padding:2rem 2rem 1.75rem;background:var(--card);border:1px solid var(--line);box-shadow:var(--shadow-md);animation:sea-rise .55s ease both}
.sea-result::before{content:"";position:absolute;inset:0 0 auto 0;height:6px;background:var(--emo)}
.sea-result-top{display:flex;align-items:center;justify-content:space-between;gap:1rem}
.sea-result .sea-eyebrow{margin:0;color:var(--muted)}
.sea-top-badge{padding:.28rem .7rem;border-radius:999px;background:var(--tint);color:var(--ink);font-size:.74rem;font-weight:600}
.sea-result-glyph{width:88px;height:88px;border-radius:26px;background:var(--tint);display:grid;place-items:center;margin:1.4rem 0 1.25rem}
.sea-result-glyph svg{width:52px;height:52px}
.sea-result-name{font-family:var(--serif);font-size:clamp(2.7rem,5vw,3.7rem);font-weight:400;line-height:1;letter-spacing:-.028em;color:var(--ink);margin-bottom:.7rem}
.sea-result-desc{color:var(--ink-2);line-height:1.55;font-size:.98rem;margin-bottom:1.6rem}
.sea-meter{height:8px;border-radius:99px;background:var(--sand);overflow:hidden}
.sea-meter i{display:block;height:100%;border-radius:inherit;background:var(--emo);transform-origin:left;animation:sea-grow .9s .15s cubic-bezier(.2,.8,.2,1) both}
.sea-result-stats{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1.2rem}
.sea-result-stats span{display:block;font-size:.78rem;color:var(--muted);margin-bottom:.2rem}
.sea-result-stats strong{font-family:var(--serif);font-size:1.5rem;font-weight:500;color:var(--ink)}
.sea-hint{margin-top:1.25rem;padding:.8rem 1rem;border-radius:14px;background:var(--ivory-2);border:1px solid var(--line);font-size:.86rem;line-height:1.5;color:var(--ink-2)}

/* ---------- Details / score list ---------- */
.sea-details{margin-top:1rem;padding-top:.9rem;border-top:1px solid var(--line);font-size:.9rem}
.sea-details summary{cursor:pointer;color:var(--accent-ink);font-weight:500;list-style:none;border-radius:6px}
.sea-details summary::-webkit-details-marker{display:none}
.sea-details summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.sea-row{display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid var(--line);color:var(--ink-2);font-variant-numeric:tabular-nums}

/* ---------- Insights ---------- */
.sea-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem 1.5rem;margin-top:.5rem;padding-top:1.25rem;border-top:1px solid var(--line)}
.sea-stat-label{font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}
.sea-stat-value{font-family:var(--serif);font-size:1.6rem;font-weight:500;color:var(--ink);margin-top:.2rem;line-height:1.15}
.sea-stat-note{font-size:.8rem;color:var(--muted);margin-top:.15rem;line-height:1.4}

/* ---------- Note ---------- */
.sea-note{display:flex;gap:.9rem;margin-top:1.5rem;padding:1.2rem 1.4rem;border-radius:18px;background:var(--ivory-2);border:1px solid var(--line);color:var(--ink-2);font-size:.9rem;line-height:1.65}
.sea-note svg{flex:none;width:20px;height:20px;margin-top:.15rem}
.sea-note strong{color:var(--ink);font-weight:600}

/* ---------- History ---------- */
.sea-history{border:1px solid var(--line);border-radius:22px;background:var(--card);overflow:hidden;box-shadow:var(--shadow-sm)}
.sea-hrow{display:grid;grid-template-columns:40px minmax(0,1fr) auto auto auto;gap:1rem;align-items:center;padding:.9rem 1.25rem;border-bottom:1px solid var(--line);font-size:.92rem;transition:background .2s ease}
.sea-hrow:last-child{border-bottom:0}
.sea-hrow:hover{background:var(--ivory-2)}
.sea-hrow.is-current{background:#FBF3EC}
.sea-hname{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:500;color:var(--ink)}
.sea-hemo{padding:.22rem .7rem;border-radius:999px;background:var(--tint);font-size:.8rem;font-weight:600;color:var(--ink)}
.sea-hconf,.sea-hdur{color:var(--muted);font-variant-numeric:tabular-nums;font-size:.85rem}
.sea-hglyph{width:36px;height:36px;border-radius:10px;background:var(--tint);display:grid;place-items:center}
.sea-hglyph svg{width:22px;height:22px}

/* ---------- Footer ---------- */
.sea-footer{display:flex;flex-wrap:wrap;justify-content:space-between;gap:.5rem 1.5rem;margin-top:3.5rem;padding-top:1.5rem;border-top:1px solid var(--line);font-size:.84rem;color:var(--muted)}
.sea-footer strong{color:var(--ink-2);font-weight:500}

/* ---------- Motion ---------- */
@keyframes sea-rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes sea-grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes sea-pulse{0%,100%{transform:scaleY(.45)}50%{transform:scaleY(1)}}
@keyframes sea-halo{0%,100%{opacity:.4;transform:scale(1)}50%{opacity:.1;transform:scale(1.07)}}
@media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}

/* ---------- Responsive ---------- */
@media (max-width:900px){
  .sea-hero{grid-template-columns:minmax(0,1fr);gap:2rem}
  .sea-hero-art{max-width:460px}
  .sea-stats{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width:640px){
  .block-container{padding:3.25rem 1rem 2rem !important}
  [class*="st-key-panel_"]{padding:1.15rem !important;border-radius:20px}
  .sea-chip{display:none}
  .sea-result{padding:1.5rem 1.25rem}
  .sea-file{flex-wrap:wrap}
  .sea-hrow{grid-template-columns:36px minmax(0,1fr) auto;gap:.75rem;padding:.8rem 1rem}
  .sea-hconf,.sea-hdur{display:none}
  .sea-facts{gap:1rem 1.75rem}
}
</style>
"""


# =============================================================================
# Small helpers
# =============================================================================
def render_html(markup, target=None):
    """Render raw HTML via st.markdown.

    Markdown treats indented lines as code and blank lines as the end of an
    HTML block, so every line is stripped and empty lines are dropped.
    """
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    (target or st).markdown(cleaned, unsafe_allow_html=True)


def keyed_container(key):
    """A container Streamlit exposes to CSS as .st-key-<key> (1.40+)."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def fmt_size(num_bytes):
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def fmt_duration(seconds):
    if seconds >= 60:
        return f"{int(seconds // 60)}:{int(seconds % 60):02d} min"
    return f"{seconds:.1f} s"


def fmt_khz(sample_rate):
    return f"{sample_rate / 1000:.2f}".rstrip("0").rstrip(".") + " kHz"


# =============================================================================
# SVG pieces
# =============================================================================
def brand_mark_svg():
    heights = [8, 16, 22, 16, 8]
    bars = "".join(
        f'<rect x="{8.5 + i * 4.4:.1f}" y="{16 - h / 2:.1f}" width="2.6" height="{h}" rx="1.3" fill="#FAF7F2"/>'
        for i, h in enumerate(heights)
    )
    return (
        '<svg viewBox="0 0 32 32" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="32" height="32" rx="9" fill="{ACCENT}"/>{bars}</svg>'
    )


def mic_icon_svg():
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="#A34A30" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="9" y="3" width="6" height="11" rx="3"/>'
        '<path d="M5.5 11.5a6.5 6.5 0 0 0 13 0"/><path d="M12 18v3"/></svg>'
    )


def file_icon_svg():
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="#A34A30" stroke-width="1.7" '
        'stroke-linecap="round" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M4 12v0M8 8v8M12 4v16M16 9v6M20 11v2"/></svg>'
    )


def info_icon_svg():
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="#A34A30" stroke-width="1.7" '
        'stroke-linecap="round" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><circle cx="12" cy="7.8" r=".6" fill="#A34A30"/></svg>'
    )


def hero_art_svg():
    """Decorative voice waveform around a microphone."""
    bars = []
    for i in range(33):
        x = 20 + i * 14
        closeness = math.exp(-(((x - 240) / 150) ** 2))
        variation = 0.45 + 0.55 * abs(math.sin(i * 1.9) * math.cos(i * 0.6 + 0.4))
        h = 12 + 170 * closeness * variation
        opacity = 0.28 + 0.72 * closeness
        delay = -((i * 0.37) % 2.4)
        duration = 1.7 + (i % 5) * 0.28
        bars.append(
            f'<rect class="sea-bar" x="{x}" y="{215 - h / 2:.1f}" width="7" height="{h:.1f}" rx="3.5" '
            f'fill="{ACCENT}" opacity="{opacity:.2f}" '
            f'style="animation-duration:{duration:.2f}s;animation-delay:{delay:.2f}s"/>'
        )
    mic = (
        f'<circle class="sea-halo" cx="240" cy="215" r="98" fill="none" stroke="{ACCENT}" stroke-width="1.5"/>'
        '<circle cx="240" cy="215" r="76" fill="#FFFDFA" stroke="#E7DDCF" stroke-width="1.5"/>'
        f'<rect x="224" y="176" width="32" height="58" rx="16" fill="{ACCENT}"/>'
        '<g stroke="#FAF7F2" stroke-opacity=".55" stroke-width="2.5" stroke-linecap="round">'
        '<line x1="231" y1="196" x2="249" y2="196"/><line x1="231" y1="205" x2="249" y2="205"/>'
        '<line x1="231" y1="214" x2="249" y2="214"/></g>'
        f'<g fill="none" stroke="{INK}" stroke-width="4" stroke-linecap="round">'
        '<path d="M208 212a32 32 0 0 0 64 0"/><line x1="240" y1="244" x2="240" y2="261"/>'
        '<line x1="225" y1="261" x2="255" y2="261"/></g>'
    )
    return (
        '<svg viewBox="0 72 480 286" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        + "".join(bars)
        + mic
        + "</svg>"
    )


def glyph_svg(pattern, color):
    """A tiny voice-print: one bar per value in `pattern` (0..1)."""
    bar_w, gap = 4.0, 2.6
    start = (60 - (len(pattern) * bar_w + (len(pattern) - 1) * gap)) / 2
    bars = "".join(
        f'<rect x="{start + i * (bar_w + gap):.1f}" y="{30 - (8 + v * 44) / 2:.1f}" '
        f'width="{bar_w}" height="{8 + v * 44:.1f}" rx="2" fill="{color}"/>'
        for i, v in enumerate(pattern)
    )
    return f'<svg viewBox="0 0 60 60" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'


def waveform_svg(env_min, env_max, bars=90):
    """A lightweight static waveform used in the upload preview."""
    peak = np.maximum(np.abs(env_min), np.abs(env_max))
    groups = [g.max() for g in np.array_split(peak, min(bars, len(peak))) if len(g)]
    top = max(max(groups), 1e-9)
    rects = []
    for i, value in enumerate(groups):
        h = max(4.0, (value / top) * 56)
        rects.append(
            f'<rect x="{i * 10}" y="{36 - h / 2:.1f}" width="5" height="{h:.1f}" rx="2.5" fill="{ACCENT}" opacity=".8"/>'
        )
    return (
        f'<svg viewBox="0 0 {len(groups) * 10 - 4} 72" preserveAspectRatio="none" aria-hidden="true" '
        'xmlns="http://www.w3.org/2000/svg">' + "".join(rects) + "</svg>"
    )


# =============================================================================
# Model + audio (unchanged logic)
# =============================================================================
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "emotion_xgboost.joblib"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    return model, encoder


# Keep this identical to the feature extraction used during training.
def extract_improved_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=40
    )

    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_chroma=12
    )

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(mel, axis=1),
        np.std(mel, axis=1)
    ])

    return features


def ensure_readable(paths):
    """Return a path librosa can decode.

    Recent librosa versions decode through soundfile only, which cannot open
    AAC/M4A. If soundfile can't read the file and ffmpeg is installed, convert
    it to a temporary PCM WAV first. This only changes how the file is
    *decoded*; feature extraction still receives ordinary audio.
    """
    source = paths[0]
    try:
        sf.SoundFile(source).close()
        return source
    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return source  # let librosa raise its own, readable error

    converted = source + ".decoded.wav"
    paths.append(converted)  # registered first so it is always cleaned up
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", source, "-vn", "-acodec", "pcm_s16le", converted],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return converted
    except Exception:
        return source


@contextmanager
def temp_audio_file(data, suffix):
    """Write bytes to a temp file (keeping the extension so the format can be
    identified), yield a readable path, and always delete everything after."""
    paths = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            paths.append(tmp.name)
        yield ensure_readable(paths)
    finally:
        for path in paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def compute_envelope(samples, bins=600):
    """Min/max per time bin: a compact summary for drawing a waveform."""
    bins = max(1, min(bins, len(samples)))
    edges = np.linspace(0, len(samples), bins + 1, dtype=int)
    mins = np.array([samples[a:b].min() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    maxs = np.array([samples[a:b].max() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    return mins, maxs


@st.cache_data(show_spinner=False, max_entries=16)
def inspect_audio(data, suffix):
    """Read-only look at a recording: duration, sample rates, waveform summary."""
    with temp_audio_file(data, suffix) as path:
        samples, _ = librosa.load(path, sr=TARGET_SR, mono=True)
        try:
            source_sr = int(librosa.get_samplerate(path))
        except Exception:
            source_sr = None
    if samples.size == 0:
        raise ValueError("The recording appears to be empty.")
    env_min, env_max = compute_envelope(samples)
    return {
        "duration": float(samples.size / TARGET_SR),
        "source_sr": source_sr,
        "env_min": env_min,
        "env_max": env_max,
    }


def analyze_recording(model, encoder, data, suffix, slot):
    """Run the original prediction pipeline, reporting progress to `slot`."""
    render_html(loading_card(0), slot)
    with temp_audio_file(data, suffix) as path:
        render_html(loading_card(1), slot)
        features = extract_improved_features(path)

        if features.shape[0] != model.n_features_in_:
            raise ValueError(
                f"Feature count mismatch. Expected {model.n_features_in_}, "
                f"but got {features.shape[0]}."
            )

        render_html(loading_card(2), slot)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        emotion = encoder.inverse_transform([prediction])[0]
        probabilities = model.predict_proba(features)[0]

    return str(emotion), [float(p) for p in probabilities]


# =============================================================================
# HTML builders
# =============================================================================
def topbar_html():
    return f"""
    <div class="sea-topbar">
      <div class="sea-brand">{brand_mark_svg()}<span>Speech Emotion AI</span></div>
      <div class="sea-chip"><i></i>XGBoost &middot; RAVDESS</div>
    </div>
    """


def hero_html(compact):
    legend = "".join(
        f'<span style="--c:{style_for(e)["color"]}"><i></i>{e.capitalize()}</span>' for e in LEGEND_ORDER
    )
    return f"""
    <div class="sea-hero{' sea-hero--compact' if compact else ''}">
      <div class="sea-hero-copy">
        <div class="sea-eyebrow">Voice emotion analysis</div>
        <div class="sea-title" role="heading" aria-level="1">Understand the <em>emotion</em> behind every voice.</div>
        <div class="sea-lede">Upload a short speech recording. Speech Emotion AI listens to its acoustic texture &mdash; tone, timbre and energy &mdash; and estimates which of eight emotions it most closely resembles.</div>
        <div class="sea-facts">
          <div class="sea-fact"><strong>8</strong><span>emotions</span></div>
          <div class="sea-fact"><strong>360</strong><span>acoustic features</span></div>
          <div class="sea-fact"><strong>5</strong><span>audio formats</span></div>
        </div>
      </div>
      <div class="sea-hero-art"><div class="sea-art">{hero_art_svg()}<div class="sea-legend">{legend}</div></div></div>
    </div>
    """


def section_header(eyebrow, title, subtitle=""):
    sub = f'<div class="sea-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    render_html(
        f"""
        <div class="sea-section">
          <div class="sea-eyebrow">{html.escape(eyebrow)}</div>
          <div class="sea-h2" role="heading" aria-level="2">{html.escape(title)}</div>
          {sub}
        </div>
        """
    )


def panel_head(title, subtitle, icon=""):
    icon_html = f'<div class="sea-panel-icon">{icon}</div>' if icon else ""
    return f"""
    <div class="sea-panel-head">
      {icon_html}
      <div>
        <div class="sea-panel-title" role="heading" aria-level="3">{html.escape(title)}</div>
        <div class="sea-panel-sub">{html.escape(subtitle)}</div>
      </div>
    </div>
    """


def tips_html():
    return """
    <div class="sea-tips"><b>For clearer results</b>
      <span class="t">One speaker</span><span class="t">Clear speech</span>
      <span class="t">Little background noise</span><span class="t">A few seconds is plenty</span>
    </div>
    """


def notice_html(title, message):
    return f"""
    <div class="sea-notice" role="alert">
      <div><strong>{html.escape(title)}</strong>{html.escape(message)}</div>
    </div>
    """


def file_card_html(name, size, suffix, info, analyzed=False):
    sample_rate = fmt_khz(info["source_sr"]) if info["source_sr"] else fmt_khz(TARGET_SR)
    return f"""
    <div class="sea-file">
      <div class="sea-file-icon">{file_icon_svg()}</div>
      <div class="sea-file-main">
        <div class="sea-file-name" title="{html.escape(name)}">{html.escape(name)}</div>
        <div class="sea-file-meta">
          <span>{suffix.lstrip('.').upper()}</span><span>{fmt_size(size)}</span>
          <span>{fmt_duration(info['duration'])}</span><span>{sample_rate}</span>
        </div>
      </div>
      <div class="sea-badge">{"Analyzed" if analyzed else "Ready to analyze"}</div>
    </div>
    <div class="sea-wavepreview">{waveform_svg(info['env_min'], info['env_max'])}</div>
    """


def loading_card(step):
    title, sub = LOADING_STEPS[step]
    percent = int((step + 1) / len(LOADING_STEPS) * 100)
    return f"""
    <div class="sea-loading" role="status" aria-live="polite">
      <div class="sea-loading-row">
        <div class="sea-loading-wave" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
        <div>
          <div class="sea-loading-title">{title}</div>
          <div class="sea-loading-sub">Step {step + 1} of {len(LOADING_STEPS)} &middot; {sub}</div>
        </div>
      </div>
      <div class="sea-track" aria-hidden="true"><i style="width:{percent}%"></i></div>
    </div>
    """


def result_card_html(result):
    pairs = sorted(zip(result["classes"], result["probs"]), key=lambda p: p[1], reverse=True)
    emotion = result["emotion"]
    style = style_for(emotion)
    confidence = dict(pairs)[emotion]
    runner_label, runner_prob = next((c, p) for c, p in pairs if c != emotion)

    hint = ""
    if confidence < 0.5:
        hint = (
            '<div class="sea-hint">The model is split between several emotions here '
            "(top score below 50%), so treat this result as tentative.</div>"
        )

    return f"""
    <div class="sea-result" style="--emo:{style['color']};--tint:{hex_to_rgba(style['color'], 0.16)}">
      <div class="sea-result-top">
        <div class="sea-eyebrow">Predicted emotion</div>
        <div class="sea-top-badge">Top match</div>
      </div>
      <div class="sea-result-glyph">{glyph_svg(style['pattern'], style['color'])}</div>
      <div class="sea-result-name">{html.escape(emotion.capitalize())}</div>
      <div class="sea-result-desc">{html.escape(style['desc'])}</div>
      <div class="sea-meter" role="img" aria-label="Model confidence {confidence * 100:.1f} percent"><i style="width:{confidence * 100:.1f}%"></i></div>
      <div class="sea-result-stats">
        <div><span>Model confidence</span><strong>{confidence * 100:.1f}%</strong></div>
        <div><span>Runner-up</span><strong>{html.escape(runner_label.capitalize())} &middot; {runner_prob * 100:.1f}%</strong></div>
      </div>
      {hint}
    </div>
    """


def score_list_html(result):
    pairs = sorted(zip(result["classes"], result["probs"]), key=lambda p: p[1], reverse=True)
    rows = "".join(
        f'<div class="sea-row"><span>{html.escape(c.capitalize())}</span><span>{p * 100:.2f}%</span></div>'
        for c, p in pairs
    )
    return f'<details class="sea-details"><summary>View exact scores</summary>{rows}</details>'


def stats_html(info):
    source_sr = info["source_sr"]
    rate_value = fmt_khz(source_sr) if source_sr else fmt_khz(TARGET_SR)
    rate_note = (
        f"Resampled to {fmt_khz(TARGET_SR)} for analysis" if source_sr and source_sr != TARGET_SR else "Analysis sample rate"
    )
    return f"""
    <div class="sea-stats">
      <div><div class="sea-stat-label">Duration</div><div class="sea-stat-value">{fmt_duration(info['duration'])}</div></div>
      <div><div class="sea-stat-label">Sample rate</div><div class="sea-stat-value">{rate_value}</div><div class="sea-stat-note">{rate_note}</div></div>
      <div><div class="sea-stat-label">Analysis format</div><div class="sea-stat-value">Mono</div><div class="sea-stat-note">{fmt_khz(TARGET_SR)}, as in training</div></div>
      <div><div class="sea-stat-label">Features</div><div class="sea-stat-value">360</div><div class="sea-stat-note">Mean &amp; std of MFCC, chroma and mel</div></div>
    </div>
    """


def note_html():
    return f"""
    <div class="sea-note">{info_icon_svg()}
      <div><strong>About these scores.</strong> The percentages are the model's estimates of how closely this
      recording resembles each emotion class in its training data &mdash; they are not definitive measurements of how
      someone actually feels. The model was trained on acted speech from the RAVDESS dataset and has a reported test
      accuracy of {REPORTED_TEST_ACCURACY}. Results can differ on other voices, accents, languages and recording
      conditions, and should not be used to make decisions about people.</div>
    </div>
    """


def history_html(history, current_id):
    rows = []
    for item in history:
        style = style_for(item["emotion"])
        confidence = dict(zip(item["classes"], item["probs"]))[item["emotion"]]
        current = " is-current" if item["id"] == current_id else ""
        rows.append(
            f'<div class="sea-hrow{current}" style="--tint:{hex_to_rgba(style["color"], 0.16)}">'
            f'<div class="sea-hglyph">{glyph_svg(style["pattern"], style["color"])}</div>'
            f'<div class="sea-hname" title="{html.escape(item["name"])}">{html.escape(item["name"])}</div>'
            f'<div class="sea-hemo">{html.escape(item["emotion"].capitalize())}</div>'
            f'<div class="sea-hconf">{confidence * 100:.1f}%</div>'
            f'<div class="sea-hdur">{fmt_duration(item["duration"])}</div></div>'
        )
    return f'<div class="sea-history">{"".join(rows)}</div>'


def footer_html():
    return """
    <div class="sea-footer" role="contentinfo">
      <span><strong>Speech Emotion AI</strong></span>
      <span>Powered by an XGBoost model trained on the RAVDESS dataset</span>
    </div>
    """


# =============================================================================
# Charts
# =============================================================================
def show_plotly(fig, key):
    config = {"displayModeBar": False}
    if PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", config=config, theme=None, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, config=config, theme=None, key=key)


def score_figure(result):
    pairs = sorted(zip(result["classes"], result["probs"]), key=lambda p: p[1], reverse=True)
    top = result["emotion"]
    labels = [c.capitalize() for c, _ in pairs]
    values = [p * 100 for _, p in pairs]
    colors = [style_for(c)["color"] if c == top else BAR_MUTED for c, _ in pairs]
    texts = [f"<b>{v:.1f}% · top</b>" if c == top else f"{v:.1f}%" for (c, _), v in zip(pairs, values)]
    ticks = [f"<b>{l}</b>" if c == top else l for (c, _), l in zip(pairs, labels)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=texts,
            textposition="outside",
            cliponaxis=False,
            textfont=dict(family=FONT_STACK, size=13, color=INK),
            hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
        )
    )
    try:  # rounded bar ends need Plotly 5.19+
        fig.update_traces(marker_cornerradius=6)
    except Exception:
        pass
    fig.update_layout(
        height=410,
        margin=dict(l=4, r=78, t=6, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=INK, size=13),
        bargap=0.36,
        showlegend=False,
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(family=FONT_STACK, color="#FAF7F2", size=13)),
    )
    fig.update_xaxes(
        range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%", gridcolor=GRID,
        zeroline=False, showline=False, tickfont=dict(color=MUTED, size=12), fixedrange=True,
    )
    fig.update_yaxes(
        autorange="reversed", tickmode="array", tickvals=labels, ticktext=ticks, showgrid=False,
        showline=False, ticks="", tickfont=dict(size=14, color=INK), fixedrange=True, automargin=True,
    )
    return fig


def score_bars_html(result):
    """Fallback chart when Plotly is not installed."""
    pairs = sorted(zip(result["classes"], result["probs"]), key=lambda p: p[1], reverse=True)
    rows = []
    for c, p in pairs:
        color = style_for(c)["color"] if c == result["emotion"] else BAR_MUTED
        weight = "600" if c == result["emotion"] else "400"
        rows.append(
            f'<div style="display:grid;grid-template-columns:90px 1fr 56px;gap:.75rem;align-items:center;margin:.55rem 0;font-weight:{weight}">'
            f'<span>{html.escape(c.capitalize())}</span>'
            f'<span style="height:12px;border-radius:99px;background:#EFE7DB;overflow:hidden"><i style="display:block;height:100%;width:{p * 100:.1f}%;background:{color};border-radius:99px"></i></span>'
            f'<span style="text-align:right">{p * 100:.1f}%</span></div>'
        )
    return f'<div>{"".join(rows)}</div>'


def waveform_figure(info):
    env_min, env_max = info["env_min"], info["env_max"]
    x = np.linspace(0, info["duration"], len(env_min))
    fig = go.Figure(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([env_max, env_min[::-1]]),
            fill="toself",
            mode="lines",
            line=dict(color=ACCENT, width=0.8),
            fillcolor="rgba(199,101,67,0.38)",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=4, r=4, t=6, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=INK, size=12),
        showlegend=False,
        dragmode="zoom",
    )
    fig.update_xaxes(
        title_text="Time (seconds)", title_font=dict(size=12, color=MUTED), tickfont=dict(color=MUTED, size=12),
        gridcolor=GRID, zeroline=False, showline=False,
    )
    fig.update_yaxes(visible=False, fixedrange=True, zeroline=True, zerolinecolor="#E3D6C5")
    return fig


# =============================================================================
# Page sections
# =============================================================================
def render_workspace(model, encoder):
    """Upload panel. Returns (info, file_hash) for the current recording."""
    info, file_hash = None, None

    with keyed_container("panel_workspace"):
        render_html(
            panel_head(
                "Drop in a voice recording",
                "Drag and drop a file onto the area below, or use Upload to browse. WAV, MP3, M4A, OGG and FLAC are supported.",
                mic_icon_svg(),
            )
        )
        uploaded = st.file_uploader(
            "Upload an audio recording",
            type=SUPPORTED_TYPES,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )

        if uploaded is None:
            render_html(tips_html())
        else:
            data = uploaded.getvalue()
            suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
            file_hash = hashlib.sha1(data).hexdigest()[:16]
            try:
                info = inspect_audio(data, suffix)
            except Exception as exc:
                render_html(
                    notice_html(
                        "We couldn't read this recording",
                        "Please check that it is a valid audio file. MP3 and M4A files also need ffmpeg "
                        f"installed. Details: {exc}",
                    )
                )
            else:
                already_done = any(r["id"] == file_hash for r in st.session_state.history)
                render_html(file_card_html(uploaded.name, len(data), suffix, info, analyzed=already_done))
                st.audio(data, format=AUDIO_MIME.get(suffix, "audio/wav"))

        left, right = st.columns([3, 1])
        with left:
            analyze = stretch_button(
                "Analyze Emotion", type="primary", key="analyze_btn", disabled=info is None
            )
        with right:
            clear = stretch_button("Clear", key="clear_btn", disabled=uploaded is None)
        status_slot = st.empty()

    if clear:
        st.session_state.uploader_key += 1
        st.rerun()

    if analyze and info is not None:
        try:
            emotion, probs = analyze_recording(model, encoder, data, suffix, status_slot)
        except Exception as exc:
            render_html(
                notice_html(
                    "Could not analyze this audio",
                    "Please check that it is a valid, supported audio file. " f"Details: {exc}",
                ),
                status_slot,
            )
        else:
            result = {
                "id": file_hash,
                "name": uploaded.name,
                "emotion": emotion,
                "classes": [str(c) for c in encoder.classes_],
                "probs": probs,
                "duration": info["duration"],
            }
            others = [r for r in st.session_state.history if r["id"] != file_hash]
            st.session_state.history = ([result] + others)[:MAX_HISTORY]
            st.rerun()

    return info, file_hash


def render_results(result, info):
    section_header("Results", "What the model heard", f"Analysis of {result['name']}")

    left, right = st.columns([5, 6], gap="large")
    with left:
        render_html(result_card_html(result))
    with right:
        with keyed_container("panel_chart"):
            render_html(
                panel_head(
                    "Emotion scores",
                    "Model-estimated score for each of the eight emotion classes.",
                )
            )
            if PLOTLY_AVAILABLE:
                show_plotly(score_figure(result), key=f"scores_{result['id']}")
            else:
                render_html(score_bars_html(result))
            render_html(score_list_html(result))

    section_header("Audio insights", "The signal behind the score")
    with keyed_container("panel_wave"):
        subtitle = (
            "Drag across the plot to zoom in; double-click to reset."
            if PLOTLY_AVAILABLE
            else "Amplitude over time."
        )
        render_html(panel_head("Waveform", subtitle))
        if PLOTLY_AVAILABLE:
            show_plotly(waveform_figure(info), key=f"wave_{result['id']}")
        else:
            render_html(f'<div class="sea-wavepreview">{waveform_svg(info["env_min"], info["env_max"])}</div>')
        render_html(stats_html(info))

    render_html(note_html())


def render_history(current_id):
    history = st.session_state.history
    head, action = st.columns([4, 1])
    with head:
        section_header("This session", "Recordings you've analyzed", f"{len(history)} so far, most recent first.")
    with action:
        st.markdown("<div style='height:3.4rem'></div>", unsafe_allow_html=True)
        if stretch_button("Clear history", key="clear_history"):
            st.session_state.history = []
            st.rerun()
    render_html(history_html(history, current_id))


# =============================================================================
# App
# =============================================================================
def main():
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("history", [])

    # Styles and top bar share one element so no empty gap appears above them.
    render_html(STYLE + topbar_html())

    try:
        model, encoder = load_model()
    except Exception as exc:
        render_html(
            notice_html(
                "The model could not be loaded",
                f"Check that the files in {MODEL_DIR} exist and that xgboost and scikit-learn are installed. "
                f"Details: {exc}",
            )
        )
        st.stop()

    # Once something has been analyzed, the hero shrinks so results sit higher.
    render_html(hero_html(compact=bool(st.session_state.history)))

    info, file_hash = render_workspace(model, encoder)

    current = next((r for r in st.session_state.history if r["id"] == file_hash), None)
    if current is not None and info is not None:
        render_results(current, info)

    if len(st.session_state.history) >= 2:
        render_history(file_hash)

    render_html(footer_html())


main()
