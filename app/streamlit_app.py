"""Q-Edge Web Application.

Serves the modern Q-Edge interface.
Run with: uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
INDEX_PATH = STATIC_DIR / "index.html"
CSS_PATH = STATIC_DIR / "style.css"
JS_PATH = STATIC_DIR / "app.js"
SAMPLE_PATH = APP_DIR / "assets" / "sample.png"

st.set_page_config(
    page_title="Q-Edge — Quantum-Assisted Edge Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit chrome so the new interface is displayed full-screen without previous UI elements
st.markdown(
    """
    <style>
    #MainMenu, header, footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    .main, .block-container, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        z-index: 999999 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_html_content() -> str:
    """Bundle the new web frontend (HTML, CSS, JS) into a self-contained page."""
    html = INDEX_PATH.read_text(encoding="utf-8")
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    js = JS_PATH.read_text(encoding="utf-8") if JS_PATH.exists() else ""

    # Inline stylesheet
    html = html.replace(
        '<link rel="stylesheet" href="style.css" />',
        f"<style>\n{css}\n</style>",
    )

    # Prepare bundled sample image if available
    sample_setup = ""
    if SAMPLE_PATH.exists():
        sample_b64 = base64.b64encode(SAMPLE_PATH.read_bytes()).decode("utf-8")
        sample_data_url = f"data:image/png;base64,{sample_b64}"
        sample_setup = f"window.__BUNDLED_SAMPLE_URL__ = '{sample_data_url}';\n"

    # Inline script
    html = html.replace(
        '<script src="app.js"></script>',
        f"<script>\n{sample_setup}{js}\n</script>",
    )

    return html


components.html(get_html_content(), height=1200, scrolling=True)
