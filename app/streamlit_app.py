"""Q-Edge Web Application.

Serves the modern Q-Edge interface.
Run with: uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import base64
import http.server
import json
import threading
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from q_edge.ui import (
    get_baseline_benchmarks,
    process_benchmark_request,
    process_pipeline_request,
    process_scaling_request,
)

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


API_PORT = 8585
_SERVER_RUNNING = False
_LOCK = threading.Lock()


class APIHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/api/benchmark", "/api/benchmarks_baseline"):
            try:
                response_data = get_baseline_benchmarks()
                resp_bytes = json.dumps(response_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as exc:
                err_bytes = json.dumps({"status": "error", "message": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
            if self.path == "/api/process":
                response_data = process_pipeline_request(payload)
            elif self.path == "/api/benchmark":
                response_data = process_benchmark_request(payload)
            elif self.path == "/api/scaling":
                response_data = process_scaling_request(payload)
            else:
                self.send_response(404)
                self.end_headers()
                return

            resp_bytes = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        except Exception as exc:
            err_bytes = json.dumps({"status": "error", "message": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(err_bytes)))
            self.end_headers()
            self.wfile.write(err_bytes)

    def log_message(self, format: str, *args: object) -> None:
        pass


def ensure_backend_server() -> int:
    global _SERVER_RUNNING
    with _LOCK:
        if _SERVER_RUNNING:
            return API_PORT
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", API_PORT), APIHandler)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            _SERVER_RUNNING = True
            return API_PORT
        except OSError:
            _SERVER_RUNNING = True
            return API_PORT


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

    port = ensure_backend_server()
    baseline_json = json.dumps(get_baseline_benchmarks())
    backend_setup = (
        f"window.__BACKEND_API__ = 'http://127.0.0.1:{port}/api/process';\n"
        f"window.__BENCHMARK_API__ = 'http://127.0.0.1:{port}/api/benchmark';\n"
        f"window.__SCALING_API__ = 'http://127.0.0.1:{port}/api/scaling';\n"
        f"window.__INITIAL_BENCHMARKS__ = {baseline_json};\n"
    )

    LOGO_PATH = STATIC_DIR / "logo.png"
    if LOGO_PATH.exists():
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        logo_data_url = f"data:image/png;base64,{logo_b64}"
        html = html.replace('src="./logo.png"', f'src="{logo_data_url}"')
        html = html.replace('src="logo.png"', f'src="{logo_data_url}"')

    # Prepare bundled sample image if available
    sample_setup = ""
    if SAMPLE_PATH.exists():
        sample_b64 = base64.b64encode(SAMPLE_PATH.read_bytes()).decode("utf-8")
        sample_data_url = f"data:image/png;base64,{sample_b64}"
        sample_setup = f"window.__BUNDLED_SAMPLE_URL__ = '{sample_data_url}';\n"

    # Inline script
    html = html.replace(
        '<script src="app.js"></script>',
        f"<script>\n{backend_setup}{sample_setup}{js}\n</script>",
    )

    return html


components.html(get_html_content(), height=1200, scrolling=True)
