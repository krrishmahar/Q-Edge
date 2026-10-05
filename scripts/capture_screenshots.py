"""Capture documentation screenshots of the running Streamlit app.

Playwright is not a project dependency. Run it ad hoc against a local Chrome install:

    uv run streamlit run app/streamlit_app.py --server.headless true --server.port 8601
    uv run --with playwright python scripts/capture_screenshots.py --url http://localhost:8601

Screenshots are written to ``docs/screenshots/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
VIEWPORT = {"width": 1600, "height": 1300}


def wait_idle(page: Page, timeout: float = 180_000) -> None:
    """Wait until Streamlit has finished re-running the script."""
    page.wait_for_timeout(800)
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=timeout)
    page.wait_for_timeout(1500)


def sidebar_select(page: Page, label: str, option: str) -> None:
    """Choose ``option`` in the sidebar selectbox labelled ``label``."""
    page.get_by_role("combobox", name=label).click()
    page.get_by_role("option", name=option, exact=True).click()
    wait_idle(page)


def shot(page: Page, name: str, full_page: bool = False) -> None:
    """Save a screenshot under ``docs/screenshots``."""
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=full_page)
    print(f"saved {path.relative_to(ROOT)}")


def scroll_main(page: Page, text: str) -> None:
    """Scroll the main area so the element containing ``text`` is at the top."""
    page.get_by_text(text, exact=True).first.evaluate("el => el.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(1000)


def main() -> None:
    """Drive the app through its main states and capture each one."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8601")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(args.url)
        page.get_by_text("Processed size").wait_for(timeout=180_000)
        wait_idle(page)

        shot(page, "01-results-overview")
        scroll_main(page, "Metrics")
        shot(page, "02-metrics-and-f1")

        page.get_by_text("Q-Edge: quantum Hadamard edge detection").first.evaluate(
            "el => el.scrollIntoView({block: 'start'})"
        )
        sidebar_select(page, "Classical comparison", "Canny")
        page.locator('[data-testid="stSidebar"]').get_by_text(
            "Show thresholded (binary) maps"
        ).click()
        wait_idle(page)
        shot(page, "03-binary-vs-canny")

        page.get_by_role("tab", name="Benchmarks").click()
        page.get_by_role("button", name="Run scaling benchmark (480p to 4K)").click()
        page.get_by_text("Runtime vs resolution").first.wait_for(timeout=300_000)
        wait_idle(page)
        shot(page, "04-benchmarks-scaling")

        page.get_by_role("tab", name="Scalability").click()
        page.wait_for_timeout(2500)
        shot(page, "05-scalability-architecture")

        page.get_by_role("tab", name="Results").click()
        sidebar_select(page, "Backend", "aer")
        shot(page, "06-aer-backend")

        sidebar_select(page, "Backend", "ibm-hardware")
        shot(page, "07-hardware-stub-error")

        sidebar_select(page, "Backend", "numpy")
        page.locator('[data-testid="stFileUploaderDropzoneInput"]').set_input_files(
            files=[
                {
                    "name": "broken.png",
                    "mimeType": "image/png",
                    "buffer": b"not really a png",
                }
            ]
        )
        wait_idle(page)
        shot(page, "08-bad-upload-error")

        browser.close()


if __name__ == "__main__":
    main()
