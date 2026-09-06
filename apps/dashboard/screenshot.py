"""Generate README screenshots of the Model Builder Dashboard using Playwright.

Usage:
    pip install playwright
    playwright install chromium
    python apps/dashboard/screenshot.py

Requires the dashboard server to be running on http://localhost:8000
Start with: uvicorn apps.dashboard.app:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

SCREENSHOTS_DIR = Path(__file__).parent.parent.parent / "docs" / "screenshots"
DASHBOARD_URL = "http://localhost:8000"


def ensure_dir():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def take_screenshots():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is not installed.")
        print("Install with: pip install playwright && playwright install chromium")
        sys.exit(1)

    ensure_dir()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # 1. Dashboard overview
        print("Capturing: dashboard-overview.png")
        page.goto(DASHBOARD_URL)
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(SCREENSHOTS_DIR / "dashboard-overview.png"), full_page=True)

        # 2. Select a dataset and show column browser
        print("Capturing: column-browser.png")
        page.select_option("#dataset-select", index=1)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "column-browser.png"), full_page=True)

        # 3. Target selector view
        print("Capturing: target-selector.png")
        page.screenshot(path=str(SCREENSHOTS_DIR / "target-selector.png"), full_page=True)

        # 4. Model history view
        print("Capturing: model-history.png")
        page.click('[data-panel="history"]')
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "model-history.png"), full_page=True)

        # Switch back to builder
        page.click('[data-panel="builder"]')
        page.wait_for_timeout(500)

        # 5. Results metrics (if training is possible)
        print("Capturing: results-metrics.png")
        # Training requires API connection — capture whatever is visible
        page.screenshot(path=str(SCREENSHOTS_DIR / "results-metrics.png"), full_page=True)

        # 6. Strategy export
        print("Capturing: strategy-export.png")
        page.screenshot(path=str(SCREENSHOTS_DIR / "strategy-export.png"), full_page=True)

        browser.close()
        print(f"\nScreenshots saved to: {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    take_screenshots()
