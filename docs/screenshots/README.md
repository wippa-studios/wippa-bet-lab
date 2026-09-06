# Dashboard Screenshots

Screenshots of the Model Builder Dashboard for the project README.

## Generating Screenshots

### Prerequisites

1. Install Playwright:
   ```bash
   pip install playwright
   playwright install chromium
   ```

2. Start the dashboard server:
   ```bash
   # From the project root
   uvicorn apps.dashboard.app:app --host 0.0.0.0 --port 8000
   ```

3. (Optional) Start the API server for full data:
   ```bash
   uvicorn betlab.api:app --host 0.0.0.0 --port 8001
   ```

### Run the Screenshot Script

```bash
python apps/dashboard/screenshot.py
```

This generates the following files:

| File | Description |
|------|-------------|
| `dashboard-overview.png` | Full dashboard with dataset selector |
| `column-browser.png` | Column browser grid with stats |
| `target-selector.png` | Target and feature selection panel |
| `results-metrics.png` | Training results and metrics cards |
| `feature-importance.png` | Feature importance bar chart |
| `strategy-export.png` | Strategy export configuration |

### Manual Screenshots

If the script fails, you can take screenshots manually:

1. Open `http://localhost:8000` in Chrome/Chromium
2. Use DevTools → Performance → Screenshot capture
3. Or use a screen recording tool and export frames

### Updating the README

Replace the image references in `README.md`:

```markdown
![Dashboard Overview](docs/screenshots/dashboard-overview.png)
![Column Browser](docs/screenshots/column-browser.png)
![Results Metrics](docs/screenshots/results-metrics.png)
```

### Notes

- Screenshots require a running server with at least one dataset loaded
- The dashboard auto-connects to the API at `/api/`
- For best results, use a 1440x900 viewport
- All screenshots use the dark theme (default and only theme)
