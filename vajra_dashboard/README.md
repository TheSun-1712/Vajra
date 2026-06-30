# VAJRA Control Room

This is an isolated dashboard skin for the existing `demo_app` simulation. It
does not modify any existing project file.

## Run

From the repository root:

```powershell
python vajra_dashboard/server.py
```

Open <http://127.0.0.1:5050>.

## Runtime resources

- Python packages already used by the project: Flask, NumPy, Pillow, PyTorch,
  and the dependencies imported by `vajra`.
- Internet access on first load for JetBrains Mono, Inter, and Chart.js.
- For a fully offline control-room deployment, vendor those font files and the
  Chart.js bundle locally, then replace the three CDN references in the template.
