"""Run the redesigned VAJRA control room without modifying the demo app."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_app.server import app, loop_state  # noqa: E402


if __name__ == "__main__":
    app.template_folder = str(Path(__file__).with_name("templates"))
    if loop_state.sim is None:
        loop_state.reset()
    app.run(host="127.0.0.1", port=5050, debug=False)
