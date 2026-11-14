"""Headless Streamlit AppTest verifying the single-screen Red-Flag Highlighter UI."""

from pathlib import Path
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"


def test_single_screen_streamlit_app_renders_and_highlights_traps() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()

    assert len(at.exception) == 0, f"Unexpected Streamlit exception: {at.exception}"
    assert len(at.metric) == 4
    # Default preset is a predatory Ride-Share ToS (Grade F -> renders 1 st.error verdict banner)
    assert len(at.error) == 1
    assert "Grade F" in at.metric[0].value or "Grade D" in at.metric[0].value

    # Trigger primary scan button
    at.button[0].click().run()
    assert len(at.exception) == 0
    assert "Grade" in at.metric[0].value
