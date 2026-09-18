"""Render the configured current-density animation.

Usage: ``python run_current_animation.py``
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics.current_animation import main


if __name__ == "__main__":
    main()
