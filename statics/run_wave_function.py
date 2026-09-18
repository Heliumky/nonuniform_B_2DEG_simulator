"""Render the configured static wavefunction probability.

Usage: ``python run_wave_function.py``
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from statics.wave_function import main


if __name__ == "__main__":
    main()
