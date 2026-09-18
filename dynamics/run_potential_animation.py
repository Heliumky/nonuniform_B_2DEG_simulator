"""Render the configured ``t=0`` sine-versus-cosine effective potentials.

Usage: ``python run_potential_animation.py``
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics.potential_comparison import main


if __name__ == "__main__":
    main()
