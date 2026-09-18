"""Run the configured driven 2DEG propagation.

Usage: ``python run_dynamics.py``
Edit user parameters in ``dynamics/config.py`` first.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamics.propagation import main


if __name__ == "__main__":
    main()
