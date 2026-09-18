"""Public stage 1: calculate and save the driven wavefunction trajectory."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .propagation import main as _trajectory_main
else:  # Support ``cd dynamics && python dynamics.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynamics.propagation import main as _trajectory_main


def main(force=False):
    """Create or reuse ``config.TRAJECTORY_FILE`` for the configured run."""
    _trajectory_main(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the configured driven wavefunction trajectory.")
    parser.add_argument("--force", action="store_true", help="recompute and overwrite the trajectory")
    try:
        main(force=parser.parse_args().force)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot create trajectory: {error}", file=sys.stderr)
        raise SystemExit(1)
