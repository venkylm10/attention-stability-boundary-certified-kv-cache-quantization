"""Alias entry point — this experiment saves no model weights (inference-only
baseline evaluation of a pretrained checkpoint), so evaluation IS the main
pipeline. See train.py for the full implementation; this module just
re-exports its `main`.
"""

from train import main

if __name__ == "__main__":
    main()
