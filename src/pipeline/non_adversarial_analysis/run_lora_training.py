#!/usr/bin/env python3
"""
Wrapper script to run LORA training from the project root.

Usage:
    python run_lora_training.py --mode prepare-data --experiment-mode qa
    python run_lora_training.py --mode train --experiment-mode qa
    python run_lora_training.py --mode full-pipeline --experiment-mode qa
"""

import sys
import os

# Add the parent of non_adversarial_setting (i.e. src/) to sys.path so that
# the package's relative imports (from .training_config import ...) resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import via the package so relative imports inside train_and_eval work.
from non_adversarial_setting.train_and_eval import main

if __name__ == "__main__":
    main()
