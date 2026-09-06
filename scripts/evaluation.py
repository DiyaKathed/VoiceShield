"""
VoiceShield Evaluation Script
=============================
Convenience entrypoint delegating to ml.evaluation.
Evaluates VoiceShieldNet on the Kaggle Fake and Real Audio held-out test split.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.evaluation import main

if __name__ == "__main__":
    main()
