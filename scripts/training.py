"""
VoiceShield Training Script
===========================
Convenience entrypoint delegating to ml.training.
Trains VoiceShieldNet on the Kaggle Fake and Real Audio Dataset.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.training import main

if __name__ == "__main__":
    main()
