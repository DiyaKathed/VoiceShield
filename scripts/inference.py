"""
VoiceShield Inference Script
============================
Convenience entrypoint delegating to ml.inference.
Runs deepfake inference on arbitrary audio files using VoiceShieldNet.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.inference import main, VoiceShieldDetector

if __name__ == "__main__":
    main()
