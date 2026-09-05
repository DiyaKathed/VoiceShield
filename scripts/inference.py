"""
VoiceShield Inference Script (IndicTTS)
=======================================
Convenience entrypoint delegating to ml.inference
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.inference import main, VoiceShieldDetector

if __name__ == "__main__":
    main()
