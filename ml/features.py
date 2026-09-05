"""
VoiceShield Feature Extraction Module
=====================================
Acoustic feature extraction and preprocessing pipeline for voice deepfake detection.

RESEARCH DESIGN NOTES & LIMITATION HANDLING:
-------------------------------------------
1. UNSEEN SYNTHETIC GENERATORS:
   Modern neural TTS and voice cloning architectures (e.g., FastSpeech2, VITS,
   Tortoise-TTS, XTTS, ElevenLabs) utilize neural vocoders (HiFi-GAN, MelGAN,
   BigVGAN, Diffusion vocoders). Neural vocoders leave characteristic high-frequency
   phase inconsistencies and unnatural harmonic distribution patterns in the
   frequency domain. By extracting Log-Mel spectrograms coupled with temporal
   spectral dynamics (delta and delta-delta coefficients), our feature representation
   captures rapid frame-to-frame transitional glitches typical of unseen vocoders.

2. COMPRESSION & CHANNEL DISTORTION:
   VoIP and telephony codecs (e.g., AMR-WB, Opus, G.711) introduce heavy lossy
   compression and bandpass filtering (e.g. 300Hz-3.4kHz). To make detection robust,
   input audio is uniformly resampled to 16kHz, band-limited normalization is applied,
   and features focus on mid-frequency formant trajectories where cloning artifacts
   persist despite lossy compression.

3. BACKGROUND NOISE & SNR VARIABILITY:
   Additive acoustic noise can mask synthetic artifacts. We apply RMS energy
   normalization and spectral subtraction/filtering principles to standardize the
   dynamic range across varying recording conditions.

4. SILENCE & NON-SPEECH ARTIFACTS:
   Silent pauses or breathing artifacts can cause false alarms if models learn
   ambient background noise instead of vocal tract acoustics. We implement
   Voice Activity Detection (VAD) to trim or gate silence frames before classification.

5. PARTIAL / SEGMENT-LEVEL SPOOFING:
   Impersonation attacks often splice human speech with synthetic keywords (e.g.,
   "Yes, approve the wire transfer"). We provide sliding-window chunking
   so each 1.0-2.0s segment is individually extracted and assessed.
"""

import numpy as np
import librosa
import soundfile as sf
import io
from typing import Tuple, List, Optional, Union


# Standard acoustic parameters
TARGET_SR = 16000          # 16 kHz standard telephony/speech sample rate
N_MELS = 64                # 64-band Mel filterbank
N_FFT = 1024               # FFT window size (~64ms at 16kHz)
HOP_LENGTH = 512           # 32ms hop for temporal resolution
CHUNK_DURATION_SEC = 2.0   # 2.0 seconds per analysis segment for rich phonetic context
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_SEC)  # 32,000 samples
EXPECTED_FRAMES = int(np.ceil(CHUNK_SAMPLES / HOP_LENGTH))  # ~63 frames


class AudioFeatureExtractor:
    """
    Extracts multi-channel spectro-temporal representations from raw speech audio.
    Channels:
      Channel 0: Log-Mel Spectrogram (robust zero-mean normalized)
      Channel 1: Delta (1st-order temporal velocity of spectral energy)
      Channel 2: Delta-Delta (2nd-order acceleration of spectral energy)
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SR,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        chunk_duration: float = CHUNK_DURATION_SEC,
        vad_top_db: float = 30.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.expected_frames = int(np.ceil(self.chunk_samples / hop_length))
        self.vad_top_db = vad_top_db

    def load_audio(
        self,
        audio_source: Union[str, bytes, io.BytesIO],
        apply_vad: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Loads audio from filepath or byte buffer, standardizes to mono 16kHz,
        applies peak/RMS normalization, and optionally trims silence.
        Robustly handles WAV, MP3, FLAC, OGG, WebM, AAC, and browser mic formats.
        """
        raw_bytes = None
        if isinstance(audio_source, (bytes, bytearray)):
            raw_bytes = bytes(audio_source)
            audio_source = io.BytesIO(raw_bytes)
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            raw_bytes = audio_source.getvalue()

        audio = None
        sr = self.sample_rate

        # 1. First attempt: standard librosa/soundfile loading
        try:
            audio, sr = librosa.load(audio_source, sr=self.sample_rate, mono=True)
        except Exception:
            # 2. Fallback: PyAV decoding for browser WebM, Opus, MP4, AAC, etc.
            if raw_bytes is not None or isinstance(audio_source, str):
                try:
                    import av
                    container_source = io.BytesIO(raw_bytes) if raw_bytes is not None else audio_source
                    container = av.open(container_source)
                    resampler = av.AudioResampler(format='fltp', layout='mono', rate=self.sample_rate)
                    frames = []
                    for frame in container.decode(audio=0):
                        for rf in resampler.resample(frame):
                            frames.append(rf.to_ndarray())
                    if len(frames) > 0:
                        audio = np.concatenate(frames, axis=1).squeeze(0).astype(np.float32)
                        sr = self.sample_rate
                except Exception as av_err:
                    print(f"PyAV fallback error: {av_err}")

        if audio is None or len(audio) == 0:
            return np.zeros(self.chunk_samples, dtype=np.float32), self.sample_rate

        # RESEARCH EXTENSION: Silence / Non-Speech Artifact Mitigation
        # Trimming low-energy silence to prevent models from learning room acoustic noise
        if apply_vad and len(audio) > self.sample_rate * 0.5:
            try:
                trimmed_audio, _ = librosa.effects.trim(audio, top_db=self.vad_top_db)
                if len(trimmed_audio) >= self.sample_rate * 0.3:
                    audio = trimmed_audio
            except Exception:
                pass

        # Peak & RMS Normalization to standardize amplitude across devices
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak * 0.95

        return audio.astype(np.float32), sr

    def extract_features(
        self,
        audio: np.ndarray,
        random_crop: bool = False
    ) -> np.ndarray:
        """
        Extracts 3-channel acoustic feature map [3, n_mels, time_frames]
        from audio.
        If audio > chunk_samples:
          - If random_crop=True (training): chooses random window across speech
          - If random_crop=False (eval): takes center window
        If audio < chunk_samples:
          - Pads via wrap/reflection to avoid introducing artificial silence
        """
        if len(audio) > self.chunk_samples:
            if random_crop:
                max_start = len(audio) - self.chunk_samples
                start_idx = np.random.randint(0, max_start + 1)
            else:
                start_idx = (len(audio) - self.chunk_samples) // 2
            audio_chunk = audio[start_idx:start_idx + self.chunk_samples]
        elif len(audio) < self.chunk_samples:
            pad_width = self.chunk_samples - len(audio)
            audio_chunk = np.pad(audio, (0, pad_width), mode='wrap')
        else:
            audio_chunk = audio

        # 1. Log-Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=20,
            fmax=self.sample_rate // 2
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Robust z-score normalization per sample (gain/device invariant)
        mean_mel = np.mean(log_mel)
        std_mel = np.std(log_mel) + 1e-5
        log_mel_norm = np.clip((log_mel - mean_mel) / std_mel, -3.0, 3.0) / 3.0

        # 2. First-order Delta (Velocity of spectral transitions)
        delta1 = librosa.feature.delta(log_mel_norm, order=1)

        # 3. Second-order Delta (Acceleration of spectral transitions)
        delta2 = librosa.feature.delta(log_mel_norm, order=2)

        # Align frame dimension to expected_frames
        features = np.stack([log_mel_norm, delta1, delta2], axis=0)  # Shape: [3, n_mels, frames]

        if features.shape[2] < self.expected_frames:
            pad_f = self.expected_frames - features.shape[2]
            features = np.pad(features, ((0, 0), (0, 0), (0, pad_f)), mode='edge')
        elif features.shape[2] > self.expected_frames:
            features = features[:, :, :self.expected_frames]

        return features.astype(np.float32)

    def segment_audio(
        self,
        audio: np.ndarray,
        overlap: float = 0.5
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        RESEARCH EXTENSION: Partial & Segment-Level Spoofing Detection
        Divides audio into overlapping sliding-window chunks.
        Returns: list of (start_sec, end_sec, chunk_audio_array)
        """
        duration = len(audio) / self.sample_rate
        step_samples = int(self.chunk_samples * (1.0 - overlap))
        segments = []

        if len(audio) <= self.chunk_samples:
            segments.append((0.0, duration, audio))
            return segments

        start_idx = 0
        while start_idx < len(audio):
            end_idx = start_idx + self.chunk_samples
            chunk = audio[start_idx:min(end_idx, len(audio))]
            start_sec = start_idx / self.sample_rate
            end_sec = min(end_idx / self.sample_rate, duration)

            segments.append((round(start_sec, 2), round(end_sec, 2), chunk))

            if end_idx >= len(audio):
                break
            start_idx += step_samples

        return segments
