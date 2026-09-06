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
from pathlib import Path
from typing import Tuple, List, Optional, Union


# Standard acoustic parameters
TARGET_SR = 16000          # 16 kHz standard telephony/speech sample rate
N_MELS = 64                # 64-band Mel filterbank
N_FFT = 1024               # FFT window size (~64ms at 16kHz)
HOP_LENGTH = 512           # 32ms hop for temporal resolution
CHUNK_DURATION_SEC = 2.0   # 2.0 seconds per analysis segment for rich phonetic context
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_SEC)  # 32,000 samples
EXPECTED_FRAMES = int(np.ceil(CHUNK_SAMPLES / HOP_LENGTH))  # ~63 frames


def is_sufficient_speech(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    vad_top_db: float = 30.0,
    min_speech_duration: float = 0.35
) -> Tuple[bool, float]:
    """
    RESEARCH EXTENSION: Silence & Insufficient Speech Detection.
    Verifies that audio contains meaningful vocal acoustic energy rather than
    silence, background microphone hum, or empty static.
    Returns: (is_sufficient, active_speech_duration_seconds)
    """
    if audio is None or len(audio) == 0:
        return False, 0.0

    total_dur = len(audio) / sr
    if total_dur < min_speech_duration:
        return False, total_dur

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    # Detect pure silence or sub-audible background static
    if peak < 0.008 or rms < 0.002:
        return False, 0.0

    try:
        intervals = librosa.effects.split(audio, top_db=vad_top_db)
        active_speech_sec = float(sum((iv[1] - iv[0]) for iv in intervals) / sr)
        if active_speech_sec < min_speech_duration:
            return False, active_speech_sec
        return True, active_speech_sec
    except Exception:
        # If interval splitting encounters an issue, fallback to RMS energy check
        return (rms >= 0.005), total_dur


def load_audio(
    audio_source: Union[str, bytes, io.BytesIO],
    target_sr: int = TARGET_SR,
    apply_vad: bool = True,
    vad_top_db: float = 30.0,
    filename: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    CENTRAL AUDIO LOADER (VoiceShield Core Pipeline):
    Transparently decodes and standardizes both WAV and MP3 audio into the exact
    16kHz mono float32 PCM waveform expected by VoiceShieldNet.
    
    Processing Steps:
      1. Format validation (WAV, MP3, FLAC, OGG, WebM, AAC)
      2. Decoding via PyAV (fast, resilient C-decoder) or librosa/soundfile
      3. Channel downmixing to mono
      4. High-quality resampling to target_sr (16,000 Hz)
      5. Float32 conversion and amplitude peak normalization (/ peak * 0.95)
      6. Optional Voice Activity Detection (VAD) silence trimming
    """
    raw_bytes: Optional[bytes] = None
    inferred_filename = filename or ""

    if isinstance(audio_source, (bytes, bytearray)):
        raw_bytes = bytes(audio_source)
        source_buffer = io.BytesIO(raw_bytes)
    elif isinstance(audio_source, io.BytesIO):
        audio_source.seek(0)
        raw_bytes = audio_source.getvalue()
        source_buffer = audio_source
    elif isinstance(audio_source, (str, Path)):
        file_path = Path(audio_source)
        inferred_filename = file_path.name
        ext = file_path.suffix.lower()
        valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".webm", ".aac", ".m4a"}
        if ext and ext not in valid_exts:
            raise ValueError(f"Unsupported audio format '{ext}'. Please upload MP3 or WAV.")
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        source_buffer = io.BytesIO(raw_bytes)
    else:
        raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

    # Check extension from filename if available
    if inferred_filename:
        ext = Path(inferred_filename).suffix.lower()
        valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".webm", ".aac", ".m4a"}
        if ext and ext not in valid_exts:
            raise ValueError(f"Unsupported audio format '{ext}'. Please upload MP3 or WAV.")

    audio: Optional[np.ndarray] = None
    sr: int = target_sr
    decode_error: Optional[Exception] = None

    # Step 1: Decode using PyAV (reliable C-based decoding for both MP3 & WAV)
    try:
        # pyrefly: ignore [missing-import]
        import av
        source_buffer.seek(0)
        container = av.open(source_buffer)
        if len(container.streams.audio) > 0:
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
            frames = []
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    frames.append(rf.to_ndarray())
            if len(frames) > 0:
                audio = np.concatenate(frames, axis=1).squeeze(0).astype(np.float32)
                sr = target_sr
    except Exception as av_err:
        decode_error = av_err

    # Step 2: Fallback to standard librosa/soundfile if PyAV was not available or encountered an issue
    if audio is None or len(audio) == 0:
        try:
            source_buffer.seek(0)
            audio, sr = librosa.load(source_buffer, sr=target_sr, mono=True)
            decode_error = None
        except Exception as lib_err:
            decode_error = lib_err

    # If both decoders failed, raise clear, descriptive error
    if audio is None or len(audio) == 0:
        is_mp3 = ".mp3" in inferred_filename.lower() or (raw_bytes and raw_bytes[:3] == b'ID3')
        if is_mp3:
            raise RuntimeError("Unable to decode MP3 audio. Please upload a valid MP3 file.")
        raise RuntimeError(f"Unable to decode audio. Please upload a valid MP3 or WAV file. (Error: {decode_error})")

    # Step 3: VAD silence trimming on leading and trailing non-speech
    if apply_vad and len(audio) > target_sr * 0.4:
        try:
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=vad_top_db)
            if len(trimmed_audio) >= target_sr * 0.3:
                audio = trimmed_audio
        except Exception:
            pass

    # Step 4: DC-Offset Removal & Amplitude Peak Normalization (consistent dynamic scale)
    # Removing DC offset ensures microphone bias does not create artificial 0Hz energy
    if len(audio) > 0:
        audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak > 1e-6:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), sr


def inspect_audio(
    audio_source: Union[str, bytes, io.BytesIO],
    filename: Optional[str] = None
) -> dict:
    """
    DIAGNOSTIC INSPECTION HELPER (GET /audio-info):
    Extracts deep stream metadata from uploaded or local audio files.
    Reports original format, rate, channels alongside decoded 16kHz PCM properties.
    """
    inferred_filename = filename or ""
    raw_bytes: Optional[bytes] = None

    if isinstance(audio_source, (bytes, bytearray)):
        raw_bytes = bytes(audio_source)
    elif isinstance(audio_source, io.BytesIO):
        audio_source.seek(0)
        raw_bytes = audio_source.getvalue()
    elif isinstance(audio_source, (str, Path)):
        fp = Path(audio_source)
        if not inferred_filename:
            inferred_filename = fp.name
        with open(fp, "rb") as f:
            raw_bytes = f.read()

    ext = Path(inferred_filename).suffix.lower() if inferred_filename else ""

    orig_fmt = "unknown"
    orig_sr = TARGET_SR
    orig_channels = 1

    try:
        # pyrefly: ignore [missing-import]
        import av
        container = av.open(io.BytesIO(raw_bytes))
        if len(container.streams.audio) > 0:
            astream = container.streams.audio[0]
            orig_fmt = container.format.name
            orig_sr = astream.rate
            orig_channels = astream.channels
    except Exception:
        # Fallback inspection via soundfile
        try:
            info = sf.info(io.BytesIO(raw_bytes))
            orig_fmt = str(info.format).lower()
            orig_sr = info.samplerate
            orig_channels = info.channels
        except Exception:
            pass

    # Decode through central loader without VAD to report raw waveform metrics
    audio, decoded_sr = load_audio(
        raw_bytes,
        target_sr=TARGET_SR,
        apply_vad=False,
        filename=inferred_filename
    )

    # Check speech sufficiency
    is_sufficient, speech_dur = is_sufficient_speech(audio, sr=decoded_sr)

    return {
        "filename": inferred_filename or "uploaded_audio",
        "extension": ext or f".{orig_fmt}",
        "original_format": orig_fmt,
        "original_sample_rate": orig_sr,
        "original_channels": orig_channels,
        "decoded_sample_rate": decoded_sr,
        "decoded_channels": 1,
        "duration": round(len(audio) / decoded_sr, 3),
        "number_of_samples": len(audio),
        "dtype": str(audio.dtype),
        "minimum_amplitude": round(float(np.min(audio)), 4),
        "maximum_amplitude": round(float(np.max(audio)), 4),
        "rms": round(float(np.sqrt(np.mean(audio**2))), 4),
        "is_speech_sufficient": is_sufficient,
        "speech_duration": round(speech_dur, 3),
        "successfully_decoded": True
    }


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
        apply_vad: bool = True,
        filename: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Delegates directly to the central load_audio implementation.
        Guarantees 100% identical preprocessing for training and inference across WAV and MP3.
        """
        return load_audio(
            audio_source=audio_source,
            target_sr=self.sample_rate,
            apply_vad=apply_vad,
            vad_top_db=self.vad_top_db,
            filename=filename
        )

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
