"""
Audio Processing Service
Handles audio preprocessing and feature extraction for audio analysis.
"""
import io
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from typing import Dict, List, Tuple, Optional
from scipy import signal
from pydub import AudioSegment


def convert_audio_format(audio_data: bytes, input_format: str = "wav") -> Tuple[np.ndarray, int]:
    """
    Convert audio bytes to numpy array with sample rate.
    
    Args:
        audio_data: Raw audio bytes
        input_format: Input audio format (wav, mp3, webm, etc.)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio_io = io.BytesIO(audio_data)
    
    # For WebM and other formats that librosa might struggle with,
    # use pydub to convert to WAV first, then load with librosa
    if input_format.lower() in ["webm", "ogg", "m4a", "mp3"]:
        try:
            # Use pydub to convert to WAV format
            audio_segment = AudioSegment.from_file(audio_io, format=input_format.lower())
            # Export to WAV bytes
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            # Now load with librosa
            y, sr = librosa.load(wav_io, sr=None)
        except Exception as e:
            # Fallback: try librosa directly (requires ffmpeg)
            audio_io.seek(0)
            try:
                y, sr = librosa.load(audio_io, sr=None)
            except Exception as e2:
                raise ValueError(f"Failed to load audio format '{input_format}': {str(e)}. Original error: {str(e2)}")
    else:
        # For WAV and other formats, use librosa directly
        try:
            y, sr = librosa.load(audio_io, sr=None)
        except Exception as e:
            # If direct load fails, try with pydub as fallback
            try:
                audio_io.seek(0)
                audio_segment = AudioSegment.from_file(audio_io, format=input_format.lower())
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                y, sr = librosa.load(wav_io, sr=None)
            except Exception as e2:
                raise ValueError(f"Failed to load audio: {str(e)}. Fallback error: {str(e2)}")
    
    return y, sr


def remove_noise(audio: np.ndarray, sample_rate: int, stationary: bool = False) -> np.ndarray:
    """
    Remove noise from audio signal.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of the audio
        stationary: Whether noise is stationary (True) or non-stationary (False)
    
    Returns:
        Denoised audio array
    """
    # Use noisereduce library for noise removal
    reduced_noise = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=stationary,
        prop_decrease=0.8  # Reduce noise by 80%
    )
    
    return reduced_noise


def segment_audio(audio: np.ndarray, sample_rate: int, segment_length_seconds: float = 1.0) -> List[np.ndarray]:
    """
    Segment audio into smaller chunks.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of the audio
        segment_length_seconds: Length of each segment in seconds
    
    Returns:
        List of audio segments
    """
    segment_length_samples = int(segment_length_seconds * sample_rate)
    segments = []
    
    for i in range(0, len(audio), segment_length_samples):
        segment = audio[i:i + segment_length_samples]
        if len(segment) > 0:
            segments.append(segment)
    
    return segments


def normalize_audio(audio: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalize audio signal.
    
    Args:
        audio: Audio signal array
        method: Normalization method ('peak' or 'rms')
    
    Returns:
        Normalized audio array
    """
    if method == "peak":
        # Peak normalization: scale to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    elif method == "rms":
        # RMS normalization: normalize to target RMS level
        target_rms = 0.1
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            return audio * (target_rms / rms)
        return audio
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_audio(
    audio_data: bytes,
    input_format: str = "wav",
    remove_noise_flag: bool = True,
    normalize_flag: bool = True,
    segment_flag: bool = False,
    segment_length_seconds: float = 1.0
) -> Dict:
    """
    Complete audio preprocessing pipeline.
    
    Args:
        audio_data: Raw audio bytes
        input_format: Input audio format
        remove_noise_flag: Whether to remove noise
        normalize_flag: Whether to normalize audio
        segment_flag: Whether to segment audio
        segment_length_seconds: Length of segments if segmenting
    
    Returns:
        Dictionary containing processed audio and metadata
    """
    # Step 1: Format Conversion
    audio, sample_rate = convert_audio_format(audio_data, input_format)
    duration = len(audio) / sample_rate
    
    # Step 2: Noise Removal
    if remove_noise_flag:
        audio = remove_noise(audio, sample_rate)
    
    # Step 3: Segmentation (optional)
    segments = None
    if segment_flag:
        segments = segment_audio(audio, sample_rate, segment_length_seconds)
    
    # Step 4: Normalization
    if normalize_flag:
        audio = normalize_audio(audio, method="peak")
    
    # Convert processed audio back to bytes (WAV format)
    audio_io = io.BytesIO()
    sf.write(audio_io, audio, sample_rate, format='WAV')
    processed_audio_bytes = audio_io.getvalue()
    
    result = {
        "processed_audio": processed_audio_bytes,
        "sample_rate": int(sample_rate),
        "duration": float(duration),
        "num_samples": int(len(audio)),
        "segmented": segment_flag,
        "num_segments": len(segments) if segments else 1
    }
    
    return result


def extract_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> Dict:
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients to extract
    
    Returns:
        Dictionary containing MFCC features and statistics
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    
    return {
        "mfcc_coefficients": mfccs.tolist(),
        "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
        "mfcc_std": np.std(mfccs, axis=1).tolist(),
        "num_coefficients": n_mfcc,
        "num_frames": int(mfccs.shape[1])
    }


def generate_spectrogram(audio: np.ndarray, sample_rate: int) -> Dict:
    """
    Generate spectrogram from audio signal.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of the audio
    
    Returns:
        Dictionary containing spectrogram data
    """
    # Compute short-time Fourier transform (STFT)
    stft = librosa.stft(audio)
    spectrogram = np.abs(stft)
    
    # Convert to decibels
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Get frequency bins
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    
    # Get time frames
    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate)
    
    return {
        "spectrogram": spectrogram_db.tolist(),
        "frequencies": frequencies.tolist(),
        "times": times.tolist(),
        "magnitude_mean": float(np.mean(spectrogram)),
        "magnitude_max": float(np.max(spectrogram)),
        "magnitude_min": float(np.min(spectrogram))
    }


def analyze_pitch_and_frequency(audio: np.ndarray, sample_rate: int) -> Dict:
    """
    Analyze pitch and frequency characteristics.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of the audio
    
    Returns:
        Dictionary containing pitch and frequency analysis
    """
    # Extract pitch using librosa's pitch tracking
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
    
    # Get fundamental frequency (F0)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    pitch_values = np.array(pitch_values)
    
    # Calculate dominant frequency
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    magnitude = np.abs(fft)
    
    # Find dominant frequency (excluding DC component)
    positive_freq_idx = np.where(freqs > 0)[0]
    dominant_freq_idx = positive_freq_idx[np.argmax(magnitude[positive_freq_idx])]
    dominant_frequency = freqs[dominant_freq_idx]
    
    # Calculate spectral centroid (brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    
    # Calculate zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    
    result = {
        "pitch_mean": float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0,
        "pitch_std": float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0,
        "pitch_min": float(np.min(pitch_values)) if len(pitch_values) > 0 else 0.0,
        "pitch_max": float(np.max(pitch_values)) if len(pitch_values) > 0 else 0.0,
        "dominant_frequency": float(dominant_frequency),
        "spectral_centroid_mean": float(np.mean(spectral_centroids)),
        "zero_crossing_rate_mean": float(np.mean(zcr))
    }
    
    return result


def analyze_duration(audio: np.ndarray, sample_rate: int) -> Dict:
    """
    Analyze duration characteristics of audio.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of the audio
    
    Returns:
        Dictionary containing duration analysis
    """
    duration = len(audio) / sample_rate
    
    # Detect silence/non-silence regions
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Threshold for silence detection (adjustable)
    silence_threshold = np.percentile(rms, 10)
    is_silence = rms < silence_threshold
    
    # Calculate actual audio duration (excluding silence)
    non_silence_frames = np.sum(~is_silence)
    actual_duration = (non_silence_frames * hop_length) / sample_rate
    
    return {
        "total_duration_seconds": float(duration),
        "actual_audio_duration_seconds": float(actual_duration),
        "silence_duration_seconds": float(duration - actual_duration),
        "silence_percentage": float((duration - actual_duration) / duration * 100) if duration > 0 else 0.0,
        "num_samples": int(len(audio)),
        "sample_rate": int(sample_rate)
    }


def extract_features(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> Dict:
    """
    Complete feature extraction pipeline.
    
    Args:
        audio: Audio signal array
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        Dictionary containing all extracted features
    """
    features = {}
    
    # Extract MFCC
    features["mfcc"] = extract_mfcc(audio, sample_rate, n_mfcc)
    
    # Generate Spectrogram
    features["spectrogram"] = generate_spectrogram(audio, sample_rate)
    
    # Analyze Pitch and Frequency
    features["pitch_frequency"] = analyze_pitch_and_frequency(audio, sample_rate)
    
    # Analyze Duration
    features["duration"] = analyze_duration(audio, sample_rate)
    
    return features

