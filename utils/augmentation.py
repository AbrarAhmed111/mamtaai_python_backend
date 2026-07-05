"""
Audio augmentation utilities for training data augmentation.

Augmentations simulate real-world microphone recording conditions so the model
learns to handle them (room reverb, background noise, mic coloring). Each function
takes a numpy float32 array + sample_rate and returns an augmented array of the
same length.
"""
import numpy as np
from scipy import signal as scipy_signal


def add_room_reverb(audio: np.ndarray, sample_rate: int, reverb_time: float = 0.25) -> np.ndarray:
    """
    Simulate room reverb by convolving with a synthetic room impulse response.
    reverb_time: RT60-like decay in seconds (0.15-0.4 for typical rooms).
    """
    impulse_len = int(reverb_time * sample_rate)
    t = np.linspace(0, reverb_time, impulse_len)
    # Exponentially decaying noise impulse response
    decay = np.exp(-6.9 * t / reverb_time)   # -60dB at reverb_time
    impulse = np.random.randn(impulse_len) * decay
    impulse /= np.max(np.abs(impulse) + 1e-9)
    reverbed = np.convolve(audio, impulse, mode='full')[:len(audio)]
    # Mix dry + wet (50/50 — strong enough to train on but not destroy the cry)
    mixed = 0.6 * audio + 0.4 * reverbed
    peak = np.max(np.abs(mixed))
    return mixed / peak if peak > 0 else mixed


def add_background_noise(audio: np.ndarray, snr_db: float = 18.0) -> np.ndarray:
    """
    Add Gaussian white noise at a given signal-to-noise ratio (dB).
    SNR 18-25 dB simulates a noisy room / phone mic pickup.
    """
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    noisy = audio + noise
    peak = np.max(np.abs(noisy))
    return noisy / peak if peak > 0 else noisy


def add_mic_coloring(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Simulate cheap microphone frequency response: slight low-frequency roll-off
    and a small peak around 1-3 kHz (phone mic characteristic).
    """
    nyquist = sample_rate / 2.0
    # High-pass at 150 Hz (cuts mic handling noise / low rumble)
    b_hp, a_hp = scipy_signal.butter(2, 150.0 / nyquist, btype='high')
    audio = scipy_signal.filtfilt(b_hp, a_hp, audio)
    # Gentle peak boost 1-3 kHz (phone mic resonance)
    low_peak = min(1000.0 / nyquist, 0.99)
    high_peak = min(3000.0 / nyquist, 0.99)
    if low_peak < high_peak:
        b_pk, a_pk = scipy_signal.butter(2, [low_peak, high_peak], btype='band')
        boosted = scipy_signal.filtfilt(b_pk, a_pk, audio)
        audio = audio + 0.3 * boosted
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def augment_sample(audio: np.ndarray, sample_rate: int, variant: int) -> np.ndarray:
    """
    Apply a deterministic augmentation variant to one audio sample.

    variant 0 = reverb + noise   (room recording simulation)
    variant 1 = noise + mic      (phone mic simulation)
    variant 2 = reverb + noise + mic (worst-case: phone speaker -> laptop mic)
    """
    rng_state = np.random.get_state()  # save so augmentation is reproducible
    np.random.seed(abs(hash(audio.tobytes())) % (2**31) + variant)

    if variant == 0:
        out = add_room_reverb(audio, sample_rate, reverb_time=0.20 + np.random.rand() * 0.15)
        out = add_background_noise(out, snr_db=20 + np.random.rand() * 8)
    elif variant == 1:
        out = add_background_noise(audio, snr_db=15 + np.random.rand() * 10)
        out = add_mic_coloring(out, sample_rate)
    else:
        out = add_room_reverb(audio, sample_rate, reverb_time=0.25 + np.random.rand() * 0.15)
        out = add_background_noise(out, snr_db=14 + np.random.rand() * 8)
        out = add_mic_coloring(out, sample_rate)

    np.random.set_state(rng_state)
    return out
