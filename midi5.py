from pydub import AudioSegment
import numpy as np
import scipy.io.wavfile as wav

def generate_square_wave(duration_ms, frequency, sample_rate):
    """Generate a square wave signal.

    Args:
        duration_ms (int): Duration of the wave in milliseconds.
        frequency (int): Frequency of the square wave in Hz.
        sample_rate (int): Sample rate in Hz.

    Returns:
        np.array: Array containing the square wave samples.
    """
    duration_s = duration_ms / 1000.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave = 0.5 * (1 + np.sign(np.sin(2 * np.pi * frequency * t)))
    return (wave * 32767).astype(np.int16)

# Parameters
duration_ms = 1000  # milliseconds
frequency = 440     # Hz
sample_rate = 44100  # Hz

# Generate square wave
square_wave_data = generate_square_wave(duration_ms, frequency, sample_rate)

# Save square wave to WAV file
wav.write("square_wave_sample.wav", sample_rate, square_wave_data)

# Load the generated WAV file using pydub
square_wave = AudioSegment.from_wav("square_wave_sample.wav")

# Export the square wave to another WAV file
square_wave.export("output.wav", format="wav")
