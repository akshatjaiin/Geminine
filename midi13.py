import random
import numpy as np
from pydub import AudioSegment

def generate_random_drum_sample(duration=1000, sample_rate=44100):
    # Generate random samples for a single channel
    num_samples = int(duration * sample_rate / 1000)
    samples = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)
    
    # Create an AudioSegment from the samples
    audio = AudioSegment(
        samples.tobytes(),  # Convert the NumPy array to bytes
        frame_rate=sample_rate,
        sample_width=2,  # 2 bytes for int16
        channels=1  # Mono
    )
    
    return audio

# Generate a 1-second random drum sample
random_sample = generate_random_drum_sample()

# Export the sample to a WAV file
random_sample.export("random_drum_sample.wav", format="wav")
print("Random drum sample generated and saved as 'random_drum_sample.wav'.")
