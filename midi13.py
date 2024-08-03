import random
from pydub import AudioSegment

def generate_random_drum_sample(duration=1000, sample_rate=44100, num_channels=2):
    # Generate random samples for each channel
    samples = [
        [random.randint(-32768, 32767) for _ in range(int(duration * sample_rate / 1000))]
        for _ in range(num_channels)
    ]
    
    # Create an AudioSegment from the samples
    audio = AudioSegment(
        data=b"".join(sample.tobytes() for sample in samples),
        frame_rate=sample_rate,
        sample_width=2,
        channels=num_channels
    )
    
    return audio

# Generate a 1-second random drum sample
random_sample = generate_random_drum_sample()

# Export the sample to a WAV file
random_sample.export("random_drum_sample.wav", format="wav")
