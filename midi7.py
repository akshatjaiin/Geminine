from pydub import AudioSegment
import numpy as np
import scipy.io.wavfile as wav
import os

# Define a function to generate a square wave signal
def generate_square_wave(frequency, duration_ms, sample_rate=44100):
    """Generate a square wave signal."""
    duration_s = duration_ms / 1000.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave = 0.5 * (1 + np.sign(np.sin(2 * np.pi * frequency * t)))
    return (wave * 32767).astype(np.int16)

# Define a function to convert note name to frequency
def note_to_frequency(note):
    """Convert a musical note to a frequency."""
    note_map = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    note_name = note[:-1]  # Note letter(s)
    octave = int(note[-1])  # Octave number
    frequency = note_map[note_name] * (2 ** (octave - 4))
    return frequency

# Function to create a note sound
def create_note(note, duration, sample_rate=44100):
    frequency = note_to_frequency(note)
    note_sound_data = generate_square_wave(frequency, duration, sample_rate)
    wav.write("temp_note.wav", sample_rate, note_sound_data)
    note_sound = AudioSegment.from_wav("temp_note.wav")
    return note_sound

# Define note durations and melody
note_duration = 300  # in milliseconds, adjust as needed for tempo
melody = [
    "E4", "E4", "E4", "C4", "D4", "E4",
    "G4", "G4", "G4", "E4", "F4", "G4",
    "C5", "C5", "C5", "A4", "G4", "A4",
    "F4", "F4", "F4", "E4", "D4", "C4"
]

# Combine the notes into a melody
melody_track = AudioSegment.empty()
for note in melody:
    note_sound = create_note(note, note_duration)
    melody_track += note_sound

# Export the melody
melody_track.export("super_mario_melody.wav", format="wav")

# Clean up temporary file
os.remove("temp_note.wav")
