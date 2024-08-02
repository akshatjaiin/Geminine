import librosa
import numpy as np

def create_note(sample_path, frequency, duration_ms):
    y, sr = librosa.load(sample_path, sr=None)
    y = librosa.effects.time_stretch(y, duration_ms / 1000)  # Adjust duration
    y = librosa.effects.pitch_shift(y, sr, n_steps=12 * np.log2(frequency / 440))  # Pitch shift
    return AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1
    )

from pydub import AudioSegment
import numpy as np
import librosa

# Load samples
melody_sample = "path/to/mario_melody_sample.wav"
bass_sample = "path/to/mario_bass_sample.wav"
kick_sample = "path/to/mario_kick_sample.wav"
snare_sample = "path/to/mario_snare_sample.wav"

def note_name_to_frequency(note):
    # Note to frequency mapping (assuming A4 = 440 Hz)
    note_map = {
        'C': 261.63,  # C4
        'C#': 277.18, # C#4 / Db4
        'D': 293.66,  # D4
        'D#': 311.13, # D#4 / Eb4
        'E': 329.63,  # E4
        'F': 349.23,  # F4
        'F#': 369.99, # F#4 / Gb4
        'G': 392.00,  # G4
        'G#': 415.30, # G#4 / Ab4
        'A': 440.00,  # A4
        'A#': 466.16, # A#4 / Bb4
        'B': 493.88   # B4
    }
    
    note_name = note[:-2]  # Note name (e.g., 'E', 'F#')
    octave = int(note[-1])  # Octave number (e.g., 4)

    if len(note_name) == 2:  # If note name includes sharp (e.g., 'C#')
        note_name = note_name[:2]
    
    # Frequency calculation: base frequency for A4 is 440 Hz
    base_frequency = note_map[note_name]
    frequency = base_frequency * (2 ** (octave - 4))  # Adjust based on octave

    return frequency

# Function to create a note from a sample (using pitch shifting)
def create_note(sample_path, frequency, duration_ms):
    y, sr = librosa.load(sample_path, sr=None)
    y = librosa.effects.time_stretch(y, duration_ms / 1000)  # Adjust duration
    y = librosa.effects.pitch_shift(y, sr, n_steps=12 * np.log2(frequency / 440))  # Pitch shift
    return AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1
    )

# Define note durations (in milliseconds)
note_duration = 250
tempo = 140  # Super Mario Bros. tempo (approximately)

# Function to create a drum hit
def create_drum_hit(sample_path, duration_ms):
    y, sr = librosa.load(sample_path, sr=None)
    y = librosa.effects.time_stretch(y, duration_ms / 1000)  # Adjust duration
    return AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1
    )

# Create the melody
melody_notes = [
    "E-4", "E-4", "E-4", "C-4", "D-4", "E-4",
    "G-4", "G-4", "G-4", "E-4", "F-4", "G-4",
    "C-5", "C-5", "C-5", "A-4", "G-4", "A-4",
    "F-4", "F-4", "F-4", "E-4", "D-4", "C-4"
]

melody_track = AudioSegment.empty()
for note in melody_notes:
    frequency = note_name_to_frequency(note)  # Function to convert note names to frequency
    note_sound = create_note(melody_sample, frequency, note_duration)
    melody_track += note_sound

# Create the bassline (adjust timing based on your bass sample)
bass_track = AudioSegment.empty()
for _ in range(16):
    bass_sound = create_note(bass_sample, note_name_to_frequency("C-3"), 500)  # Adjust timing as needed
    bass_track += bass_sound

# Create the drum track (adjust timing based on your drum samples)
drum_track = AudioSegment.empty()
for _ in range(16):
    drum_track += create_drum_hit(kick_sample, 250)  # Kick on the first beat
    drum_track += create_drum_hit(snare_sample, 250)  # Snare on the third beat

# Combine the tracks
final_track = melody_track.overlay(bass_track).overlay(drum_track)

# Add effects (adjust parameters as needed)
final_track = final_track.low_pass_filter(4000)  # Add a low-pass filter for a chiptune feel
final_track = final_track.apply_gain(-6)  # Reduce volume slightly

# Save the audio
final_track.export("super_mario_theme.wav", format="wav")
