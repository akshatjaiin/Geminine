import numpy as np
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from pydub.generators import Sine

# Function to generate a square wave sound
def generate_square_wave(frequency, duration_ms):
    sample_rate = 44100
    duration_s = duration_ms / 1000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    waveform = 0.5 * (1 + np.sign(np.sin(2 * np.pi * frequency * t)))
    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM
    wavfile.write("square_wave.wav", sample_rate, waveform)
    return AudioSegment.from_wav("square_wave.wav")

# Function to create a note
def create_note(frequency, duration):
    note_sound = generate_square_wave(frequency, duration)
    return note_sound

# Function to convert note name to frequency
def note_name_to_frequency(note):
    # Split the note name and accidental (if any)
    accidental = note[-2:] if len(note) > 2 else note[-1]
    base_note = note[:-2] if len(note) > 2 else note[:-1]
    
    # Note map without accidentals
    note_map = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00, 
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }

    # Base frequency of the note
    base_freq = note_map[base_note]

    # Calculate frequency with accidentals
    if accidental == '-':
        base_freq *= 2 ** (-1 / 12)  # Flatten the note
    elif accidental == '#':
        base_freq *= 2 ** (1 / 12)   # Sharp the note

    # Determine the octave adjustment
    octave = int(note[-1])
    frequency = base_freq * (2 ** (octave - 4))

    return frequency


# Define note durations (in milliseconds)
note_duration = 250  # Adjust as needed
# Example melody
melody_notes = [
    "E-4", "E-4", "E-4", "C-4", "D-4", "E-4",
    "G-4", "G-4", "G-4", "E-4", "F-4", "G-4",
    "C-5", "C-5", "C-5", "A-4", "G-4", "A-4",
    "F-4", "F-4", "F-4", "E-4", "D-4", "C-4"
]

# Create melody audio segment
melody_track = AudioSegment.empty()
for note in melody_notes:
    frequency = note_name_to_frequency(note)
    note_sound = create_note(frequency, note_duration)
    melody_track += note_sound

# Example bass and drum tracks (placeholder for demonstration)
# You need to create or load bass_track and drum_track in a similar way
bass_track = AudioSegment.empty()  # Replace with actual bass track code
drum_track = AudioSegment.empty()  # Replace with actual drum track code

# Combine all tracks
final_track = melody_track.overlay(bass_track).overlay(drum_track)

# Save the audio
final_track.export("super_mario_theme.wav", format="wav")
