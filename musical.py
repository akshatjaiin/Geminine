import numpy as np
import random
from instruments import Instrument

def generate_random_melody(length: int) -> list:
    """Generate a random melody."""
    melody = []
    for _ in range(length):
        note = random.choice(range(60, 72))  # Choose a random note between MIDI 60 (Middle C) and 71
        melody.append(note)
    return melody

def generate_random_chords() -> list:
    """Generate a list of random chords."""
    chords = [
        [60, 64, 67],  # C major
        [62, 65, 69],  # D minor
        [65, 69, 72],  # F major
        [67, 71, 74],  # G major
    ]
    return random.choices(chords, k=4)  # Pick 4 random chords

def create_random_song(instrument: Instrument, melody_length: int, melody_duration: float) -> None:
    """Create a random song using the provided instrument."""
    melody = generate_random_melody(melody_length)
    chords = generate_random_chords()
    
    # Add melody to the song
    for note in melody:
        instrument.record_key(note, melody_duration)
    
    # Add chords to the song
    for chord in chords:
        instrument.record_chord(chord, melody_duration)
    
    # Export the song
    sample_rate = 44100
    filename = 'rando1.mp3'
    Instrument.export_to_mp3(filename, instrument.sample, sample_rate)
    print(f"Song saved to {filename}")

# Example usage
if __name__ == "__main__":
    bit_rate = 44100
    instrument = Instrument(bit_rate)
    melody_length = 16  # Number of notes in the melody
    melody_duration = 0.5  # Duration of each note/chord in seconds
    
    create_random_song(instrument, melody_length, melody_duration)
