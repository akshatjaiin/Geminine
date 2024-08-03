import numpy as np
import os
from instruments import Instrument
def test_instrument_functions():
    # Create an instance of the Instrument class
    instrument = Instrument()

    # Test parameters
    duration = 1.0
    sample_rate = 44100

    # Generate and record a melody
    notes = ['C4', 'E4', 'G4', 'C5']
    melody = instrument.generate_melody(notes=notes, duration=2, wave_type='sine', sample_rate=sample_rate)

    # Record melody
    instrument.record_key(60, duration, notes)  # Middle C
    instrument.record_chord([60, 64, 67], duration)  # C Major chord
    instrument.record_drum(np.random.randn(44100), duration)  # Random drum sample
    instrument.record_flute(440, duration)  # A4 flute sound

    # Apply effects
    melody_with_echo = instrument.apply_echo(melody, delay=0.1, decay=0.5, sample_rate=sample_rate)
    melody_with_reverb = instrument.apply_reverb(melody, reverb_amount=0.5, sample_rate=sample_rate)
    melody_with_gain = instrument.apply_gain(melody, gain=2.0, sample_rate=sample_rate)

    # Combine effects
    combined_waveform = instrument.combine_tracks(melody_with_echo, melody_with_reverb)

    # Add silence
    waveform_with_silence = instrument.add_silence(combined_waveform, duration=1.0, sample_rate=sample_rate)

    # Save to WAV
    output_path = 'combined_output.wav'
    instrument.to_wav(output_path)
    
    print(f"Combined output saved to {output_path}")

if __name__ == "__main__":
    test_instrument_functions()
