import pretty_midi
from midi2audio import FluidSynth
import pydub
import io
import os
import tempfile

# Create a PrettyMIDI object and an instrument
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # 0 is the program number for Acoustic Grand Piano

# Define the notes for a C major scale
notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C, D, E, F, G, A, B, C

# Add notes to the instrument
for note_number in notes:
    note = pretty_midi.Note(
        velocity=100, pitch=note_number, start=0.5 * notes.index(note_number), end=0.5 * (notes.index(note_number) + 1)
    )
    instrument.notes.append(note)

# Add the instrument to the PrettyMIDI object
midi.instruments.append(instrument)

# Save the MIDI data to an in-memory file-like object
midi_data = io.BytesIO()
midi.write(midi_data)
midi_data.seek(0)

# Create a temporary file for the MIDI data
with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as midi_file:
    midi_file.write(midi_data.read())
    midi_file.close()
    midi_filename = midi_file.name

# Convert MIDI to audio using FluidSynth
soundfont_path = 'path/to/your/soundfont.sf2'  # Update this path
fluid_synth = FluidSynth(soundfont_path)
audio_path = midi_filename.replace('.mid', '.wav')
fluid_synth.midi_to_audio(midi_filename, audio_path)

# Load and play the audio using pydub
audio = pydub.AudioSegment.from_wav(audio_path)
playback = pydub.playback.play(audio)

# Wait for the playback to finish
playback.wait_done()

# Clean up temporary files
os.remove(midi_filename)
os.remove(audio_path)

print("Playback finished")
