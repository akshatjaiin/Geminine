import pretty_midi
import pygame
import time
import io

# Initialize pygame.midi
pygame.midi.init()

# Find the default MIDI output
midi_out_id = pygame.midi.get_default_output_id()
midi_out = pygame.midi.Output(midi_out_id)

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

# Parse the saved MIDI file
parsed_midi = pretty_midi.PrettyMIDI(midi_data)

# Play the notes using pygame.midi
for instrument in parsed_midi.instruments:
    for note in instrument.notes:
        midi_out.note_on(note.pitch, note.velocity)
        time.sleep(note.end - note.start)
        midi_out.note_off(note.pitch, note.velocity)

# Close the MIDI output
midi_out.close()

# Quit pygame.midi
pygame.midi.quit()

print("Playback finished")
