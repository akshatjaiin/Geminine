from PyMusic_Instrument import Instrument

# Create an instrument instance
piano = Instrument("piano")

# Generate a note (assuming the library has similar methods)
note = piano.play("C4", duration=500)  # Play C4 note for 500 ms

# Save or play the note
note.export("piano_note.wav", format="wav")
