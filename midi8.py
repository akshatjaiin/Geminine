import mido
from mido import MidiFile, MidiTrack, Message

# Define note to MIDI number conversion
def note_name_to_number(note):
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    if len(note) == 3:  # Case like C-4, D#4, etc.
        note_name = note[:2]
        octave = int(note[2])
    else:  # Case like C4, D3, etc.
        note_name = note[0]
        octave = int(note[1])
    
    note_base = note_map[note_name[0]]
    
    if len(note_name) > 1:
        if note_name[1] == '-':
            note_base -= 1  # Flat
        elif note_name[1] == '#':
            note_base += 1  # Sharp
    
    return 12 * (octave + 1) + note_base

# Parse measure line
def parse_measure(measure_line):
    parts = measure_line.split()
    track_index = int(parts[0][1:])
    start_time = int(parts[1])
    channel = int(parts[2])
    measure_index = int(parts[3])
    note_duration = int(parts[4])
    
    notes = []
    for i in range(5, len(parts), 2):
        note = parts[i]
        if note == '0':
            # No note, just a rest
            notes.append((None, note_duration))
        else:
            note_name = note.split('-')[0]
            octave = note.split('-')[1]
            midi_note = note_name_to_number(note_name + octave)
            notes.append((midi_note, note_duration))
    
    return track_index, channel, notes, note_duration

# Create a MIDI file and add tracks
mid = MidiFile()
tracks = [MidiTrack() for _ in range(3)]
for track in tracks:
    mid.tracks.append(track)

# Text data
text_data = """
T0 0 4
T1 0 4
T2 0 4

M0 0 0 0 4 E-4 4 E-4 4 E-4 4 C-4 4 D-4 4 E-4 4 
M0 0 0 0 4 G-4 4 G-4 4 G-4 4 E-4 4 F-4 4 G-4 4 
M0 0 0 0 4 C-5 4 C-5 4 C-5 4 A-4 4 G-4 4 A-4 4 
M0 0 0 0 4 F-4 4 F-4 4 F-4 4 E-4 4 D-4 4 C-4 4

M1 0 0 0 4 C-3 4 C-3 4 C-3 4 C-3 4 C-3 4 C-3 4
M1 0 0 0 4 C-3 4 C-3 4 C-3 4 C-3 4 C-3 4 C-3 4
M1 0 0 0 4 G-2 4 G-2 4 G-2 4 G-2 4 G-2 4 G-2 4 
M1 0 0 0 4 G-2 4 G-2 4 G-2 4 G-2 4 G-2 4 G-2 4

M2 0 0 0 4 C-4 4 0 4 0 4 0 4 0 4 
M2 0 0 0 4 0 4 C-4 4 0 4 0 4 0 4 
M2 0 0 0 4 0 4 0 4 C-4 4 0 4 0 4 0 4
M2 0 0 0 4 0 4 0 4 0 4 C-4 4 0 4 0 4
"""

# Parse the text data and add notes to the MIDI tracks
for line in text_data.splitlines():
    line = line.strip()
    if line.startswith('M'):
        track_index, channel, notes, duration = parse_measure(line)
        for note, length in notes:
            if note is None:
                # Add a rest (note_off with zero velocity)
                tracks[track_index].append(Message('note_off', note=0, velocity=0, time=length * 480, channel=channel))
            else:
                # Add a note
                tracks[track_index].append(Message('note_on', note=note, velocity=64, time=0, channel=channel))
                tracks[track_index].append(Message('note_off', note=note, velocity=64, time=length * 480, channel=channel))

# Save the MIDI file
mid.save('super_mario_music.mid')
