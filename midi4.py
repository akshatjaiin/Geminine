import mido
from mido import MidiFile, MidiTrack, Message

def note_name_to_number(note):
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    # Extract the base note and octave
    if len(note) > 2:  # e.g., C#4 or D-3
        note_name = note[:-1]  # Note and accidental
        octave = int(note[-1])  # Last character as octave
    else:  # e.g., C4 or D3
        note_name = note[0]
        octave = int(note[1])
    
    note_base = note_map[note_name[0]]
    
    if len(note_name) > 1:
        if note_name[1] == '-':
            note_base -= 1  # Flat
        elif note_name[1] == '#':
            note_base += 1  # Sharp
    
    return 12 * octave + note_base

def parse_measure(measure_line):
    parts = measure_line.split()
    track_index = int(parts[0][1:])
    start_time = int(parts[1])
    channel = int(parts[2])
    measure_index = int(parts[3])
    note_duration = int(parts[4])
    
    notes = []
    for note in parts[5:]:
        # Handle notes and durations
        if len(note) >= 2:
            if note[-1].isdigit():  # Check if the last character is a digit (octave)
                octave = note[-1]
                note_name = note[:-1]
            else:  # Notes with accidentals
                octave = note[-2:]
                note_name = note[:-2]
            
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
M1 0 0 0 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4
M2 0 0 0 4 C-4 4 C-4 4 C-4 4 C-4 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4 G-3 4
"""


# Parse the text data and add notes to the MIDI tracks
for line in text_data.splitlines():
    line = line.strip()
    if line.startswith('M'):
        track_index, channel, notes, duration = parse_measure(line)
        
        # Track time for each note
        current_time = 0
        for note, length in notes:
            # Note on
            tracks[track_index].append(Message('note_on', note=note, velocity=64, time=current_time, channel=channel))
            # Note off
            tracks[track_index].append(Message('note_off', note=note, velocity=64, time=duration * 480, channel=channel))  
            current_time += duration * 480  # Update current_time

# Save the MIDI file
mid.save('output2.mid')