from pydub import AudioSegment

# Load samples 
melody_sample = AudioSegment.from_wav("mario_melody_sample.wav")
bass_sample = AudioSegment.from_wav("mario_bass_sample.wav")
kick_sample = AudioSegment.from_wav("mario_kick_sample.wav")
snare_sample = AudioSegment.from_wav("mario_snare_sample.wav")

# Function to create a note from a sample (using pitch shifting)
def create_note(sample, frequency, duration_ms):
    # Calculate the pitch shift (semitones) based on the desired frequency
    base_frequency = 440  # Reference frequency (A4)
    semitones = 12 * np.log2(frequency / base_frequency)
    return sample.set_frame_rate(frequency).set_duration(duration_ms).pitch_shift(semitones)

# Define note durations (in milliseconds)
note_duration = 250 
tempo = 140  # Super Mario Bros. tempo (approximately)

# Function to create a drum hit
def create_drum_hit(sample, time_ms):
    return sample.set_duration(time_ms)  # Make sure the drum hit is the right duration 

# Create the melody
melody_notes = [
    "E-4", "E-4", "E-4", "C-4", "D-4", "E-4", 
    "G-4", "G-4", "G-4", "E-4", "F-4", "G-4", 
    "C-5", "C-5", "C-5", "A-4", "G-4", "A-4",
    "F-4", "F-4", "F-4", "E-4", "D-4", "C-4"
]

melody_track = AudioSegment.empty()
current_time = 0
for note in melody_notes:
    frequency = note_name_to_frequency(note)  # Function from your previous code
    note_sound = create_note(melody_sample, frequency, note_duration)
    melody_track += note_sound
    current_time += note_duration

# Create the bassline (adjust timing based on your bass sample)
bass_track = AudioSegment.empty()
current_time = 0
for i in range(16):
    bass_sound = create_note(bass_sample, note_name_to_frequency("C-3"), 500)  # Adjust timing as needed
    bass_track += bass_sound
    current_time += 500

# Create the drum track (adjust timing based on your drum samples)
drum_track = AudioSegment.empty()
current_time = 0
for i in range(16):
    drum_track += create_drum_hit(kick_sample, 250)  # Kick on the first beat
    current_time += 250
    drum_track += create_drum_hit(snare_sample, 250)  # Snare on the third beat
    current_time += 250

# Combine the tracks
final_track = melody_track.overlay(bass_track).overlay(drum_track)

# Add effects (adjust parameters as needed)
final_track = final_track.low_pass_filter(4000)  # Add a low-pass filter for a chiptune feel
final_track = final_track.apply_gain(-6)  # Reduce volume slightly

# Save the audio
final_track.export("super_mario_theme.wav", format="wav")