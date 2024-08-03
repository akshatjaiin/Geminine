import numpy as np
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
from pydub.playback import play
import random

# --- Waveform Generators ---

def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    return waveform

def generate_square_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
    return waveform

def generate_triangle_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1)
    return waveform

def generate_sawtooth_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * (2 * (t * frequency - np.floor(t * frequency + 0.5)))
    return waveform

# --- Effects ---

def apply_gain(waveform, gain, sample_rate=44100):
    waveform = np.clip(waveform * (10 ** (gain / 20)), -1, 1)
    return waveform

def apply_low_pass_filter(waveform, cutoff_freq, sample_rate=44100):
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    filtered_waveform = lfilter(b, a, waveform)
    return filtered_waveform

# --- Note and Melody Generation ---

def note_to_frequency(note):
    notes = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
        'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
        'A#': 466.16, 'B': 493.88
    }
    base_note, octave = note[:-1], int(note[-1])
    frequency = notes[base_note] * (2 ** (octave - 4))
    return frequency

def create_note_wave(note, duration, wave_type='sine', sample_rate=44100):
    frequency = note_to_frequency(note)
    if wave_type == 'sine':
        waveform = generate_sine_wave(frequency, duration, sample_rate)
    elif wave_type == 'square':
        waveform = generate_square_wave(frequency, duration, sample_rate)
    elif wave_type == 'triangle':
        waveform = generate_triangle_wave(frequency, duration, sample_rate)
    elif wave_type == 'sawtooth':
        waveform = generate_sawtooth_wave(frequency, duration, sample_rate)
    return waveform

def save_wavefile(filename, waveform, sample_rate=44100):
    waveform = np.int16(waveform * 32767)
    wavfile.write(filename, sample_rate, waveform)

def generate_melody(notes, duration, wave_type='sine', sample_rate=44100):
    melody = np.concatenate([create_note_wave(note, duration, wave_type, sample_rate) for note in notes])
    return melody

def generate_bassline(notes, duration, wave_type='sine', sample_rate=44100):
    bassline = np.concatenate([create_note_wave(note, duration, wave_type, sample_rate) for note in notes])
    return bassline

# --- Percussion and Drum Patterns ---

def create_percussion_sound(sample, duration, sample_rate=44100):
    sample_waveform = generate_sine_wave(sample, duration, sample_rate)
    return sample_waveform

def generate_drum_pattern(pattern, duration, sample_rate=44100):
    drum_track = np.zeros(int(duration * sample_rate))
    for hit in pattern:
        start = int(hit[0] * sample_rate)
        end = start + int(hit[1] * sample_rate)
        drum_track[start:end] += create_percussion_sound(hit[2], hit[1], sample_rate)
    return drum_track

# --- Random Melody, Bassline, and Drum Pattern Generation ---

def generate_random_melody(length, note_duration, wave_type='sine', sample_rate=44100):
    possible_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
    melody_notes = random.choices(possible_notes, k=length)
    melody = generate_melody(melody_notes, note_duration, wave_type, sample_rate)
    return melody

def generate_random_bassline(length, note_duration, wave_type='square', sample_rate=44100):
    possible_notes = ['C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2', 'C3']
    bassline_notes = random.choices(possible_notes, k=length)
    bassline = generate_bassline(bassline_notes, note_duration, wave_type, sample_rate)
    return bassline

def generate_random_drum_pattern(length, note_duration, sample_rate=44100):
    drum_pattern = []
    for _ in range(length):
        hit_time = random.uniform(0, note_duration)
        hit_duration = random.uniform(0.05, 0.15)
        frequency = random.uniform(100, 300)
        drum_pattern.append((hit_time, hit_duration, frequency))
    drum_track = generate_drum_pattern(drum_pattern, length * note_duration, sample_rate)
    return drum_track

# --- Mixing and Saving ---

def mix_tracks_pydub(*tracks):
    combined = tracks[0]
    for track in tracks[1:]:
        combined = combined.overlay(track)
    return combined

def numpy_to_pydub(waveform, sample_rate=44100):
    waveform = np.int16(waveform * 32767)
    audio_segment = AudioSegment(
        waveform.tobytes(), 
        frame_rate=sample_rate,
        sample_width=waveform.dtype.itemsize, 
        channels=1
    )
    return audio_segment

# --- Generate a Random Song ---

melody_length = 16
note_duration = 0.5  # Duration of each note in seconds
sample_rate = 44100

# Generate random elements
random_melody = generate_random_melody(melody_length, note_duration, wave_type='sawtooth', sample_rate=sample_rate)
random_bassline = generate_random_bassline(melody_length, note_duration, wave_type='triangle', sample_rate=sample_rate)
random_drum_pattern = generate_random_drum_pattern(melody_length, note_duration, sample_rate=sample_rate)

# Convert numpy arrays to Pydub AudioSegment
melody_segment = numpy_to_pydub(random_melody, sample_rate)
bassline_segment = numpy_to_pydub(random_bassline, sample_rate)
drum_segment = numpy_to_pydub(random_drum_pattern, sample_rate)

# Mix tracks using Pydub
final_track = mix_tracks_pydub(melody_segment, bassline_segment, drum_segment)

# Apply effects using Pydub
final_track = final_track.low_pass_filter(3000).apply_gain(-6)

# Save the final track to a WAV file
final_track.export("random_song_pydub3.wav", format="wav")

print("Random song generated and saved")

# Optional: Play the final track (requires audio playback support)
# play(final_track)
