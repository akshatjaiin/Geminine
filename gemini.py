import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv() 
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

# generates waveforms
import numpy as np
import scipy.io.wavfile as wavfile

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

# apply effects
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

# create musical nodes
def note_to_frequency(note):
    # Simplified note-to-frequency conversion (A4=440Hz, A4 is the reference)
    notes = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
        'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
        'A#': 466.16, 'B': 493.88
    }
    base_note, octave = note[:-1], int(note[-1])
    frequency = notes[base_note] * (2 ** (octave - 4))
    return frequency

def create_note_wave(note, duration, sample_rate=44100):
    frequency = note_to_frequency(note)
    waveform = generate_sine_wave(frequency, duration, sample_rate)
    return waveform

# save to wav file
def save_wavefile(filename, waveform, sample_rate=44100):
    waveform = np.int16(waveform * 32767)  # Convert waveform to 16-bit PCM format
    wavfile.write(filename, sample_rate, waveform)

# generate music sequence
def generate_melody(notes, duration, sample_rate=44100):
    melody = np.concatenate([create_note_wave(note, duration, sample_rate) for note in notes])
    return melody

def generate_bassline(notes, duration, sample_rate=44100):
    bassline = np.concatenate([create_note_wave(note, duration, sample_rate) for note in notes])
    return bassline

# generate rhythm and precurursion
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

# combine tracks
def mix_tracks(*tracks):
    combined = np.zeros_like(tracks[0])
    for track in tracks:
        combined[:len(track)] += track
    return np.clip(combined, -1, 1)

tools = [generate_sine_wave, generate_square_wave, generate_triangle_wave, generate_sawtooth_wave, apply_gain, apply_low_pass_filter, note_to_frequency, create_note_wave, save_wavefile, generate_melody, generate_bassline, create_percussion_sound, generate_drum_pattern, mix_tracks]
system_instruction="you are a music engine thats generates the melodies according to the users requirement"

# Toggle this to switch between Gemini 1.5 with a system instruction, or Gemini 1.0 Pro.
use_sys_inst = False

model_name = 'gemini-1.5-flash' if use_sys_inst else 'gemini-1.0-pro'

if use_sys_inst:
  model = genai.GenerativeModel(
      model_name, tools= tools, system_instruction=system_instruction)
  convo = model.start_chat(enable_automatic_function_calling=True)

else:
  model = genai.GenerativeModel(model_name, tools=tools)
  convo = model.start_chat(
      history=[
          {'role': 'user', 'parts': [system_instruction]},
          {'role': 'model', 'parts': ['OK I understand. I will do my best!']}
        ],
      enable_automatic_function_calling=True)

running = 1
while running < 6:
  def send_message(message):
    return convo.send_message(message)
  running+=1

