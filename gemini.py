import os
import google.generativeai as genai
from dotenv import load_dotenv
from instruments import Instrument
import requests
import json

load_dotenv() 
api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=api_key)



function_dict = {
    "generate_melody": {
        "function": Instrument().generate_melody,
        "args": ["notes", "duration", "wave_type", "sample_rate"],
    },
    "apply_reverb": {
        "function": Instrument().apply_reverb,
        "args": ["audio", "reverb_amount", "sample_rate"],
    },
    "generate_bassline": {
        "function": Instrument().generate_bassline,
        "args": ["notes", "duration", "wave_type", "sample_rate"],
    },
    "apply_gain": {
        "function": Instrument().apply_gain,
        "args": ["waveform", "gain", "sample_rate"],
    },
    "apply_low_pass_filter": {
        "function": Instrument().apply_low_pass_filter,
        "args": ["waveform", "cutoff_freq", "sample_rate"],
    },
    "note_to_frequency": {
        "function": Instrument().note_to_frequency,
        "args": ["note"],
    },
    "create_note_wave": {
        "function": Instrument().create_note_wave,
        "args": ["note", "duration", "wave_type", "sample_rate"],
    },
    "record_key": {
        "function": Instrument().record_key,
        "args": ["key", "duration", "notes"],
    },
    "record_chord": {
        "function": Instrument().record_chord,
        "args": ["chords", "duration"],
    },
    "record_drum": {
        "function": Instrument().record_drum,
        "args": ["drum_sample", "duration"],
    },
    "record_flute": {
        "function": Instrument().record_flute,
        "args": ["frequency", "duration"],
    },
    "play": {
        "function": Instrument().play,
        "args": [],
    },
    "close": {
        "function": Instrument().close,
        "args": [],
    },
    "clear_sample": {
        "function": Instrument().clear_sample,
        "args": [],
    },
    "to_wav": {
        "function": Instrument().to_wav,
        "args": ["path"],
    },
    "create_percussion_sound": {
        "function": Instrument().create_percussion_sound,
        "args": ["sample", "duration", "sample_rate"],
    },
    "generate_drum_pattern": {
        "function": Instrument().generate_drum_pattern,
        "args": ["pattern", "duration", "sample_rate"],
    },
    "apply_echo": {
        "function": Instrument().apply_echo,
        "args": ["waveform", "delay", "decay", "sample_rate"],
    },
    "pitch_shift": {
        "function": Instrument().pitch_shift,
        "args": ["waveform", "semitones", "sample_rate"],
    },
    "time_stretch": {
        "function": Instrument().time_stretch,
        "args": ["waveform", "stretch_factor", "sample_rate"],
    },
    "numpy_to_pydub": {
        "function": Instrument().numpy_to_pydub,
        "args": ["waveform", "sample_rate"],
    },
    "combine_tracks": {
        "function": Instrument().combine_tracks,
        "args": ["track1", "track2"],
    },
    "add_silence": {
        "function": Instrument().add_silence,
        "args": ["waveform", "duration", "sample_rate"],
    },
    "mix_tracks_pydub": {
        "function": Instrument().mix_tracks_pydub,
        "args": ["tracks"],
    },
}


prompt = "Create  a melody for mario bros in 5 second"

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(
    prompt,
    tools=[{
        'function_declarations': [function_dict],
    }],
)

function_call = response.candidates[0].content.parts[0].function_call
args = function_call.args
function_name = function_call.name




response = model.generate_content(
    "Based on this information i created a file respond to the student in a friendly manner.",
)
print(response.text)