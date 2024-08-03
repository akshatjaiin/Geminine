import os
import google.generativeai as genai
from dotenv import load_dotenv
from instruments import Instrument
load_dotenv() 
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


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
def generate_function_call(instruction: str):
    """
    Generates a Python function call from a natural language instruction.
    """
    payload = {
        "model": "text-davinci-003",
        "prompt": f"You are a function caller. Your job is to use functions in the Instrument library.\n You have access to these functions: {function_dict.keys()}\n Generate a Python function call based on this instruction: {instruction}",
        "max_tokens": 100,
        "temperature": 0.7,  
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    command = response.json()["choices"][0]["text"]  
    return command

def execute_function_call(function_call: str):
    """
    Safely executes a Python function call in Windows.
    """
    try:
        locals_dict = {"Instrument": Instrument, **function_dict}
        exec(function_call, globals(), locals_dict)
    except Exception as e:
        print(f"Error executing command: {e}")

# --- Example Usage ---

instruction = "Generate a melody using the generate_melody function. Use notes C4, D4, E4, and a sine wave. Make the duration of each note 0.5 seconds and save it to a file name gem.wav"

function_call = generate_function_call(instruction) 

print(f"Generated function call: {function_call}")

execute_function_call(function_call)
