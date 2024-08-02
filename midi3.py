from midi2audio import FluidSynth
from pydub import AudioSegment
import os

def convert_midi_to_mp3(midi_file, output_mp3_file):
    """Converts a MIDI file to MP3.

    Args:
        midi_file (str): Path to the MIDI file.
        output_mp3_file (str): Path to the output MP3 file.
    """

    # Use FluidSynth to render the MIDI file to a WAV file
    temp_wav_file = output_mp3_file.replace('.mp3', '.wav')
    fs = FluidSynth()
    fs.midi_to_audio(midi_file, temp_wav_file)

    # Convert the WAV file to MP3 using pydub
    sound = AudioSegment.from_wav(temp_wav_file)
    sound.export(output_mp3_file, format="mp3")

    # Remove the temporary WAV file
    os.remove(temp_wav_file)

# Example usage:
midi_file_path = "output.mid"
output_mp3_file_path = "your_output_file.mp3"

convert_midi_to_mp3(midi_file_path, output_mp3_file_path)
