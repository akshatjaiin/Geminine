from midi2audio import FluidSynth
from pydub import AudioSegment

def convert_midi_to_mp3(midi_file, output_mp3_file):
    """Converts a MIDI file to MP3.

    Args:
        midi_file (str): Path to the MIDI file.
        output_mp3_file (str): Path to the output MP3 file.
    """

    # Use FluidSynth to render the MIDI file to a WAV file
    fs = FluidSynth()
    fs.midi_to_audio(midi_file, output_mp3_file.replace('.mp3', '.wav'))

    # Convert the WAV file to MP3 using pydub
    sound = AudioSegment.from_wav(output_mp3_file.replace('.mp3', '.wav'))
    sound.export(output_mp3_file, format="mp3")

# Example usage:
midi_file_path = "your_midi_file.mid"
output_mp3_file_path = "your_output_file.mp3"

convert_midi_to_mp3(midi_file_path, output_mp3_file_path)