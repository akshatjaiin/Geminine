from collections.abc import Iterable
import struct
import sys
import numpy as np
import pyaudio
from scipy.io import wavfile
from pydub import AudioSegment
import random

class PlayerNotInitialisedError(AttributeError):
    """
    Test
    """
    pass


class Instrument:
    """
    Can play any piano notes, you have to input the key number for corresponding frequencies.

    :param bit_rate: Generally value of bit rate is 44100 or 48000, it is proportional to wavelength of frequency generated.
    :param no_play: Use this if you don't want to play the sample but use it for other purposes.

    Attributes:
        :graphing_sample: Used for graphing see the usage page.
        :total_time: Numpy array representing time at split seconds.
        :play_time: Total time for which the sample has been recorded.
    """
    def __init__(self, bit_rate: int = 44100, no_play: bool = False):
        self._BITRATE = bit_rate
        self.no_play = no_play

        if not self.no_play:
            self._player = pyaudio.PyAudio()

        self._sample = np.array([])
        self.graphing_sample = []

        self.total_time = np.array([])
        self.play_time = 0
        
    @property
    def sample(self) -> np.ndarray:
        return self._sample

    @sample.setter
    def sample(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Sample must be a numpy array")
        self._sample = value

    @staticmethod
    def get_hz(key_number: int) -> float:
        return 2 ** ((key_number - 49) / 12) * 440

    def record_key(self, key: int, duration: float) -> None:
        key_hz = self.get_hz(key)
        reciprocal_hz = 1 / key_hz
        phase_completer = reciprocal_hz * (round(key_hz * duration) + 0.25) - duration
        t = np.linspace(0.25 * reciprocal_hz, duration + phase_completer, round(self._BITRATE * duration))
        time = t + self.play_time
        wave = np.sin(2 * np.pi * key_hz * t)

        self.graphing_sample.append((key, time, wave))
        self._sample = np.concatenate((self.sample, wave))
        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + phase_completer - 0.25 * reciprocal_hz + 1 / round(self._BITRATE * duration)
    def record_chord(self, chords: Iterable, duration: float) -> None:
        """
        Adds the given chords to the sample.

        :param chords: The iterable of chords that you want to play at the same time. Each chord can be a single note (int) or a list/tuple of notes.
        :param duration: Duration of each chord.
        """
        sinusoidal_superposition = np.zeros(int(self._BITRATE * duration))
        max_phase_completer = 0
        max_initial_deflection = 0

        for chord in chords:
            if isinstance(chord, int):
                chord = [chord]  # Treat a single int as a chord with one note
            for key in chord:
                key_hz = self.get_hz(key)
                reciprocal_hz = 1 / key_hz
                # Making sure the wave ends and starts at maxima.
                phase_completer = reciprocal_hz * (round(key_hz * duration) + 0.25) - duration
                initial_deflection = 0.25 * reciprocal_hz
                t = np.linspace(initial_deflection, duration + phase_completer, int(self._BITRATE * duration))
                sinusoidal_superposition += np.sin(2 * np.pi * key_hz * t)

                if initial_deflection > max_initial_deflection:
                    max_initial_deflection = initial_deflection
                if phase_completer > max_phase_completer:
                    max_phase_completer = phase_completer

        # Keeping it in a range of [-1 , 1]
        sinusoidal_superposition /= np.abs(sinusoidal_superposition).max()

        t = np.linspace(max_initial_deflection, duration + max_phase_completer, int(self._BITRATE * duration))
        time = t + self.play_time
        self.graphing_sample.append((tuple(chords), time, sinusoidal_superposition))
        self._sample = np.concatenate((self.sample, sinusoidal_superposition))

        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + max_phase_completer - max_initial_deflection + 1 / round(self._BITRATE * duration)

    def record_drum(self, drum_sample: np.ndarray, duration: float) -> None:
        """
        Adds the drum sample in the sample.
        """
        t = np.linspace(0, duration, int(self._BITRATE * duration))
        drum_wave = np.interp(t, np.linspace(0, duration, len(drum_sample)), drum_sample)
        self._sample = np.concatenate((self._sample, drum_wave))
        self.total_time = np.concatenate((self.total_time, t + self.play_time))
        self.play_time += duration

    def record_flute(self, frequency: float, duration: float) -> None:
        """
        Adds a flute-like sound to the sample.

        :param frequency: Frequency of the flute note.
        :param duration: Duration of the flute note.
        """
        t = np.linspace(0, duration, int(self._BITRATE * duration), endpoint=False)
        # Generate a sine wave as a base
        sine_wave = np.sin(2 * np.pi * frequency * t)

        # Apply a simple envelope to simulate a flute's sound, with an attack and decay
        attack_time = 0.1
        decay_time = 0.1
        sustain_level = 0.7
        release_time = 0.2

        envelope = np.ones_like(t)
        envelope[t < attack_time] = t[t < attack_time] / attack_time
        envelope[(t >= attack_time) & (t < (duration - decay_time))] = 1
        envelope[(t >= (duration - decay_time))] = sustain_level + (1 - sustain_level) * (duration - t[t >= (duration - decay_time)]) / release_time

        # Apply the envelope to the sine wave
        flute_wave = sine_wave * envelope

        # Normalize and add to sample
        self._sample = np.concatenate((self._sample, flute_wave))
        t += self.play_time
        self.total_time = np.concatenate((self.total_time, t))
        self.play_time += duration
            
    def play(self) -> None:
        """
        Plays the sample.
        """
        if self.no_play:
            raise PlayerNotInitialisedError("Player not initialised")
        parsed_sample = self.sample.astype(np.float32)
        stream = self._player.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=self._BITRATE,
                                output=True,
                                frames_per_buffer=512)

        stream.write(parsed_sample.tobytes())

        stream.stop_stream()
        stream.close()

    def close(self) -> None:
        """
        Terminates PyAudio.
        """
        if self.no_play:
            return
        self._player.terminate()

    def clear_sample(self) -> None:
        """
        Clears the sample.
        """
        self._sample = np.array([])

    def to_wav(self, path: str) -> None:
        """
        Convert the sample to wav file format.

        :param path: Path of the file where it will be written.
        """
        # headers for wav format http://www.topherlee.com/software/pcm-tut-wavformat.html
        header = b""
        header += b'RIFF'
        # Leaving an empty space which will be left at the end.
        header += b'\x00\x00\x00\x00'
        header += b'WAVE'
        header += b"fmt "
        sample = self._sample.astype(np.float32)

        fmt_chunk = struct.pack("<HHIIHH", 3, 1, self._BITRATE, self._BITRATE*4, 4, 32)
        fmt_chunk += b"\x00\x00"

        header += struct.pack('<I', len(fmt_chunk))
        header += fmt_chunk

        # added this because it is a non-PCM file.
        header += b'fact'
        header += struct.pack('<II', 4, sample.shape[0])

        file = open(path, "wb")
        file.write(header)
        file.write(b"data")
        file.write(struct.pack('<I', sample.nbytes))

        if sample.dtype.byteorder == '=' and sys.byteorder == 'big':
            sample = sample.byteswap()

        file.write(sample.tobytes())
        # filling that empty space.
        size = file.tell()
        file.seek(4)
        file.write(struct.pack('<I', size - 8))
        file.close()

    @property
    def sample(self):
        """
        Numpy array containing sample data.
        """
        return self._sample
    
# class ModifyAudio:
    @staticmethod
    def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a sine wave.

        :param frequency: Frequency of the sine wave in Hz.
        :param duration: Duration of the wave in seconds.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the sine wave.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
        return waveform

    @staticmethod
    def generate_square_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a square wave.

        :param frequency: Frequency of the square wave in Hz.
        :param duration: Duration of the wave in seconds.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the square wave.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
        return waveform

    @staticmethod
    def generate_triangle_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a triangle wave.

        :param frequency: Frequency of the triangle wave in Hz.
        :param duration: Duration of the wave in seconds.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the triangle wave.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1)
        return waveform

    @staticmethod
    def generate_sawtooth_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a sawtooth wave.

        :param frequency: Frequency of the sawtooth wave in Hz.
        :param duration: Duration of the wave in seconds.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the sawtooth wave.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * (2 * (t * frequency - np.floor(t * frequency + 0.5)))
        return waveform
    

    # --- Effects ---

    @staticmethod
    def apply_gain(waveform: np.ndarray, gain: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply gain to the waveform.

        :param waveform: Numpy array containing the waveform to which gain will be applied.
        :param gain: Gain factor to be applied to the waveform. Must be a positive float.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the waveform with gain applied.
        """
        # Ensure gain is non-negative
        if gain < 0:
            raise ValueError("Gain must be a non-negative value.")
        
        # Apply gain
        waveform_with_gain = waveform * gain
        
        return waveform_with_gain

    @staticmethod
    def apply_low_pass_filter(waveform: np.ndarray, cutoff_freq: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply a low-pass filter to the waveform.

        :param waveform: Numpy array containing the waveform to be filtered.
        :param cutoff_freq: Cutoff frequency for the low-pass filter in Hz.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the filtered waveform.
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        filtered_waveform = lfilter(b, a, waveform)
        return filtered_waveform

    @staticmethod
    def note_to_frequency(note: str) -> float:
        """
        Convert a musical note to its corresponding frequency.

        :param note: Musical note in the format 'NoteOctave' (e.g., 'A4', 'C#5').
        :return: Frequency of the note in Hz.
        :raises ValueError: If the note is not recognized.
        """
        notes = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
            'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
            'A#': 466.16, 'B': 493.88
        }
        base_note, octave = note[:-1], int(note[-1])
        if base_note not in notes:
            raise ValueError(f"Note '{base_note}' is not recognized.")
        frequency = notes[base_note] * (2 ** (octave - 4))
        return frequency
    
    @staticmethod
    def create_note_wave(note: str, duration: float, wave_type: str = 'sine', sample_rate: int = 44100) -> np.ndarray:
        """
        Create a waveform for a single note.

        :param note: Musical note in the format 'NoteOctave' (e.g., 'A4', 'C#5').
        :param duration: Duration of the note in seconds.
        :param wave_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth'). Default is 'sine'.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the waveform for the note.
        :raises ValueError: If the wave_type is not recognized.
        """
        frequency = ModifyAudio.note_to_frequency(note)
        if wave_type == 'sine':
            waveform = ModifyAudio.generate_sine_wave(frequency, duration, sample_rate)
        elif wave_type == 'square':
            waveform = ModifyAudio.generate_square_wave(frequency, duration, sample_rate)
        elif wave_type == 'triangle':
            waveform = ModifyAudio.generate_triangle_wave(frequency, duration, sample_rate)
        elif wave_type == 'sawtooth':
            waveform = ModifyAudio.generate_sawtooth_wave(frequency, duration, sample_rate)
        else:
            raise ValueError(f"Wave type '{wave_type}' is not recognized.")
        return waveform

    @staticmethod
    def save_wavefile(filename: str, waveform: np.ndarray, sample_rate: int = 44100) -> None:
        """
        Save the waveform to a WAV file.

        :param filename: Name of the file to save the waveform to.
        :param waveform: Numpy array containing the waveform to be saved.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        """
        waveform = np.int16(waveform * 32767)
        wavfile.write(filename, sample_rate, waveform)

    @staticmethod
    def generate_melody(notes: list[str], duration: float, wave_type: str = 'sine', sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a melody from a list of notes.

        :param notes: List of musical notes in the format 'NoteOctave' (e.g., ['A4', 'C#5']).
        :param duration: Duration of each note in seconds.
        :param wave_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth'). Default is 'sine'.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the generated melody.
        """
        melody = np.concatenate([ModifyAudio.create_note_wave(note, duration, wave_type, sample_rate) for note in notes])
        return melody

    @staticmethod
    def generate_bassline(notes: list[str], duration: float, wave_type: str = 'sine', sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a bassline from a list of notes.

        :param notes: List of musical notes in the format 'NoteOctave' (e.g., ['E2', 'G#2']).
        :param duration: Duration of each note in seconds.
        :param wave_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth'). Default is 'sine'.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the generated bassline.
        """
        bassline = np.concatenate([ModifyAudio.create_note_wave(note, duration, wave_type, sample_rate) for note in notes])
        return bassline
    

    @staticmethod
    def create_note_wave(note: str, duration: float, wave_type: str = 'sine', sample_rate: int = 44100) -> np.ndarray:
        """
        Create a waveform for a single note.

        :param note: Musical note in the format 'NoteOctave' (e.g., 'A4', 'C#5').
        :param duration: Duration of the note in seconds.
        :param wave_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth'). Default is 'sine'.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the waveform for the note.
        :raises ValueError: If the wave_type is not recognized.
        """
        frequency = ModifyAudio.note_to_frequency(note)
        if wave_type == 'sine':
            waveform = ModifyAudio.generate_sine_wave(frequency, duration, sample_rate)
        elif wave_type == 'square':
            waveform = ModifyAudio.generate_square_wave(frequency, duration, sample_rate)
        elif wave_type == 'triangle':
            waveform = ModifyAudio.generate_triangle_wave(frequency, duration, sample_rate)
        elif wave_type == 'sawtooth':
            waveform = ModifyAudio.generate_sawtooth_wave(frequency, duration, sample_rate)
        else:
            raise ValueError(f"Wave type '{wave_type}' is not recognized.")
        return waveform

    @staticmethod
    def save_wavefile(filename: str, waveform: np.ndarray, sample_rate: int = 44100) -> None:
        """
        Save the waveform to a WAV file.

        :param filename: Name of the file to save the waveform to.
        :param waveform: Numpy array containing the waveform to be saved.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        """
        waveform = np.int16(waveform * 32767)
        wavfile.write(filename, sample_rate, waveform)

    @staticmethod
    def generate_melody(self, notes: list[str], duration: float, wave_type: str = 'sine', sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a melody from a list of notes.

        :param notes: List of musical notes in the format 'NoteOctave' (e.g., ['A4', 'C#5']).
        :param duration: Duration of each note in seconds.
        :param wave_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth'). Default is 'sine'.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the generated melody.
        """
        melody = np.concatenate([self.create_note_wave(note, duration, wave_type, sample_rate) for note in notes])
        return melody

    @staticmethod
    def generate_bassline(notes: list[str], duration: float, wave_type: str = 'sine', sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a bassline from a list of notes.

        :param notes: List of musical notes in the format 'NoteOctave' (e.g., ['E2', 'G#2']).
        :param duration: Duration of each note in seconds.
        :param wave_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth'). Default is 'sine'.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Numpy array containing the generated bassline.
        """
        bassline = np.concatenate([ModifyAudio.create_note_wave(note, duration, wave_type, sample_rate) for note in notes])
        return bassline
    

    @staticmethod
    def apply_reverb(audio: np.ndarray, reverb_amount: float = 0.5, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply reverb effect to an audio signal.

        :param audio: Input audio waveform as a numpy array.
        :param reverb_amount: Amount of reverb to apply (0.0 to 1.0). Default is 0.5.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Audio waveform with reverb applied.
        """
        decay = reverb_amount * np.arange(len(audio)) / sample_rate
        reverb_signal = np.convolve(audio, decay, mode='full')[:len(audio)]
        return audio + reverb_signal

    @staticmethod
    def save_wavefile(filename: str, audio: np.ndarray, sample_rate: int = 44100) -> None:
        """
        Save an audio waveform to a WAV file.

        :param filename: Name of the file to save.
        :param audio: Audio waveform as a numpy array.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        """
        wavfile.write(filename, sample_rate, np.int16(audio * 32767))

    # --- Percussion and Drum Patterns ---

    @staticmethod
    def create_percussion_sound(sample: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Create a percussion sound based on a sample frequency.

        :param sample: Frequency of the percussion sample in Hz.
        :param duration: Duration of the sound in seconds.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Percussion sound waveform as a numpy array.
        """
        sample_waveform = ModifyAudio.generate_sine_wave(sample, duration, sample_rate)
        return sample_waveform

    @staticmethod
    def generate_drum_pattern(pattern: list[tuple[float, float, float]], duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a drum pattern based on a list of hits.

        :param pattern: List of tuples, each containing (hit_time, hit_duration, frequency).
        :param duration: Duration of the drum pattern in seconds.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Drum track waveform as a numpy array.
        """
        drum_track = np.zeros(int(duration * sample_rate))
        for hit in pattern:
            start = int(hit[0] * sample_rate)
            end = start + int(hit[1] * sample_rate)
            drum_track[start:end] += ModifyAudio.create_percussion_sound(hit[2], hit[1], sample_rate)
        return drum_track

    # --- Random Melody, Bassline, and Drum Pattern Generation ---

    @staticmethod
    def generate_random_bassline(length: int, note_duration: float, wave_type: str = 'square', sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a random bassline using predefined notes.

        :param length: Number of notes in the bassline.
        :param note_duration: Duration of each note in seconds.
        :param wave_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth'). Default is 'square'.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Bassline waveform as a numpy array.
        """
        possible_notes = ['C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2', 'C3']
        bassline_notes = random.choices(possible_notes, k=length)
        bassline = ModifyAudio.generate_bassline(bassline_notes, note_duration, wave_type, sample_rate)
        return bassline

    # --- Mixing and Saving ---
    def apply_reverb(waveform, reverb_amount=0.3, sample_rate=44100):
        """
        Applies a simple reverb effect by adding a delayed and attenuated copy of the waveform to itself.

        :param waveform: The input waveform.
        :param reverb_amount: The amount of reverb to apply (0 to 1).
        :param sample_rate: The sample rate of the audio.
        :return: Waveform with reverb applied.
        """
        delay_samples = int(0.03 * sample_rate)  # 30ms delay
        reverb_wave = np.zeros_like(waveform)
        reverb_wave[delay_samples:] = waveform[:-delay_samples] * reverb_amount
        return np.clip(waveform + reverb_wave, -1, 1)

    def apply_echo(waveform, delay=0.2, decay=0.5, sample_rate=44100):
        """
        Adds an echo effect by delaying the original waveform and adding it back to the signal.

        :param waveform: The input waveform.
        :param delay: The delay of the echo in seconds.
        :param decay: The decay factor for the echo (0 to 1).
        :param sample_rate: The sample rate of the audio.
        :return: Waveform with echo applied.
        """
        delay_samples = int(sample_rate * delay)
        echo_wave = np.zeros_like(waveform)
        echo_wave[delay_samples:] = waveform[:-delay_samples] * decay
        return np.clip(waveform + echo_wave, -1, 1)

    # --- Advanced Audio Manipulation ---

    def pitch_shift(waveform, semitones, sample_rate=44100):
        """
        Shifts the pitch of the waveform by a specified number of semitones.

        :param waveform: The input waveform.
        :param semitones: The number of semitones to shift (positive or negative).
        :param sample_rate: The sample rate of the audio.
        :return: Pitch-shifted waveform.
        """
        factor = 2 ** (semitones / 12.0)
        indices = np.round(np.arange(0, len(waveform), factor))
        indices = indices[indices < len(waveform)].astype(int)
        return waveform[indices]

    def time_stretch(waveform, stretch_factor, sample_rate=44100):
        """
        Stretches or compresses the time of the waveform.

        :param waveform: The input waveform.
        :param stretch_factor: The factor by which to stretch or compress the time (greater than 1 stretches, less than 1 compresses).
        :param sample_rate: The sample rate of the audio.
        :return: Time-stretched waveform.
        """
        indices = np.round(np.arange(0, len(waveform), 1/stretch_factor))
        indices = indices[indices < len(waveform)].astype(int)
        return waveform[indices]

    @staticmethod
    def numpy_to_pydub(waveform: np.ndarray, sample_rate: int = 44100) -> AudioSegment:
        """
        Converts a numpy waveform to a PyDub AudioSegment.

        :param waveform: Input waveform as a numpy array.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: PyDub AudioSegment.
        """
        if not isinstance(waveform, np.ndarray):
            raise TypeError("Waveform must be a numpy array")
        if waveform.dtype != np.int16:
            waveform = waveform.astype(np.int16)

        # Convert numpy array to AudioSegment
        return AudioSegment(
            waveform.tobytes(),
            frame_rate=sample_rate,
            sample_width=waveform.dtype.itemsize,
            channels=1
        )

    @staticmethod
    def export_to_mp3(self, filename: str, waveform: np.ndarray, sample_rate: int = 44100, bitrate: str = "192k") -> None:
        """
        Exports the waveform to an MP3 file using PyDub.

        :param filename: The name of the file to save.
        :param waveform: The waveform data to save.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :param bitrate: The bitrate of the MP3 file. Default is "192k".
        :return: None
        """
        try:
            # Convert the waveform to a pydub AudioSegment
            audio_segment = self.numpy_to_pydub(waveform, sample_rate)

            # Export the AudioSegment as an MP3 file
            audio_segment.export(filename, format="mp3", bitrate=bitrate)

            print(f"Successfully exported to {filename}")

        except Exception as e:
            print(f"An error occurred during export: {e}")

    @staticmethod
    def export_to_flac(self, filename: str, waveform: np.ndarray, sample_rate: int = 44100) -> None:
        """
        Exports the waveform to a FLAC file using PyDub.

        :param filename: The name of the file to save.
        :param waveform: The waveform data to save.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :return: None
        """
        audio_segment = self.numpy_to_pydub(waveform, sample_rate)
        audio_segment.export(filename, format="flac")

    @staticmethod
    def export_to_ogg(filename: str, waveform: np.ndarray, sample_rate: int = 44100, quality: int = 5) -> None:
        """
        Exports the waveform to an OGG file using PyDub.

        :param filename: The name of the file to save.
        :param waveform: The waveform data to save.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :param quality: The quality level of the OGG file (0 to 10). Default is 5.
        :return: None
        """
        audio_segment = ModifyAudio.numpy_to_pydub(waveform, sample_rate)
        audio_segment.export(filename, format="ogg", quality=quality)

    # --- Utilities ---

    @staticmethod         
    def combine_tracks(track1: np.ndarray, track2: np.ndarray) -> np.ndarray:
        """
        Combines two waveforms into one by averaging their values.

        :param track1: The first waveform.
        :param track2: The second waveform.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :return: Combined waveform as a numpy array.
        """
        max_len = max(len(track1), len(track2))
        
        if len(track1) < max_len:
            track1 = np.pad(track1, (0, max_len - len(track1)), 'constant')
        if len(track2) < max_len:
            track2 = np.pad(track2, (0, max_len - len(track2)), 'constant')
        
        combined = track1 + track2
        return combined

    @staticmethod
    def add_silence(waveform: np.ndarray, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Adds silence to the beginning and end of the waveform.

        :param waveform: The input waveform.
        :param duration: Duration of silence in seconds.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :return: Waveform with added silence.
        """
        silence = np.zeros(int(sample_rate * duration))
        return np.concatenate((silence, waveform, silence))

    @staticmethod
    def mix_tracks_pydub(*tracks: AudioSegment) -> AudioSegment:

        """
        Mixes multiple tracks together using PyDub's overlay method.

        :param tracks: A variable number of `AudioSegment` objects to be mixed together.
        :return: A single `AudioSegment` containing the mixed tracks.
        """
        combined = tracks[0]
        for track in tracks[1:]:
            combined = combined.overlay(track)
        return combined
    
    @staticmethod
    def export_to_wav(self, filename: str, waveform: np.ndarray, sample_rate: int = 44100) -> None:
        """
        Exports the waveform to a WAV file using PyDub.

        :param filename: The name of the file to save.
        :param waveform: The waveform data to save.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :return: None
        """
        audio_segment = self.numpy_to_pydub(waveform, sample_rate)
        audio_segment.export(filename, format="wav")