from collections.abc import Iterable
import struct
import sys
import numpy as np
import pyaudio
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import random

class PlayerNotInitialisedError(AttributeError):
    """
    Custom error for uninitialized player.
    """
    pass

class Instrument:
    """
    Class for playing and recording different musical instruments.
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
        self.sample = np.concatenate((self.sample, wave))
        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + phase_completer - 0.25 * reciprocal_hz + 1 / round(self._BITRATE * duration)
    
    def record_chord(self, chords: Iterable, duration: float) -> None:
        sinusoidal_superposition = np.empty(int(self._BITRATE * duration))
        max_phase_completer = 0
        max_initial_deflection = 0
        for chord in chords:
            for key in chord:
                key_hz = self.get_hz(key)
                reciprocal_hz = 1 / key_hz
                phase_completer = reciprocal_hz * (round(key_hz * duration) + 0.25) - duration
                initial_deflection = 0.25 * reciprocal_hz
                t = np.linspace(initial_deflection, duration + phase_completer, int(self._BITRATE * duration))
                sinusoidal_superposition += np.sin(2 * np.pi * key_hz * t)

                if initial_deflection > max_initial_deflection:
                    max_initial_deflection = initial_deflection
                if phase_completer > max_phase_completer:
                    max_phase_completer = phase_completer

        sinusoidal_superposition /= sinusoidal_superposition.max()

        t = np.linspace(max_initial_deflection, duration + max_phase_completer, int(self._BITRATE * duration))
        time = t + self.play_time
        self.graphing_sample.append((tuple(chords), time, sinusoidal_superposition))
        self.sample = np.concatenate((self.sample, sinusoidal_superposition))

        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + max_phase_completer - max_initial_deflection + 1 / round(self._BITRATE * duration)

    def record_drum(self, drum_sample: np.ndarray, duration: float) -> None:
        t = np.linspace(0, duration, int(self._BITRATE * duration))
        drum_wave = np.interp(t, np.linspace(0, duration, len(drum_sample)), drum_sample)
        self._sample = np.concatenate((self._sample, drum_wave))
        self.total_time = np.concatenate((self.total_time, t + self.play_time))
        self.play_time += duration

    def record_flute(self, frequency: float, duration: float) -> None:
        t = np.linspace(0, duration, int(self._BITRATE * duration), endpoint=False)
        sine_wave = np.sin(2 * np.pi * frequency * t)
        attack_time = 0.1
        decay_time = 0.1
        sustain_level = 0.7
        release_time = 0.2

        envelope = np.ones_like(t)
        envelope[t < attack_time] = t[t < attack_time] / attack_time
        envelope[(t >= attack_time) & (t < (duration - decay_time))] = 1
        envelope[(t >= (duration - decay_time))] = sustain_level + (1 - sustain_level) * (duration - t[t >= (duration - decay_time)]) / release_time

        flute_wave = sine_wave * envelope
        self._sample = np.concatenate((self._sample, flute_wave))
        t += self.play_time
        self.total_time = np.concatenate((self.total_time, t))
        self.play_time += duration
            
    def play(self) -> None:
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
        if self.no_play:
            return
        self._player.terminate()

    def clear_sample(self) -> None:
        self._sample = np.array([])

    def to_wav(self, path: str) -> None:
        header = b""
        header += b'RIFF'
        header += b'\x00\x00\x00\x00'
        header += b'WAVE'
        header += b"fmt "
        sample = self._sample.astype(np.float32)

        fmt_chunk = struct.pack("<HHIIHH", 3, 1, self._BITRATE, self._BITRATE * 4, 4, 32)
        fmt_chunk += b"\x00\x00"

        header += struct.pack('<I', len(fmt_chunk))
        header += fmt_chunk
        header += b'fact'
        header += struct.pack('<II', 4, sample.shape[0])

        file = open(path, "wb")
        file.write(header)
        file.write(b"data")
        file.write(struct.pack('<I', sample.nbytes))

        if sample.dtype.byteorder == '=' and sys.byteorder == 'big':
            sample = sample.byteswap()

        file.write(sample.tobytes())
        size = file.tell()
        file.seek(4)
        file.write(struct.pack('<I', size - 8))
        file.close()

class ModifyAudio:
    @staticmethod
    def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
        return waveform

    @staticmethod
    def generate_square_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
        return waveform

    @staticmethod
    def generate_triangle_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1)
        return waveform

    @staticmethod
    def generate_sawtooth_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * (2 * (t * frequency - np.floor(t * frequency + 0.5)))
        return waveform

    @staticmethod
    def apply_gain(waveform: np.ndarray, gain: float, sample_rate: int = 44100) -> np.ndarray:
        if gain < 0:
            raise ValueError("Gain must be non-negative")
        return waveform * gain

    @staticmethod
    def apply_low_pass_filter(waveform: np.ndarray, cutoff_freq: float, sample_rate: int = 44100) -> np.ndarray:
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(6, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, waveform)

    @staticmethod
    def generate_random_melody(self, num_notes: int = 10, min_freq: float = 261.63, max_freq: float = 523.25, duration: float = 0.5, sample_rate: int = 44100) -> np.ndarray:
        melody = np.array([])
        for _ in range(num_notes):
            frequency = random.uniform(min_freq, max_freq)
            note = self.generate_sine_wave(frequency, duration, sample_rate)
            melody = np.concatenate((melody, note))
        return melody

    @staticmethod
    def generate_random_bassline(num_notes: int = 10, min_freq: float = 41.20, max_freq: float = 82.41, duration: float = 0.5, sample_rate: int = 44100) -> np.ndarray:
        bassline = np.array([])
        for _ in range(num_notes):
            frequency = random.uniform(min_freq, max_freq)
            note = ModifyAudio.generate_sine_wave(frequency, duration, sample_rate)
            bassline = np.concatenate((bassline, note))
        return bassline
    

    
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

    @staticmethod
    def generate_random_drum_pattern(length: int, note_duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Generate a random drum pattern.

        :param length: Number of beats in the pattern.
        :param note_duration: Duration of each beat in seconds.
        :param sample_rate: Number of samples per second (Hz). Default is 44100.
        :return: Drum track waveform as a numpy array.
        """
        drum_pattern = []
        for _ in range(length):
            hit_time = random.uniform(0, note_duration)
            hit_duration = random.uniform(0.05, 0.15)
            frequency = random.uniform(100, 300)
            drum_pattern.append((hit_time, hit_duration, frequency))
        drum_track = ModifyAudio.generate_drum_pattern(drum_pattern, length * note_duration, sample_rate)
        return drum_track
    
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
        waveform = np.int16(waveform * 32767)
        audio_segment = AudioSegment(
            waveform.tobytes(), 
            frame_rate=sample_rate,
            sample_width=waveform.dtype.itemsize, 
            channels=1
        )
        return audio_segment

    @staticmethod
    def export_to_mp3(filename: str, waveform: np.ndarray, sample_rate: int = 44100, bitrate: str = "192k") -> None:
        """
        Exports the waveform to an MP3 file using PyDub.

        :param filename: The name of the file to save.
        :param waveform: The waveform data to save.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :param bitrate: The bitrate of the MP3 file. Default is "192k".
        :return: None
        """
        audio_segment = ModifyAudio.numpy_to_pydub(waveform, sample_rate)
        audio_segment.export(filename, format="mp3", bitrate=bitrate)

    @staticmethod
    def export_to_flac(filename: str, waveform: np.ndarray, sample_rate: int = 44100) -> None:
        """
        Exports the waveform to a FLAC file using PyDub.

        :param filename: The name of the file to save.
        :param waveform: The waveform data to save.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :return: None
        """
        audio_segment = ModifyAudio.numpy_to_pydub(waveform, sample_rate)
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
    def combine_tracks(track1: np.ndarray, track2: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Combines two waveforms into one by averaging their values.

        :param track1: The first waveform.
        :param track2: The second waveform.
        :param sample_rate: The sample rate of the audio. Default is 44100.
        :return: Combined waveform as a numpy array.
        """
        combined = track1 + track2
        if np.max(np.abs(combined)) == 0:
            return None
        return np.clip(combined / np.max(np.abs(combined)), -1, 1)

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
