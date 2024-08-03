from collections.abc import Iterable
import struct
import sys
import numpy as np
import pyaudio
from scipy.io import wavfile
from pydub import AudioSegment

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

    @staticmethod
    def get_hz(key_number: int) -> float:
        """
        Get frequency of the key via its key number.

        :return: frequency of the key.
        """
        return 2 ** ((key_number - 49) / 12) * 440

    def record_key(self, key: int, duration: float) -> None:
        """
        Adds the key in the sample.

        :param key: The key number of the key's name.
        :param duration: Time for which the key it to be played.
        :return: None
        """
        key_hz = self.get_hz(key)
        reciprocal_hz = 1 / key_hz
        # making sure the wave ends and starts at maxima.
        phase_completer = reciprocal_hz*(round(key_hz*duration) + 0.25) - duration
        t = np.linspace(0.25*reciprocal_hz, duration + phase_completer, round(self._BITRATE * duration))
        # sinusoidal waves are a function of sine with args 2*pi*frequency*t.
        time = t + self.play_time
        wave = np.sin(2 * np.pi * key_hz * t)

        self.graphing_sample.append((key, time, wave))

        self.sample = np.concatenate((self.sample, wave))
        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + phase_completer - 0.25*reciprocal_hz + 1/round(self._BITRATE * duration)

    def record_chord(self, chords: Iterable, duration: float) -> None:
        """
        Adds the given chords in the sample.

        :param chords: The iterable of chords that you want to play at same time.
        :param duration: Duration of each chord.
        """
        sinusoidal_superposition = np.empty((int(self._BITRATE * duration)))
        max_phase_completer = 0
        max_initial_deflection = 0
        for chord in chords:
            for key in chord:
                key_hz = self.get_hz(key)
                reciprocal_hz = 1 / key_hz
                # making sure the wave ends and starts at maxima.
                phase_completer = reciprocal_hz * (round(key_hz * duration) + 0.25) - duration
                initial_deflection = 0.25*reciprocal_hz
                t = np.linspace(initial_deflection, duration + phase_completer, int(self._BITRATE * duration))
                sinusoidal_superposition += np.sin(2 * np.pi * key_hz * t)

                if initial_deflection > max_initial_deflection:
                    max_initial_deflection = initial_deflection
                if phase_completer > max_phase_completer:
                    max_phase_completer = phase_completer

        # keeping it in a range of [-1 , 1]
        sinusoidal_superposition = sinusoidal_superposition / sinusoidal_superposition.max()

        t = np.linspace(max_initial_deflection, duration + max_phase_completer, int(self._BITRATE * duration))
        time = t+self.play_time
        self.graphing_sample.append((tuple(chords), time, sinusoidal_superposition))
        self.sample = np.concatenate((self.sample, sinusoidal_superposition))

        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + max_phase_completer - max_initial_deflection + 1/round(self._BITRATE * duration)

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
        sample = self.sample.astype(np.float32)

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
    
class modify_audio:
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