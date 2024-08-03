from collections.abc import Iterable
import struct
import sys
import numpy as np
import pyaudio
from scipy.io import wavfile
from pydub import AudioSegment
import random

class PlayerNotInitialisedError(AttributeError):
    """Exception raised when trying to play audio without initialising the player."""
    pass

class Instrument:
    """
    Class to handle audio synthesis and playback.

    :param bit_rate: Bit rate of the audio.
    :param no_play: Whether to disable playback functionality.
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
        """Get frequency of the key via its key number."""
        return 2 ** ((key_number - 49) / 12) * 440

    def record_key(self, key: int, duration: float) -> None:
        """Record a single key."""
        key_hz = self.get_hz(key)
        reciprocal_hz = 1 / key_hz
        phase_completer = reciprocal_hz * (round(key_hz * duration) + 0.25) - duration
        t = np.linspace(0.25 * reciprocal_hz, duration + phase_completer, round(self._BITRATE * duration))
        time = t + self.play_time
        wave = np.sin(2 * np.pi * key_hz * t)

        self.graphing_sample.append((key, time, wave))

        self._sample = np.concatenate((self._sample, wave))
        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + phase_completer - 0.25 * reciprocal_hz + 1 / round(self._BITRATE * duration)

    def record_chord(self, chords: Iterable, duration: float) -> None:
        """Record multiple keys played as a chord."""
        sinusoidal_superposition = np.empty((int(self._BITRATE * duration)))
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

        sinusoidal_superposition /= np.max(np.abs(sinusoidal_superposition))

        t = np.linspace(max_initial_deflection, duration + max_phase_completer, int(self._BITRATE * duration))
        time = t + self.play_time
        self.graphing_sample.append((tuple(chords), time, sinusoidal_superposition))
        self._sample = np.concatenate((self._sample, sinusoidal_superposition))

        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + max_phase_completer - max_initial_deflection + 1 / round(self._BITRATE * duration)

    def record_drum(self, drum_sample: np.ndarray, duration: float) -> None:
        """Record a drum sample."""
        t = np.linspace(0, duration, int(self._BITRATE * duration))
        drum_wave = np.interp(t, np.linspace(0, duration, len(drum_sample)), drum_sample)
        self._sample = np.concatenate((self._sample, drum_wave))
        self.total_time = np.concatenate((self.total_time, t + self.play_time))
        self.play_time += duration

    def record_flute(self, frequency: float, duration: float) -> None:
        """Record a flute-like sound."""
        t = np.linspace(0, duration, int(self._BITRATE * duration), endpoint=False)
        sine_wave = np.sin(2 * np.pi * frequency * t)

        attack_time = 0.1
        decay_time = 0.1
        sustain_level = 0
