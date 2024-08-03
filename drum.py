import numpy as np
import soundfile as sf
import pyaudio
import struct
import sys
from collections.abc import Iterable

class Instrument:
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
        return 2 ** ((key_number - 49) / 12) * 440

    def record_key(self, key: int, duration: float) -> None:
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

    def record_chord(self, chords:Iterable, duration: float) -> None:
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

        sinusoidal_superposition = sinusoidal_superposition / sinusoidal_superposition.max()
        t = np.linspace(max_initial_deflection, duration + max_phase_completer, int(self._BITRATE * duration))
        time = t + self.play_time
        self.graphing_sample.append((tuple(chords), time, sinusoidal_superposition))
        self._sample = np.concatenate((self._sample, sinusoidal_superposition))
        self.total_time = np.concatenate((self.total_time, time))
        self.play_time += duration + max_phase_completer - max_initial_deflection + 1 / round(self._BITRATE * duration)

    def record_drum(self, drum_file: str, duration: float) -> None:
        """
        Loads a drum sample and adds it to the sample array.
        """
        drum_sample, _ = sf.read(drum_file)
        drum_sample = np.interp(np.linspace(0, len(drum_sample), int(self._BITRATE * duration)), 
                                np.arange(len(drum_sample)), drum_sample)
        self._sample = np.concatenate((self._sample, drum_sample))
        self.total_time = np.concatenate((self.total_time, np.linspace(self.play_time, self.play_time + duration, int(self._BITRATE * duration))))
        self.play_time += duration

    def play(self) -> None:
        if self.no_play:
            raise PlayerNotInitialisedError("Player not initialised")
        parsed_sample = self._sample.astype(np.float32)
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

        fmt_chunk = struct.pack("<HHIIHH", 3, 1, self._BITRATE, self._BITRATE*4, 4, 32)
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

    @property
    def sample(self):
        return self._sample
