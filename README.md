# Audio Instrument and Effects Processor

This project provides a comprehensive audio instrument simulation and processing framework. It allows for the creation of custom audio waveforms, playing of piano notes, recording musical chords, and applying various audio effects, including gain, reverb, echo, and more. It also supports saving the audio as a `.wav` file and manipulating audio using PyDub.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Generating Notes](#generating-notes)
  - [Recording and Playback](#recording-and-playback)
  - [Applying Effects](#applying-effects)
  - [Saving and Exporting](#saving-and-exporting)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [License](#license)

## Installation

1. **Install the Required Packages**:
   Ensure you have Python installed. You can install the necessary dependencies by running:

   ```bash
   pip install -r requirements.txt
pip install numpy scipy pyaudio pydub
Features
Waveform Generation:

Sine, square, triangle, and sawtooth waveforms.
Creation of waveforms for specific musical notes.
Musical Instrument Simulation:

Simulate piano keys by calculating the frequency of a note from its key number.
Record individual notes, chords, and percussion instruments.
Audio Effects:

Apply effects such as gain, low-pass filtering, reverb, and echo to the audio samples.
Pitch shift and time stretch the audio waveforms.
Wave Manipulation:

Combine multiple waveforms, add silence, or overlay tracks using PyDub.
Audio Export:

Save audio samples as .wav files.
Convert numpy arrays to PyDub's AudioSegment for further audio manipulation.
Usage
Generating Notes
You can generate notes for different musical instruments or frequencies:

python
Copy code
instrument = Instrument(bit_rate=44100, no_play=True)
note_wave = instrument.create_note_wave("A4", 2.0, wave_type='sine')
Recording and Playback
You can record individual notes or chords, and play them using PyAudio:

python
Copy code
instrument.record_key(49, 1.0)  # Record key 49 (A4) for 1 second
instrument.play()
Record a chord:

python
Copy code
instrument.record_chord([49, 52, 56], 2.0)  # A4, C#5, E5
Applying Effects
You can apply various effects like gain, reverb, or filters:


wave_with_gain = instrument.apply_gain(note_wave, gain=2.0)
filtered_wave = instrument.apply_low_pass_filter(wave_with_gain, cutoff_freq=2000)
Saving and Exporting
Export the generated audio to a .wav file:


instrument.to_wav("output.wav")
Convert a numpy waveform to a PyDub AudioSegment:

![image](https://github.com/user-attachments/assets/e0d26663-e44c-426b-a3da-e5713f5ecef5)

audio_segment = instrument.numpy_to_pydub(waveform)
audio_segment.export("output.mp3", format="mp3")
Project Structure
plaintext
Copy code
├── instrument.py          # Main script containing the Instrument class
├── requirements.txt       # Dependencies required for the project
├── README.md              # Project documentation
└── .pys             # Example usage script
