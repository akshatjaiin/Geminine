from Instrument.Instrument import Instrument

piano = Instrument(bit_rate = 44100)
piano.record_key(52, duration=0.3)  # C5
piano.record_chord([(52, 56, 61)], duration=0.3)  # C5 E5 A5

piano.play()
piano.close()   # Terminates PyAudio

guitar = Instrument(44100)
guitar.record_key(43, duration=3)  # A
guitar.play()
guitar.clear_sample()  # clears the sample
guitar.close()