import pyaudio
import numpy as np
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print('start recording')

frames = []
seconds = 3
for i in range(0,int(RATE / CHUNK * seconds)):
    data = stream.read(CHUNK)
    frames.append(data)

print('record stopped')

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("output.wav",'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()