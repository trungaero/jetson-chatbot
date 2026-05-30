import pyaudio, wave

p = pyaudio.PyAudio()

# List input devices
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(f"  [{i}] {info['name']}")

# Record 5 seconds
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
print("Recording 5s...")
frames = [stream.read(1024) for _ in range(int(16000 / 1024 * 5))]
stream.stop_stream()
stream.close()

# Save
wf = wave.open("/tmp/test_pyaudio.wav", "wb")
wf.setnchannels(1)
wf.setsampwidth(2)
wf.setframerate(16000)
wf.writeframes(b"".join(frames))
wf.close()
p.terminate()
print("Saved to /tmp/test_pyaudio.wav")