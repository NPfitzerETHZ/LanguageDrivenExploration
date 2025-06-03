import pyaudio

pa = pyaudio.PyAudio()

print("=== List of Audio Devices ===")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(f"{i}: {info['name']} - "
          f"{'Input' if info['maxInputChannels'] > 0 else 'Output'} - "
          f"Sample Rate: {info['defaultSampleRate']}")
