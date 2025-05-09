import av

container = av.open('../test_video.mp4')

audio_stream = container.streams.audio[0]

bitrate = audio_stream.codec_context.bit_rate

print(f"Аудиобитрейт: {bitrate} bps")
