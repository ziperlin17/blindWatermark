import av

# Открываем контейнер (файл)
container = av.open('test_video.mp4')

# Берём первый аудиопоток
audio_stream = container.streams.audio[0]

# Считываем битрейт из codec_context
# Обычно он хранится в битах в секунду (bps)
bitrate = audio_stream.codec_context.bit_rate

print(f"Аудиобитрейт: {bitrate} bps")
