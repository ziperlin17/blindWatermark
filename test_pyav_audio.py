import av

container = av.open('large.mp4')

# Общий битрейт контейнера (если доступно)
print(f"Container bit_rate: {container.bit_rate} bit/s")

for stream in container.streams:
    # Skip any non-audio/non-video streams
    if stream.type not in ('video', 'audio'):
        print(f"Skipping stream #{stream.index} of type '{stream.type}'")
        continue

    ctx = stream.codec_context
    # Skip if there's no codec context
    if ctx is None:
        print(f"  Stream #{stream.index} has no codec context, skipping")
        continue

    print(f"\nStream #{stream.index}:")
    print(f"  Type:  {stream.type}")
    print(f"  Codec: {ctx.name}")  # e.g. 'h264', 'aac', 'dts'

    if stream.type == 'video':
        print(f"  Resolution: {ctx.width}×{ctx.height}")
        # average_rate is a Fraction; convert to float or keep as rational
        print(f"  Frame rate: {stream.average_rate}")
        print(f"  Bit rate:   {ctx.bit_rate} bit/s")

    else:  # audio
        print(f"  Sample rate:     {ctx.sample_rate} Hz")
        # Determine bits per sample if possible:
        try:
            # ctx.format is an AudioFormat with 'bytes' attribute
            bits = ctx.format.bytes * 8
        except Exception:
            # Fallback: parse digits from format name (e.g. 's16' → 16)
            fmt_name = ctx.format.name if ctx.format else ''
            bits = int(''.join(filter(str.isdigit, fmt_name)) or 0)
        print(f"  Bits per sample: {bits}")
        print(f"  Channels:        {ctx.channels}")
        print(f"  Bit rate:        {ctx.bit_rate} bit/s")