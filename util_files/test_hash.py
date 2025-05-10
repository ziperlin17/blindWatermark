import hashlib
import json

def compute_track_meta_hash(video_id: int, secret: bytes = b""):
    # ID + секрет → байты → SHA3-256
    id_bytes = video_id.to_bytes(8, "big")
    data = secret + id_bytes if secret else id_bytes
    return hashlib.sha3_256(data).hexdigest()

def inject_track_meta_hash(json_path: str, output_path: str, video_id: int, secret: bytes = b""):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta_hash = compute_track_meta_hash(video_id, secret)

    # Находим раздел с TrackID / TrackDuration
    for key, section in data.items():
        if isinstance(section, dict) and "track_id" in [v.get("key_formatted") for v in section.values() if isinstance(v, dict)]:
            # Вставляем новое поле
            section["TrackMetaHash"] = {
                "editable": False,
                "key_formatted": "track_meta_hash",
                "value": meta_hash
            }
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Пример вызова
inject_track_meta_hash("watermarked_ffmpeg_t9.json", "output_with_meta.json", video_id=0x1234567890ABCDEF)