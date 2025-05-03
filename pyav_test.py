import av
import logging
import sys
import os
from fractions import Fraction # Нужен для заглушки, если импорт av не удался

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Импорт PyAV с обработкой ошибок ---
try:
    from av.error import FFmpegError, EOFError as FFmpegEOFError
    PYAV_AVAILABLE = True
    logging.info("PyAV library and FFmpegError imported successfully.")
except ImportError:
    PYAV_AVAILABLE = False
    logging.error("PyAV library not found! Install it: pip install av")
    class av_dummy:
        class VideoFrame: pass
        FFmpegError = Exception
        EOFError = EOFError
    av = av_dummy
    FFmpegError = Exception
    FFmpegEOFError = EOFError

def test_hw_decode_formats(video_path, hwaccel_options_list):
    """
    Пытается декодировать несколько кадров видео с разными опциями HWAccel
    и выводит формат пикселей полученных кадров.

    Args:
        video_path (str): Путь к видеофайлу.
        hwaccel_options_list (list): Список словарей с опциями для av.open.
                                     Пример: [{'hwaccel': 'cuda'}, {'hwaccel': 'dxva2'}, {}]
                                     Пустой словарь {} означает декодирование на CPU.
    """
    if not PYAV_AVAILABLE:
        print("PyAV not available. Cannot perform test.")
        return

    if not os.path.exists(video_path):
        print(f"Error: Input video file not found at '{video_path}'")
        return

    print(f"\n--- Testing HW Decode Formats for: {video_path} ---")

    for i, options in enumerate(hwaccel_options_list):
        print(f"\n--- Test {i+1}: Options = {options} ---")
        input_container = None
        try:
            logging.info(f"Attempting to open with options: {options}")
            # Передаем опции как container_options (или options, если нужно)
            input_container = av.open(video_path, mode='r', container_options=options)
            logging.info(f"Opened successfully.")

            video_stream = input_container.streams.video[0]
            video_stream_index = video_stream.index
            print(f"Found video stream #{video_stream_index} ({video_stream.codec.name})")

            decoded_frames = 0
            unique_formats = set()

            # Декодируем несколько первых пакетов/кадров
            for packet in input_container.demux():
                if packet.stream.index != video_stream_index:
                    continue
                if packet.dts is None: continue
                if decoded_frames >= 5: break # Достаточно нескольких кадров для теста

                try:
                    for frame in packet.decode():
                        if frame and isinstance(frame, av.VideoFrame):
                            format_name = frame.format.name if frame.format else "Unknown"
                            print(f"  Frame {decoded_frames}: PTS={frame.pts}, Format={format_name}, Size={frame.width}x{frame.height}")
                            if frame.format:
                                unique_formats.add(format_name)
                            decoded_frames += 1
                            if decoded_frames >= 5: break
                except (FFmpegError, ValueError) as e_decode:
                    logging.warning(f"Error decoding packet: {e_decode}")
                    # Можно добавить break, если ошибка критична

            if decoded_frames == 0:
                 print("  No video frames decoded.")
            else:
                 print(f"  Unique frame formats detected: {unique_formats or 'None'}")


        except (FFmpegError, ValueError, IndexError, AttributeError, Exception) as e:
            print(f"  ERROR opening or processing with options {options}: {e}")
            logging.error(f"Error testing options {options}: {e}", exc_info=True)
        finally:
            if input_container:
                try:
                    input_container.close()
                except FFmpegError as e_close:
                    logging.error(f"Error closing container: {e_close}")

    print("\n--- Test Finished ---")


# --- Пример использования ---
if __name__ == "__main__":
    # Укажите путь к вашему тестовому видео
    # test_video_file = 'input.mp4' # Видео H.264/AAC
    test_video_file = 'test_video.mp4' # Видео H.264/AAC

    # Список опций HWAccel для тестирования
    # Добавьте/удалите нужные вам варианты
    options_to_test = [
        {}, # Тест CPU декодирования (без опций)
        {'hwaccel': 'auto'}, # Попытка автовыбора
        {'hwaccel': 'cuda', 'hwaccel_device': '0'}, # CUDA/NVDEC
        {'hwaccel': 'cuvid', 'hwaccel_device': '0'}, # CUVID (старый API NVDEC)
        {'hwaccel': 'nvdec', 'hwaccel_device': '0'}, # NVDEC (новый API, может требоваться в опциях кодека)
        # {'hwaccel': 'dxva2'}, # Для Windows, если поддерживается FFmpeg
        # {'hwaccel': 'd3d11va'},# Для Windows, если поддерживается FFmpeg
        # {'hwaccel': 'qsv', 'hwaccel_device': 'auto'}, # Для Intel QSV
        # {'hwaccel': 'vaapi', 'hwaccel_device': '/dev/dri/renderD128'}, # Пример для VAAPI Linux
        # {'hwaccel': 'amf'}, # Для AMD AMF
    ]

    test_hw_decode_formats(test_video_file, options_to_test)