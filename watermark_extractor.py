import cv2
import numpy as np
import logging
import time
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional

# --- Константы (должны совпадать с эмбеддером!) ---
# Настройки адаптивной альфы (на основе статьи 1)
LAMBDA_PARAM: float = 0.04  # Параметр λ из формулы α = λ / (1 + exp(-(Ee/Ev)))
ALPHA_MIN: float = 1.01     # Минимальное значение альфы (немного > 1)
ALPHA_MAX: float = 1.20     # Максимальное значение альфы

# Общие настройки
N_RINGS: int = 8           # Количество колец для разбиения
DEFAULT_RING_INDEX: int = 3 # Индекс кольца по умолчанию, если не используется адаптивный выбор
FPS: int = 30              # Используется только для логирования времени
LOG_FILENAME: str = 'watermarking.log' # Лог файл (дозапись)

# --- Логирование ---
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='a',  # 'a' - дописывать в существующий лог
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logging.info("Запуск скрипта извлечения вотермарки.")

# --- Общие функции (Идентичные эмбеддеру) ---

def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    """Одномерное DCT-преобразование с нормировкой."""
    return dct(signal_1d, type=2, norm='ortho')

def idct_1d(coeffs_1d: np.ndarray) -> np.ndarray:
    """Обратное одномерное DCT-преобразование с нормировкой."""
    return idct(coeffs_1d, type=2, norm='ortho')

def rgb2ycbcr(frame_bgr: np.ndarray) -> np.ndarray:
    """Перевод BGR-кадра в яркостную плоскость Y (YCbCr) в диапазоне [0, 1]."""
    if frame_bgr.dtype != np.uint8:
        logging.warning(f"rgb2ycbcr: Ожидался тип uint8, получен {frame_bgr.dtype}. Попытка конвертации.")
        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_ycbcr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
        y_plane = frame_ycbcr[:, :, 0].astype(np.float32) / 255.0
        return y_plane
    except cv2.error as e:
        logging.error(f"Ошибка конвертации BGR->YCbCr: {e}")
        h, w, _ = frame_bgr.shape
        return np.full((h, w), 0.5, dtype=np.float32)

def dtcwt_transform(y_plane: np.ndarray) -> Pyramid:
    """Применяет DTCWT-преобразование к яркостному каналу (один уровень)."""
    t = Transform2d()
    pyramid = t.forward(y_plane.astype(np.float32), nlevels=1)
    return pyramid

# Важно: ИСПОЛЬЗОВАТЬ ТУ ЖЕ ВЕРСИЮ, ЧТО И В ЭМБЕДДЕРЕ!
def ring_division(lowpass_subband: np.ndarray, n_rings: int = 8) -> List[List[Tuple[int, int]]]:
    """
    Векторизованное разбиение низкочастотной подполосы на кольца.
    Возвращает список координат пикселей для каждого кольца.
    """
    H, W = lowpass_subband.shape
    cx, cy = H // 2, W // 2

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)

    max_r = distances.max()
    if max_r == 0:
        delta_r = 1
    else:
        delta_r = max_r / n_rings

    ring_bins = np.linspace(0, max_r + 1e-6, n_rings + 1)
    ring_indices = np.digitize(distances, ring_bins) - 1

    rings_coords = [[] for _ in range(n_rings)]
    valid_indices = ring_indices < n_rings
    coords_all = np.column_stack((y_coords[valid_indices], x_coords[valid_indices]))
    indices_all = ring_indices[valid_indices]

    for ring_idx in range(n_rings):
        rings_coords[ring_idx] = [tuple(coord) for coord in coords_all[indices_all == ring_idx]]

    for i, coords in enumerate(rings_coords):
        if not coords:
            logging.debug(f"Кольцо {i} не содержит пикселей при извлечении.") # Debug level для экстрактора

    return rings_coords

def calculate_entropies(ring_vals: np.ndarray) -> Tuple[float, float]:
    """Вычисляет визуальную (Ev) и краевую (Ee) энтропию для значений кольца."""
    eps = 1e-12
    if ring_vals.size == 0:
        return 0.0, 0.0

    hist, bin_edges = np.histogram(ring_vals, bins=256, range=(ring_vals.min(), ring_vals.max()), density=True)
    bin_width = np.diff(bin_edges)
    probabilities = hist * bin_width[0] # предполагаем одинаковую ширину бинов

    probabilities = probabilities[probabilities > eps]
    if probabilities.size == 0:
        return 0.0, 0.0

    visual_entropy = -np.sum(probabilities * np.log2(probabilities))
    edge_entropy = -np.sum(probabilities * np.exp(1.0 - probabilities))

    return visual_entropy, edge_entropy

def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    """
    Вычисляет адаптивный коэффициент альфа на основе энтропий (статья 1).
    ИДЕНТИЧНО ЭМБЕДДЕРУ!
    """
    eps = 1e-12
    visual_entropy, edge_entropy = calculate_entropies(ring_vals)

    if abs(visual_entropy) < eps:
        entropy_ratio = 0.0
    else:
        entropy_ratio = edge_entropy / visual_entropy

    # Используем ту же логику масштабирования, что и в эмбеддере
    sigmoid_ratio = 1 / (1 + np.exp(-entropy_ratio))
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid_ratio

    logging.debug( # Используем debug level для альфы при извлечении
        f"[ADAPTIVE_ALPHA_EXTRACT] Frame={frame_number}, ring={ring_index}, "
        f"Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, final_alpha={final_alpha:.4f}"
    )
    return final_alpha

# ИСПОЛЬЗОВАТЬ ТУ ЖЕ ВЕРСИЮ, ЧТО И В ЭМБЕДДЕРЕ!
def select_embedding_ring(lowpass_subband: np.ndarray, rings_coords: List[List[Tuple[int, int]]]) -> int:
    """Выбирает индекс кольца для встраивания/извлечения (на основе макс. энергии)."""
    max_energy = -1.0
    selected_index = DEFAULT_RING_INDEX
    energies = []

    for i, coords in enumerate(rings_coords):
        if not coords:
            energies.append(0.0)
            continue
        ring_vals = np.array([lowpass_subband[r, c] for (r, c) in coords], dtype=np.float32)
        energy = np.sum(ring_vals**2)
        energies.append(energy)
        if energy > max_energy:
            max_energy = energy
            selected_index = i

    logging.debug(f"Энергии колец (извлечение): {[f'{e:.2f}' for e in energies]}. Выбрано кольцо: {selected_index}")

    # Проверка на случай пустого выбранного/дефолтного кольца (аналогично эмбеддеру)
    if not rings_coords[selected_index]:
         logging.error(f"Адаптивно выбрано пустое кольцо {selected_index} при извлечении! Используется дефолтное: {DEFAULT_RING_INDEX}")
         if not rings_coords[DEFAULT_RING_INDEX]:
              logging.error(f"Дефолтное кольцо {DEFAULT_RING_INDEX} тоже пустое при извлечении!")
              for idx, coords in enumerate(rings_coords):
                   if coords:
                        selected_index = idx
                        logging.warning(f"Найдено непустое кольцо {selected_index} вместо пустого дефолтного при извлечении.")
                        break
              else:
                   logging.critical("Все кольца пусты при извлечении! Невозможно извлечь водяной знак.")
                   raise ValueError("Не найдено ни одного непустого кольца для извлечения.")
         else:
              selected_index = DEFAULT_RING_INDEX

    return selected_index

# --- Функции извлечения ---

def extract_frame_pair(
    frame1: np.ndarray,
    frame2: np.ndarray,
    ring_index: int, # Теперь ring_index определяется снаружи
    n_rings: int = N_RINGS,
    frame_number: int = 0
) -> int:
    """Извлекает один бит из пары кадров, используя заданный ring_index."""
    try:
        Y1 = rgb2ycbcr(frame1)
        Y2 = rgb2ycbcr(frame2)

        pyr1 = dtcwt_transform(Y1)
        pyr2 = dtcwt_transform(Y2)

        L1 = pyr1.lowpass
        L2 = pyr2.lowpass

        # Разбиение на кольца (должно быть идентично эмбеддеру)
        rings1_coords = ring_division(L1, n_rings=n_rings)
        rings2_coords = ring_division(L2, n_rings=n_rings)

        # Используем ЗАДАННЫЙ ring_index
        coords_1 = rings1_coords[ring_index]
        coords_2 = rings2_coords[ring_index]

        # Проверка на пустоту кольца
        if not coords_1 or not coords_2:
            logging.error(f"Кольцо {ring_index} пустое при извлечении бита из пары {frame_number}/{frame_number+1}. Возврат бита 0 (неопределенность).")
            return 0 # Возвращаем дефолтный бит или выбрасываем исключение?

        ring_vals_1 = np.array([L1[r, c] for (r, c) in coords_1], dtype=np.float32)
        ring_vals_2 = np.array([L2[r, c] for (r, c) in coords_2], dtype=np.float32)

        # Вычисляем ожидаемую альфу ТЕМ ЖЕ СПОСОБОМ, ЧТО И ЭМБЕДДЕР
        # Используем значения ПЕРВОГО кадра для расчета, как и в эмбеддере
        alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_index, frame_number)
        eps = 1e-12
        # Вычисляем порог
        threshold = (alpha + 1.0 / (alpha + eps)) / 2.0

        # DCT и SVD
        dct1 = dct_1d(ring_vals_1)
        dct2 = dct_1d(ring_vals_2)

        # Преобразуем в вектор-столбец
        U1, S1_vals, Vt1 = svd(dct1.reshape(-1, 1), full_matrices=False)
        U2, S2_vals, Vt2 = svd(dct2.reshape(-1, 1), full_matrices=False)

        s1 = S1_vals[0] if S1_vals.size > 0 else 0.0
        s2 = S2_vals[0] if S2_vals.size > 0 else 0.0

        ratio = s1 / (s2 + eps) # Используем сингулярные значения

        logging.info(
            f"[EXTRACT_BIT] Pair={frame_number//2}, Ring={ring_index}, "
            f"s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f}, threshold={threshold:.4f}, alpha={alpha:.4f}"
        )

        # Принятие решения
        bit_extracted = 0 if ratio >= threshold else 1
        return bit_extracted

    except Exception as e:
        logging.error(f"Ошибка при извлечении бита из пары кадров {frame_number}/{frame_number+1}: {e}", exc_info=True)
        return 0 # Возвращаем 0 в случае ошибки

def extract_watermark_from_video(
    frames: List[np.ndarray],
    bit_count: int,
    n_rings: int = N_RINGS,
    adaptive_ring_selection: bool = True, # Должно соответствовать режиму встраивания
    default_ring_index: int = DEFAULT_RING_INDEX
) -> List[int]:
    """Извлекает водяной знак из видео."""
    logging.info(f"Начало извлечения {bit_count} бит водяного знака. Адаптивное кольцо: {adaptive_ring_selection}.")
    start_time = time.time()
    extracted_bits = []
    num_frames = len(frames)
    pair_count = num_frames // 2

    for i in range(pair_count):
        if len(extracted_bits) >= bit_count:
            break

        idx1 = 2 * i
        idx2 = idx1 + 1
        if idx2 >= num_frames:
             logging.warning(f"Не хватает второго кадра для пары {i}. Завершение извлечения.")
             break

        f1 = frames[idx1]
        f2 = frames[idx2]

        try:
            # --- Определение ring_index для текущей пары ---
            current_ring_index: int
            if adaptive_ring_selection:
                Y1 = rgb2ycbcr(f1)
                pyr1 = dtcwt_transform(Y1)
                L1 = pyr1.lowpass
                rings1_coords = ring_division(L1, n_rings=n_rings)
                current_ring_index = select_embedding_ring(L1, rings1_coords)
            else:
                current_ring_index = default_ring_index
            # --- ---

            # Извлечение бита с использованием определенного индекса кольца
            bit = extract_frame_pair(f1, f2, ring_index=current_ring_index, n_rings=n_rings, frame_number=idx1)
            extracted_bits.append(bit)

        except Exception as e:
             logging.error(f"Критическая ошибка при обработке пары {i} (кадры {idx1}, {idx2}): {e}", exc_info=True)
             # Можно добавить бит ошибки (например, -1) или пропустить
             # extracted_bits.append(-1)

    end_time = time.time()
    logging.info(f"Извлечение завершено. Извлечено бит: {len(extracted_bits)}. Время: {end_time - start_time:.2f} сек.")
    return extracted_bits

# --- Функция чтения видео (Идентична эмбеддеру) ---
def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Считывает видео и возвращает список кадров BGR и FPS."""
    frames = []
    fps = FPS
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Не удалось открыть видео: {video_path}")
            return frames, fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = FPS
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                frames.append(frame)
                frame_count += 1
            else:
                logging.warning(f"Пропущен неверный формат кадра #{frame_count} в {video_path}")
        cap.release()
        logging.info(f"Видео для извлечения прочитано: {video_path}, Кадров: {len(frames)}, FPS: {fps:.2f}")
    except Exception as e:
        logging.error(f"Ошибка при чтении видео {video_path}: {e}", exc_info=True)
    if not frames:
        logging.error(f"Не удалось прочитать кадры из {video_path}")
    return frames, fps


# --- Главная функция ---
def main():
    video_path = "watermarked_output.mp4" # Путь к видео с водяным знаком

    # 1. Чтение видео
    frames, _ = read_video(video_path) # FPS здесь не так важен
    if not frames:
        return
    logging.info(f"Прочитано кадров для извлечения: {len(frames)}")

    # 2. Параметры извлечения (должны соответствовать встраиванию!)
    expected_bit_count = 64 # Укажите, сколько бит было встроено
    use_adaptive_ring = True # Установите True/False в зависимости от того, как был встроен ВЗ

    # 3. Извлечение водяного знака
    extracted_bits = extract_watermark_from_video(
        frames=frames,
        bit_count=expected_bit_count,
        n_rings=N_RINGS,
        adaptive_ring_selection=use_adaptive_ring,
        default_ring_index=DEFAULT_RING_INDEX
    )

    # 4. Вывод результата
    logging.info(f"Извлеченный водяной знак ({len(extracted_bits)} бит): {''.join(map(str, extracted_bits))}")
    print(f"Извлеченный водяной знак ({len(extracted_bits)} бит): {''.join(map(str, extracted_bits))}")

    # Дополнительно: Сравнение с оригинальным (если он известен)
    # original_watermark = [...] # Загрузите или сгенерируйте оригинальный ВЗ
    # if len(extracted_bits) == len(original_watermark):
    #     ber = sum(1 for i in range(len(extracted_bits)) if extracted_bits[i] != original_watermark[i]) / len(extracted_bits)
    #     logging.info(f"Bit Error Rate (BER): {ber:.4f}")
    #     print(f"Bit Error Rate (BER): {ber:.4f}")
    # else:
    #     logging.warning("Длина извлеченного и оригинального ВЗ не совпадает для расчета BER.")


if __name__ == "__main__":
    main()