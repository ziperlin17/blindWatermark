# -*- coding: utf-8 -*-
# Файл: extractor.py (улучшенная версия)

import cv2
import numpy as np
import logging
import time
import json
import os
import imagehash
from PIL import Image
from scipy.fftpack import dct  # idct не нужен для экстрактора
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any

# --- Константы (ДОЛЖНЫ СОВПАДАТЬ С EMBEDDER!) ---
# Настройки адаптивной альфы (Используются для расчета порога)
LAMBDA_PARAM: float = 0.04
ALPHA_MIN: float = 1.005
ALPHA_MAX: float = 1.1

# Общие настройки
N_RINGS: int = 8
DEFAULT_RING_INDEX: int = 4
FPS: int = 30  # Используется только для информации в логах
LOG_FILENAME: str = 'watermarking_extract.log'  # Имя лог-файла для экстрактора
SELECTED_RINGS_FILE: str = 'selected_rings.json'  # Файл для чтения выбранных колец
ORIGINAL_WATERMARK_FILE: str = 'original_watermark.txt' # Файл для чтения исходного ВЗ

# --- Настройки Адаптивности (ВАЖНО: ДОЛЖНЫ СООТВЕТСТВОВАТЬ РЕЖИМУ EMBEDDING!) ---
RING_SELECTION_METHOD: str = 'deterministic' # Должен совпадать с эмбеддером! ('adaptive', 'deterministic', 'keypoint', 'multi_ring', 'fixed')
RING_SELECTION_METRIC: str = 'entropy' # Должен совпадать с эмбеддером, если RING_SELECTION_METHOD='adaptive'
USE_CHROMINANCE_EMBEDDING: bool = False # Должен совпадать с эмбеддером!
EMBED_COMPONENT: int = 1 # Должен совпадать с эмбеддером! (0=Y, 1=Cr, 2=Cb)
NUM_RINGS_TO_USE: int = 3 # Должен совпадать с эмбеддером, если RING_SELECTION_METHOD='multi_ring'
USE_SAVED_RINGS: bool = True  # Использовать сохраненные кольца из файла (Рекомендуется True для стабильности)

# --- Настройка Видео Входа (Ожидаемое расширение) ---
# Должно соответствовать OUTPUT_EXTENSION из embedder.py
INPUT_EXTENSION: str = '.avi'

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='w',  # 'w' - Перезаписывать лог при каждом запуске ЭКСТРАКТОРА
    level=logging.INFO, # Ставим INFO по умолчанию
    # level=logging.DEBUG, # Раскомментируйте для максимальной детализации
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
logging.info(f"--- Запуск Скрипта Извлечения (Улучшенная версия) ---")
logging.info(
    f"Ожидаемые настройки эмбеддера: Метод выбора кольца='{RING_SELECTION_METHOD}', Метрика='{RING_SELECTION_METRIC if RING_SELECTION_METHOD == 'adaptive' else 'N/A'}', N_RINGS={N_RINGS}")
logging.info(f"Параметры Альфа (ожидаемые): MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(
    f"Компонент извлечения: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, Использовать сохраненные кольца: {USE_SAVED_RINGS}")


# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
# ВАЖНО: Эти функции должны быть ИДЕНТИЧНЫ соответствующим функциям в embedder.py
# РЕКОМЕНДАЦИЯ: Вынести их в watermarking_utils.py и импортировать здесь.
# ============================================================

def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    # logging.debug(f"Input shape: {signal_1d.shape}")
    result = dct(signal_1d, type=2, norm='ortho')
    # logging.debug(f"Output shape: {result.shape}")
    return result

# --- dtcwt_transform ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции dtcwt_transform из embedder.py
def dtcwt_transform(y_plane: np.ndarray, frame_number: int = -1) -> Optional[Pyramid]:
    """Выполняет прямое DTCWT преобразование с обработкой нечетных размеров."""
    # ... (код функции dtcwt_transform из embedder.py) ...
    func_start_time = time.time()
    logging.debug(f"[F:{frame_number}] Input plane shape: {y_plane.shape}, dtype: {y_plane.dtype}, Min: {np.min(y_plane):.4f}, Max: {np.max(y_plane):.4f}")
    if np.any(np.isnan(y_plane)):
        logging.warning(f"[F:{frame_number}] NaNs detected in input plane!")

    try:
        t = Transform2d()
        rows, cols = y_plane.shape
        pad_rows = rows % 2 != 0
        pad_cols = cols % 2 != 0

        if pad_rows or pad_cols:
            logging.debug(f"[F:{frame_number}] Padding input for DTCWT: rows_pad={pad_rows}, cols_pad={pad_cols}")
            y_plane_padded = np.pad(y_plane, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        else:
            y_plane_padded = y_plane

        pyramid = t.forward(y_plane_padded.astype(np.float32), nlevels=1)

        if hasattr(pyramid, 'lowpass') and pyramid.lowpass is not None:
            lp = pyramid.lowpass
            logging.debug(
                f"[F:{frame_number}] DTCWT lowpass shape: {lp.shape}, dtype: {lp.dtype}, Min: {np.min(lp):.4f}, Max: {np.max(lp):.4f}")
            if np.any(np.isnan(lp)):
                logging.warning(f"[F:{frame_number}] NaNs detected in lowpass component!")
        else:
            logging.error(f"[F:{frame_number}] DTCWT forward did not produce a valid lowpass attribute!")
            return None

        logging.debug(f"[F:{frame_number}] DTCWT transform time: {time.time() - func_start_time:.4f}s")
        pyramid.padding_info = (pad_rows, pad_cols)
        return pyramid
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception during DTCWT transform: {e}", exc_info=True)
        return None

# --- ring_division ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции ring_division из embedder.py
def ring_division(lowpass_subband: np.ndarray, n_rings: int = N_RINGS, frame_number: int = -1) -> List[List[Tuple[int, int]]]:
    """Векторизованное разбиение низкочастотной подполосы на кольца."""
    # ... (код функции ring_division из embedder.py) ...
    func_start_time = time.time()
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Input is not a 2D numpy array! Type: {type(lowpass_subband)}")
        return [[] for _ in range(n_rings)]
    H, W = lowpass_subband.shape
    if H < 2 or W < 2:
        logging.error(f"[F:{frame_number}] Subband dimensions too small: H={H}, W={W}. Cannot create rings.")
        return [[] for _ in range(n_rings)]
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
    min_dist, max_dist = np.min(distances), np.max(distances)
    if max_dist < 1e-6:
        logging.warning(f"[F:{frame_number}] Max distance is near zero ({max_dist:.4f}). Forcing n_rings=1.")
        ring_bins = np.array([0.0, 1.0])
        n_rings_eff = 1
    else:
        delta_r = max_dist / n_rings
        ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1)
        n_rings_eff = n_rings
    if len(ring_bins) < 2:
        logging.error(f"[F:{frame_number}] Invalid ring bins generated!")
        return [[] for _ in range(n_rings)]
    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    rings_coords_tuples = [[] for _ in range(n_rings)]
    pixel_counts = np.zeros(n_rings, dtype=int)
    for ring_idx in range(n_rings_eff):
        coords_for_ring_np = np.argwhere(ring_indices == ring_idx)
        rings_coords_tuples[ring_idx] = [tuple(coord) for coord in coords_for_ring_np]
        pixel_counts[ring_idx] = len(rings_coords_tuples[ring_idx])
    total_pixels_in_rings = np.sum(pixel_counts)
    total_pixels_in_subband = H * W
    if total_pixels_in_rings != total_pixels_in_subband:
        logging.warning(f"[F:{frame_number}] Pixel count mismatch! Rings: {total_pixels_in_rings}, Subband: {total_pixels_in_subband}.")
    logging.debug(f"[F:{frame_number}] Ring division time: {time.time() - func_start_time:.4f}s")
    return rings_coords_tuples


# --- calculate_entropies ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции calculate_entropies из embedder.py
def calculate_entropies(ring_vals: np.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    """Вычисляет визуальную и краевую энтропию для массива значений кольца."""
    # ... (код функции calculate_entropies из embedder.py) ...
    eps = 1e-12
    if ring_vals.size == 0:
        logging.debug(f"[F:{frame_number}, R:{ring_index}] calculate_entropies called with empty array.")
        return 0.0, 0.0
    ring_vals_clipped = np.clip(ring_vals, 0.0, 1.0)
    hist, _ = np.histogram(ring_vals_clipped, bins=256, range=(0.0, 1.0), density=False)
    total_count = ring_vals_clipped.size
    if total_count == 0: return 0.0, 0.0
    probabilities = hist / total_count
    probabilities = probabilities[probabilities > eps]
    if probabilities.size == 0: return 0.0, 0.0
    visual_entropy = -np.sum(probabilities * np.log2(probabilities))
    edge_entropy = -np.sum(probabilities * np.exp(1.0 - probabilities))
    return visual_entropy, edge_entropy

# --- compute_adaptive_alpha_entropy ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции compute_adaptive_alpha_entropy из embedder.py
def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    """Вычисляет адаптивную альфу на основе энтропии и локальной вариации кольца."""
    # ... (код функции compute_adaptive_alpha_entropy из embedder.py) ...
    func_start_time = time.time()
    eps = 1e-12
    if ring_vals.size == 0:
        logging.warning(f"[F:{frame_number}, R:{ring_index}] compute_adaptive_alpha called with empty ring_vals. Returning ALPHA_MIN.")
        return ALPHA_MIN
    visual_entropy, edge_entropy = calculate_entropies(ring_vals, frame_number, ring_index)
    local_variance = np.var(ring_vals)
    texture_factor = 1.0 / (1.0 + np.clip(local_variance, 0, 1) * 10.0)
    if abs(visual_entropy) < eps:
        entropy_ratio = 0.0
        logging.warning(f"[F:{frame_number}, R:{ring_index}] Visual entropy is near zero (Ev={visual_entropy:.4f}). Setting ratio to 0.")
    else:
        entropy_ratio = edge_entropy / visual_entropy
    sigmoid_ratio = 1.0 / (1.0 + np.exp(-entropy_ratio)) * texture_factor
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid_ratio
    final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)
    logging.info(f"[F:{frame_number}, R:{ring_index}] Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, " +
                 f"Var={local_variance:.4f}, Texture={texture_factor:.4f}, " +
                 f"Ratio={entropy_ratio:.4f}, Sigmoid={sigmoid_ratio:.4f} -> final_alpha={final_alpha:.4f}")
    return final_alpha

# --- deterministic_ring_selection ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции deterministic_ring_selection из embedder.py
def deterministic_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    """Выбирает кольцо детерминированно на основе перцептуального хеша кадра."""
    # ... (код функции deterministic_ring_selection из embedder.py) ...
    func_start_time = time.time()
    try:
        small_frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        if small_frame.ndim == 3 and small_frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        elif small_frame.ndim == 2:
             gray_frame = small_frame
        else:
            logging.error(f"[F:{frame_number}] Invalid frame dimension for hashing: {small_frame.ndim}")
            return DEFAULT_RING_INDEX
        pil_img = Image.fromarray(gray_frame)
        phash = imagehash.phash(pil_img)
        hash_str = str(phash)
        if not hash_str:
             logging.warning(f"[F:{frame_number}] Perceptual hash resulted in empty string. Using default ring.")
             return DEFAULT_RING_INDEX
        try:
            hash_int = int(hash_str, 16)
        except ValueError:
             logging.error(f"[F:{frame_number}] Could not convert hash '{hash_str}' to integer. Using default ring.")
             return DEFAULT_RING_INDEX
        selected_ring = hash_int % n_rings
        logging.info(
            f"[F:{frame_number}] Deterministic ring selection: hash={hash_str}, selected_ring={selected_ring}, time={time.time() - func_start_time:.4f}s")
        return selected_ring
    except cv2.error as e:
        logging.error(f"[F:{frame_number}] OpenCV error in deterministic_ring_selection: {e}", exc_info=False)
        return DEFAULT_RING_INDEX
    except Exception as e:
        logging.error(f"[F:{frame_number}] Error in deterministic_ring_selection: {e}", exc_info=True)
        return DEFAULT_RING_INDEX

# --- keypoint_based_ring_selection ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции keypoint_based_ring_selection из embedder.py
def keypoint_based_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    """Выбирает кольцо на основе среднего положения ключевых точек FAST."""
    # ... (код функции keypoint_based_ring_selection из embedder.py) ...
    func_start_time = time.time()
    try:
        if frame.ndim == 3 and frame.shape[2] == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2: gray = frame
        else: logging.error(f"[F:{frame_number}] Invalid frame dimension for keypoint detection: {frame.ndim}"); return DEFAULT_RING_INDEX
        fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        keypoints = fast.detect(gray, None)
        if not keypoints: logging.warning(f"[F:{frame_number}] No FAST keypoints detected, using default ring {DEFAULT_RING_INDEX}"); return DEFAULT_RING_INDEX
        num_keypoints = len(keypoints)
        x_sum = sum(kp.pt[0] for kp in keypoints); y_sum = sum(kp.pt[1] for kp in keypoints)
        x_avg = x_sum / num_keypoints; y_avg = y_sum / num_keypoints
        h, w = gray.shape[:2]
        x_norm = x_avg / w if w > 0 else 0.5; y_norm = y_avg / h if h > 0 else 0.5
        dist_from_center = np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
        selected_ring = int((dist_from_center / 0.5) * n_rings) if dist_from_center > 0 else 0
        selected_ring = max(0, min(selected_ring, n_rings - 1))
        logging.info(f"[F:{frame_number}] Keypoint-based selection: keypoints={num_keypoints}, avg=({x_avg:.1f},{y_avg:.1f}), norm=({x_norm:.2f},{y_norm:.2f}), dist={dist_from_center:.3f}, selected_ring={selected_ring}, time={time.time() - func_start_time:.4f}s")
        return selected_ring
    except cv2.error as e: logging.error(f"[F:{frame_number}] OpenCV error in keypoint_based_ring_selection: {e}", exc_info=False); return DEFAULT_RING_INDEX
    except Exception as e: logging.error(f"[F:{frame_number}] Error in keypoint_based_ring_selection: {e}", exc_info=True); return DEFAULT_RING_INDEX

# --- select_embedding_ring ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции select_embedding_ring из embedder.py
def select_embedding_ring(
        lowpass_subband: np.ndarray, rings_coords: List[List[Tuple[int, int]]],
        metric: str = RING_SELECTION_METRIC, frame_number: int = -1
) -> int:
    """Выбирает наиболее подходящее кольцо для встраивания на основе метрики."""
    func_start_time = time.time()
    best_metric_value = -float('inf')
    selected_index = DEFAULT_RING_INDEX
    metric_values = []
    n_rings_available = len(rings_coords)
    known_metrics = ['entropy', 'energy', 'variance', 'mean_abs_dev'] # Список известных метрик

    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Invalid lowpass_subband input!")
        return DEFAULT_RING_INDEX

    # Проверяем метрику один раз перед циклом
    is_metric_unknown = metric not in known_metrics
    if is_metric_unknown:
         logging.warning(f"[F:{frame_number}] Unknown metric '{metric}', defaulting to 'entropy' for calculations.")
         metric_to_use = 'entropy' # Используем эту переменную для вычислений
    else:
         metric_to_use = metric # Используем переданную метрику

    logging.debug(f"[F:{frame_number}] Selecting ring using calculation metric: '{metric_to_use}' (Original request: '{metric}')")

    for i, coords in enumerate(rings_coords):
        current_metric = -float('inf')
        if not coords:
            metric_values.append(current_metric)
            continue
        try:
            # Извлечение значений кольца (как и раньше)
            max_r, max_c = lowpass_subband.shape
            valid_coords_indices = [(r, c) for r, c in coords if 0 <= r < max_r and 0 <= c < max_c]
            if not valid_coords_indices:
                 metric_values.append(current_metric)
                 continue
            rows, cols = zip(*valid_coords_indices)
            ring_vals = lowpass_subband[rows, cols].astype(np.float32)
            if ring_vals.size == 0:
                 metric_values.append(current_metric)
                 continue

            # Вычисление метрики на основе metric_to_use
            if metric_to_use == 'entropy':
                visual_entropy, _ = calculate_entropies(ring_vals, frame_number, i)
                current_metric = visual_entropy
            elif metric_to_use == 'energy':
                current_metric = np.sum(ring_vals ** 2)
            elif metric_to_use == 'variance':
                current_metric = np.var(ring_vals)
            elif metric_to_use == 'mean_abs_dev':
                current_metric = np.mean(np.abs(ring_vals - 0.5))
            # else: не нужен, так как metric_to_use всегда будет известной метрикой

            metric_values.append(current_metric)
            if current_metric > best_metric_value:
                 best_metric_value = current_metric
                 selected_index = i
        except Exception as e:
            logging.error(f"[F:{frame_number}, R:{i}] Error calculating metric '{metric_to_use}' for ring: {e}", exc_info=False)
            metric_values.append(-float('inf'))

    metric_log_str = ", ".join([f"{i}:{v:.4f}" if v > -float('inf') else f"{i}:Err" for i, v in enumerate(metric_values)])
    logging.debug(f"[F:{frame_number}] Ring metrics calculated using '{metric_to_use}': [{metric_log_str}]")
    logging.info(f"[F:{frame_number}] Adaptive ring selection result: Ring={selected_index} (Value: {best_metric_value:.4f})")

    # --- Проверка валидности выбранного кольца (остается без изменений) ---
    if not (0 <= selected_index < n_rings_available and rings_coords[selected_index]):
        logging.error(f"[F:{frame_number}] Selected ring {selected_index} is invalid or empty! Checking default {DEFAULT_RING_INDEX}.")
        if 0 <= DEFAULT_RING_INDEX < n_rings_available and rings_coords[DEFAULT_RING_INDEX]:
            selected_index = DEFAULT_RING_INDEX
            logging.warning(f"[F:{frame_number}] Using default ring {selected_index}.")
        else:
            logging.warning(f"[F:{frame_number}] Default ring {DEFAULT_RING_INDEX} also invalid/empty. Searching for first available...")
            found_non_empty = False
            for idx, coords_check in enumerate(rings_coords):
                if coords_check:
                    selected_index = idx
                    logging.warning(f"[F:{frame_number}] Using first non-empty ring {selected_index}.")
                    found_non_empty = True
                    break
            if not found_non_empty:
                logging.critical(f"[F:{frame_number}] All rings are empty!")
                raise ValueError(f"Frame {frame_number}: All rings are empty, cannot select a ring.")

    logging.debug(f"[F:{frame_number}] Ring selection process time: {time.time() - func_start_time:.4f}s")
    return selected_index
# --- load_saved_rings ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции load_saved_rings из embedder.py (если она там есть, или добавьте ее)
# Если ее нет в embedder.py, вот реализация:
def load_saved_rings() -> List[int]:
    """Загружает сохраненные индексы колец из файла JSON."""
    if not os.path.exists(SELECTED_RINGS_FILE):
        logging.warning(f"Saved rings file '{SELECTED_RINGS_FILE}' not found.")
        return []
    try:
        with open(SELECTED_RINGS_FILE, 'r') as f:
            rings = json.load(f)
        if isinstance(rings, list) and all(isinstance(r, int) for r in rings):
            logging.info(f"Loaded {len(rings)} saved rings from {SELECTED_RINGS_FILE}")
            return rings
        else:
            logging.error(f"Invalid format in {SELECTED_RINGS_FILE}. Expected a list of integers.")
            return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {SELECTED_RINGS_FILE}: {e}")
        return []
    except IOError as e:
        logging.error(f"Error reading saved rings file {SELECTED_RINGS_FILE}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading saved rings: {e}", exc_info=True)
        return []

# ============================================================
# --- Функции Работы с Видео (Только Чтение) ---
# ============================================================

# --- read_video ---
# ВСТАВЬТЕ СЮДА ТОЧНУЮ КОПИЮ функции read_video из embedder.py
def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Считывает видео и возвращает список кадров BGR и FPS."""
    # ... (код функции read_video из embedder.py) ...
    func_start_time = time.time()
    logging.info(f"Reading video from: {video_path}")
    frames = []; fps = float(FPS)
    cap = None; expected_height, expected_width = -1, -1
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed to open video: {video_path}"); return frames, fps
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        if fps_read > 0: fps = float(fps_read); logging.info(f"Detected FPS: {fps:.2f}")
        else: logging.warning(f"Failed to get FPS. Using default: {fps}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, ~{frame_count_prop} frames")
        expected_height, expected_width = height, width
        frame_index = 0; read_count = 0; none_frame_count = 0; invalid_shape_count = 0
        while True:
            ret, frame = cap.read()
            frame_number_log = frame_index + 1
            if not ret: logging.info(f"End of stream indicated by cap.read() after reading {read_count} frames (index {frame_index})."); break
            if frame is None: logging.warning(f"Received None frame at index {frame_index}. Skipping."); none_frame_count += 1; frame_index += 1; continue
            if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
                 current_h, current_w = frame.shape[:2]
                 if current_h == expected_height and current_w == expected_width: frames.append(frame); read_count += 1;
                 else: logging.warning(f"Frame {frame_number_log} shape {(current_h, current_w)} != expected ({expected_height},{expected_width}). Skipping."); invalid_shape_count += 1
            else: logging.warning(f"Frame {frame_number_log} is not valid BGR image (shape: {frame.shape}, dtype: {frame.dtype}). Skipping."); invalid_shape_count += 1
            frame_index += 1
            if frame_index > frame_count_prop * 1.5 and frame_count_prop > 0: logging.error(f"Read too many frames ({frame_index} vs {frame_count_prop}). Stopping."); break
        logging.info(f"Finished reading. Valid frames: {len(frames)}. Total processed: {frame_index}. Skipped None: {none_frame_count}, Invalid shape: {invalid_shape_count}")
    except Exception as e: logging.error(f"Exception during video reading: {e}", exc_info=True)
    finally:
        if cap is not None and cap.isOpened(): cap.release(); logging.debug("Video capture released.")
    if not frames: logging.error(f"No valid frames were read from {video_path}")
    logging.debug(f"Read video time: {time.time() - func_start_time:.4f}s")
    return frames, fps


# ============================================================
# --- ЛОГИКА ИЗВЛЕЧЕНИЯ (Extract) ---
# ============================================================

def extract_frame_pair(
        frame1: np.ndarray, frame2: np.ndarray, ring_index: int,
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    """Извлекает один бит из пары кадров, используя ЗАДАННЫЙ ring_index."""
    func_start_time = time.time()
    pair_num_log = frame_number // 2
    logging.debug(f"--- Extract Start: Pair {pair_num_log} (Frame {frame_number}), Target Ring: {ring_index} ---")
    try:
        # 1. Проверка входных данных
        if frame1 is None or frame2 is None:
             logging.error(f"[P:{pair_num_log}, F:{frame_number}] Input frame is None.")
             return None
        if frame1.shape != frame2.shape:
             logging.error(f"[P:{pair_num_log}, F:{frame_number}] Frame shapes mismatch: {frame1.shape} vs {frame2.shape}")
             return None
        # Проверка типа данных перед конвертацией
        if frame1.dtype != np.uint8: frame1 = np.clip(frame1, 0, 255).astype(np.uint8); logging.warning(f"[P:{pair_num_log}] Frame 1 type was {frame1.dtype}, converted to uint8.")
        if frame2.dtype != np.uint8: frame2 = np.clip(frame2, 0, 255).astype(np.uint8); logging.warning(f"[P:{pair_num_log}] Frame 2 type was {frame2.dtype}, converted to uint8.")

        # 2. Цвет -> YCrCb
        try:
            frame1_ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
            frame2_ycrcb = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)
        except cv2.error as e:
            logging.error(f"[P:{pair_num_log}, F:{frame_number}] Color conversion failed: {e}")
            return None

        # 3. Выбор компонента и нормализация
        comp_name = ['Y', 'Cr', 'Cb'][embed_component]
        logging.debug(f"[P:{pair_num_log}] Using {comp_name} component for extraction")
        try:
            comp1 = frame1_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
            comp2 = frame2_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
        except IndexError:
             logging.error(f"[P:{pair_num_log}, F:{frame_number}] Invalid embed_component index: {embed_component}")
             return None

        # 4. DTCWT
        pyr1 = dtcwt_transform(comp1, frame_number=frame_number)
        pyr2 = dtcwt_transform(comp2, frame_number=frame_number + 1)
        if pyr1 is None or pyr2 is None or pyr1.lowpass is None or pyr2.lowpass is None:
            logging.error(f"[P:{pair_num_log}, F:{frame_number}] DTCWT failed during extraction.")
            return None
        L1 = pyr1.lowpass # Не копируем, только читаем
        L2 = pyr2.lowpass

        # 5. Получение координат НУЖНОГО кольца
        rings1_coords = ring_division(L1, n_rings=n_rings, frame_number=frame_number)
        rings2_coords = ring_division(L2, n_rings=n_rings, frame_number=frame_number + 1)

        # Проверка валидности целевого кольца
        if not (0 <= ring_index < n_rings and ring_index < len(rings1_coords) and ring_index < len(rings2_coords)):
            logging.error(f"[P:{pair_num_log}, F:{frame_number}] Invalid target ring_index {ring_index} (max is {n_rings-1}).")
            return None
        coords_1 = rings1_coords[ring_index]
        coords_2 = rings2_coords[ring_index]
        if not coords_1 or not coords_2:
             logging.error(f"[P:{pair_num_log}, F:{frame_number}] Target ring {ring_index} is empty in one or both frames during extraction.")
             return None

        # 6. Извлечение значений и расчет альфы/порога
        try:
             # Используем multi-indexing
             rows1, cols1 = zip(*coords_1)
             ring_vals_1 = L1[rows1, cols1].astype(np.float32)
             rows2, cols2 = zip(*coords_2)
             ring_vals_2 = L2[rows2, cols2].astype(np.float32)
        except IndexError:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] IndexError extracting ring values.", exc_info=False)
            return None
        except Exception as e:
             logging.error(f"[P:{pair_num_log}, R:{ring_index}] Error extracting ring values: {e}", exc_info=False)
             return None

        logging.debug(f"[P:{pair_num_log}, R:{ring_index}] Extracted {len(ring_vals_1)} vals (F1), {len(ring_vals_2)} vals (F2).")
        if ring_vals_1.size == 0 or ring_vals_2.size == 0:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] Extracted empty ring values after validation.")
            return None

        # ВАЖНО: Используем ту же функцию расчета альфы, что и эмбеддер,
        # и значения ПЕРВОГО кадра для консистентности!
        alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_index, frame_number)
        eps = 1e-12
        # Порог для сравнения отношения s1/s2
        threshold = (alpha + 1.0 / (alpha + eps)) / 2.0
        logging.debug(f"[P:{pair_num_log}, R:{ring_index}] Using alpha={alpha:.4f}, threshold={threshold:.4f}")

        # 7. DCT, SVD, Вычисление отношения
        dct1 = dct_1d(ring_vals_1)
        dct2 = dct_1d(ring_vals_2)
        try:
            # Вычисляем только сингулярные значения для скорости
            S1_vals = svd(dct1.reshape(-1, 1), compute_uv=False)
            S2_vals = svd(dct2.reshape(-1, 1), compute_uv=False)
        except np.linalg.LinAlgError as e:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] SVD failed during extraction: {e}.")
            return None

        s1 = S1_vals[0] if S1_vals.size > 0 else 0.0
        s2 = S2_vals[0] if S2_vals.size > 0 else 0.0

        # Защита от деления на ноль
        ratio = s1 / (s2 + eps)

        # 8. Принятие решения
        # Если s1/s2 >= threshold, то бит 0, иначе 1
        bit_extracted = 0 if ratio >= threshold else 1
        logging.info(
            f"[P:{pair_num_log}, R:{ring_index}] s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f} vs threshold={threshold:.4f} -> Extracted Bit={bit_extracted}")

        total_pair_time = time.time() - func_start_time
        logging.debug(f"--- Extract Finish: Pair {pair_num_log}. Total Time: {total_pair_time:.4f} sec ---")
        return bit_extracted

    except Exception as e:
        # Ловим любые другие ошибки внутри функции
        logging.error(f"!!! UNHANDLED EXCEPTION in extract_frame_pair (Pair {pair_num_log}, Frame {frame_number}): {e}", exc_info=True)
        return None


# --- НОВАЯ Функция для извлечения с использованием нескольких колец ---
def extract_frame_pair_multi_ring(
        frame1: np.ndarray, frame2: np.ndarray, ring_indices: List[int],
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    """Извлекает один бит из пары кадров, используя несколько колец и голосование."""
    pair_num_log = frame_number // 2
    if not ring_indices:
        logging.error(f"[P:{pair_num_log}] No ring indices provided for multi-ring extraction")
        return None

    logging.info(f"[P:{pair_num_log}] Multi-ring extraction started for rings: {ring_indices}")
    bits = []
    # Извлекаем бит из каждого заданного кольца
    for ring_idx in ring_indices:
        # Вызываем стандартную функцию извлечения для одного кольца
        bit = extract_frame_pair(frame1, frame2, ring_idx, n_rings, frame_number, embed_component)
        if bit is not None:
            bits.append(bit)
            logging.debug(f"[P:{pair_num_log}] Multi-ring: Extracted bit={bit} from ring {ring_idx}")
        else:
             logging.warning(f"[P:{pair_num_log}] Multi-ring: Failed to extract bit from ring {ring_idx}. Skipping vote.")

    # Если не удалось извлечь ни одного бита
    if not bits:
        logging.error(f"[P:{pair_num_log}] Multi-ring: Failed to extract any bits from rings {ring_indices}")
        return None

    # Голосование большинством
    zeros = bits.count(0)
    ones = bits.count(1)
    # Если голосов поровну, можно выбрать 0 (или 1, или вернуть None - зависит от стратегии)
    final_bit = 0 if zeros >= ones else 1

    logging.info(f"[P:{pair_num_log}] Multi-ring extraction result: votes: 0={zeros}, 1={ones} -> final_bit={final_bit}")
    return final_bit


def extract_watermark_from_video(
        frames: List[np.ndarray], bit_count: int, n_rings: int = N_RINGS,
        ring_selection_method: str = RING_SELECTION_METHOD,
        ring_selection_metric: str = RING_SELECTION_METRIC, # Используется только если selection='adaptive' И use_saved_rings=False
        default_ring_index: int = DEFAULT_RING_INDEX,
        embed_component: int = EMBED_COMPONENT,
        num_rings_to_use: int = NUM_RINGS_TO_USE, # Используется только если selection='multi_ring' И use_saved_rings=False
        use_saved_rings: bool = USE_SAVED_RINGS
) -> List[Optional[int]]:
    """Извлекает водяной знак из видео, используя заданный метод выбора кольца или сохраненные кольца."""
    logging.info(f"Starting extraction of {bit_count} bits.")
    logging.info(f"Ring Selection Method to use (if not using saved): '{ring_selection_method}'")
    logging.info(f"Embedding component: {['Y', 'Cr', 'Cb'][embed_component]}, Use saved rings: {use_saved_rings}")
    start_time = time.time()
    extracted_bits: List[Optional[int]] = [None] * bit_count # Инициализируем список нужной длины
    num_frames = len(frames)
    pair_count = num_frames // 2
    processed_pairs = 0
    error_pairs = 0

    # Загружаем сохраненные кольца, если включена опция
    saved_rings: List[int] = []
    if use_saved_rings:
        saved_rings = load_saved_rings()
        if saved_rings:
            logging.info(f"Successfully loaded {len(saved_rings)} saved rings for extraction.")
            # Проверяем, совпадает ли длина сохраненных колец с ожидаемым кол-вом бит
            if len(saved_rings) < bit_count:
                 logging.warning(f"Number of saved rings ({len(saved_rings)}) is less than expected bit count ({bit_count}). Extraction will be incomplete if dynamic selection is off.")
                 # Можно либо остановить, либо продолжить с тем, что есть
            elif len(saved_rings) > bit_count:
                 logging.warning(f"Number of saved rings ({len(saved_rings)}) is more than expected bit count ({bit_count}). Using only the first {bit_count} rings.")
                 saved_rings = saved_rings[:bit_count]
        else:
            logging.warning("Failed to load saved rings or file not found. Will determine rings dynamically based on RING_SELECTION_METHOD.")
            use_saved_rings = False # Отключаем флаг, если не удалось загрузить

    # Определяем, сколько бит МАКСИМУМ можно извлечь
    max_extractable_bits = min(pair_count, bit_count)
    if max_extractable_bits < bit_count:
        logging.warning(f"Not enough frame pairs ({pair_count}) to extract the full {bit_count} bits. Attempting to extract {max_extractable_bits} bits.")

    logging.info(f"Attempting to extract {max_extractable_bits} bits from {pair_count} available pairs.")

    # Цикл по парам кадров для извлечения битов
    for i in range(max_extractable_bits):
        idx1 = 2 * i
        idx2 = idx1 + 1
        pair_num_log = i
        logging.debug(f"Processing pair {pair_num_log} (Frames {idx1}, {idx2})")

        # Проверка наличия кадров (на всякий случай)
        if idx2 >= num_frames or frames[idx1] is None or frames[idx2] is None:
            logging.error(f"Frame {idx1} or {idx2} is missing or None. Skipping pair {pair_num_log}.")
            extracted_bits[i] = None # Явно записываем None
            error_pairs += 1
            continue

        f1 = frames[idx1]
        f2 = frames[idx2]
        bit = None # Результат извлечения для этой пары

        try:
            # --- Определение Кольца/Колец для извлечения ---
            target_ring_index = -1
            target_ring_indices = [] # Для multi-ring

            if use_saved_rings:
                # Используем сохраненное кольцо для этой пары
                if i < len(saved_rings):
                    saved_ring = saved_rings[i]
                    # Проверяем валидность сохраненного индекса
                    if 0 <= saved_ring < n_rings:
                        target_ring_index = saved_ring
                        target_ring_indices = [target_ring_index] # Для единообразия
                        logging.info(f"[P:{pair_num_log}] Using saved ring index: {target_ring_index}")
                    else:
                         logging.error(f"[P:{pair_num_log}] Invalid saved ring index {saved_ring} found for pair {i}. Skipping pair.")
                         extracted_bits[i] = None; error_pairs += 1; continue
                else:
                     # Эта ситуация не должна возникать из-за проверок выше, но на всякий случай
                     logging.error(f"[P:{pair_num_log}] Index {i} out of bounds for saved rings list ({len(saved_rings)}). Skipping pair.")
                     extracted_bits[i] = None; error_pairs += 1; continue
            else:
                # Динамически определяем кольцо на основе метода эмбеддера
                logging.debug(f"[P:{pair_num_log}] Determining ring dynamically using method: '{ring_selection_method}'")
                if ring_selection_method == 'deterministic':
                    target_ring_index = deterministic_ring_selection(f1, n_rings, frame_number=idx1)
                    target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'keypoint':
                    target_ring_index = keypoint_based_ring_selection(f1, n_rings, frame_number=idx1)
                    target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'multi_ring':
                    # Здесь нужно определить N колец так же, как в эмбеддере
                    # Копируем логику выбора N лучших колец по метрике из embed_frame_pair
                    try:
                        if embed_component == 0: comp1_sel = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
                        elif embed_component == 1: comp1_sel = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 1].astype(np.float32) / 255.0
                        else: comp1_sel = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 2].astype(np.float32) / 255.0
                        pyr1_sel = dtcwt_transform(comp1_sel, frame_number=idx1)
                        if pyr1_sel is None or pyr1_sel.lowpass is None: raise RuntimeError("DTCWT failed for multi-ring selection")
                        L1_sel = pyr1_sel.lowpass
                        rings_coords_sel = ring_division(L1_sel, n_rings=n_rings, frame_number=idx1)
                        metric_values = []
                        temp_metric = RING_SELECTION_METRIC
                        for ring_i, coords in enumerate(rings_coords_sel):
                             if coords:
                                 rows, cols = zip(*coords); ring_vals = L1_sel[rows, cols].astype(np.float32)
                                 if ring_vals.size > 0:
                                     if temp_metric == 'entropy': v_e, _ = calculate_entropies(ring_vals, idx1, ring_i); metric_values.append((v_e, ring_i))
                                     elif temp_metric == 'energy': metric_values.append((np.sum(ring_vals**2), ring_i))
                                     elif temp_metric == 'variance': metric_values.append((np.var(ring_vals), ring_i))
                                     elif temp_metric == 'mean_abs_dev': metric_values.append((np.mean(np.abs(ring_vals - 0.5)), ring_i))
                                     else: v_e, _ = calculate_entropies(ring_vals, idx1, ring_i); metric_values.append((v_e, ring_i))
                                 else: metric_values.append((-float('inf'), ring_i))
                             else: metric_values.append((-float('inf'), ring_i))
                        metric_values.sort(key=lambda x: x[0], reverse=True)
                        target_ring_indices = [idx for val, idx in metric_values[:num_rings_to_use] if val > -float('inf')]
                        if not target_ring_indices: target_ring_indices = [DEFAULT_RING_INDEX]; logging.warning(f"[P:{pair_num_log}] Multi-ring dynamic: No valid rings found. Using default {DEFAULT_RING_INDEX}.")
                        logging.info(f"[P:{pair_num_log}] Multi-ring dynamic selection (metric '{temp_metric}'): Target rings {target_ring_indices}")
                    except Exception as sel_err:
                        logging.error(f"[P:{pair_num_log}] Error during multi-ring dynamic selection: {sel_err}. Using default ring.")
                        target_ring_indices = [DEFAULT_RING_INDEX]
                    # target_ring_index остается -1, так как используется список target_ring_indices
                elif ring_selection_method == 'adaptive':
                    # Адаптивный выбор требует вычисления метрик
                    try:
                        if embed_component == 0: comp1_sel = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
                        elif embed_component == 1: comp1_sel = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 1].astype(np.float32) / 255.0
                        else: comp1_sel = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 2].astype(np.float32) / 255.0
                        pyr1_sel = dtcwt_transform(comp1_sel, frame_number=idx1)
                        if pyr1_sel is None or pyr1_sel.lowpass is None: raise RuntimeError("DTCWT failed for adaptive selection")
                        L1_sel = pyr1_sel.lowpass
                        rings_coords_sel = ring_division(L1_sel, n_rings=n_rings, frame_number=idx1)
                        target_ring_index = select_embedding_ring(L1_sel, rings_coords_sel, metric=ring_selection_metric, frame_number=idx1)
                        target_ring_indices = [target_ring_index]
                    except Exception as sel_err:
                         logging.error(f"[P:{pair_num_log}] Error during adaptive ring selection: {sel_err}. Using default ring.")
                         target_ring_index = DEFAULT_RING_INDEX
                         target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'fixed':
                    target_ring_index = default_ring_index
                    target_ring_indices = [target_ring_index]
                    logging.info(f"[P:{pair_num_log}] Using fixed ring index: {target_ring_index}")
                else: # Неизвестный метод
                     logging.error(f"[P:{pair_num_log}] Unknown RING_SELECTION_METHOD '{ring_selection_method}' specified. Using default ring {DEFAULT_RING_INDEX}.")
                     target_ring_index = default_ring_index
                     target_ring_indices = [target_ring_index]

            # --- Извлечение бита ---
            if ring_selection_method == 'multi_ring' and not use_saved_rings:
                # Используем функцию для multi-ring
                bit = extract_frame_pair_multi_ring(f1, f2, ring_indices=target_ring_indices, n_rings=n_rings,
                                                    frame_number=idx1, embed_component=embed_component)
            elif target_ring_index != -1:
                 # Используем функцию для одного кольца
                 bit = extract_frame_pair(f1, f2, ring_index=target_ring_index, n_rings=n_rings,
                                          frame_number=idx1, embed_component=embed_component)
            elif target_ring_indices: # Случай multi-ring с сохраненными кольцами (обрабатываем как single ring для каждого)
                 # Или если multi-ring динамически выбрал только одно кольцо
                 if len(target_ring_indices) == 1:
                      target_ring_index = target_ring_indices[0]
                      bit = extract_frame_pair(f1, f2, ring_index=target_ring_index, n_rings=n_rings,
                                               frame_number=idx1, embed_component=embed_component)
                 else: # Если сохраненные кольца использовались для multi_ring в эмбеддере
                      logging.warning(f"[P:{pair_num_log}] Saved rings used, but method is multi_ring. Performing multi-ring extraction.")
                      bit = extract_frame_pair_multi_ring(f1, f2, ring_indices=target_ring_indices, n_rings=n_rings,
                                                          frame_number=idx1, embed_component=embed_component)

            else:
                 logging.error(f"[P:{pair_num_log}] No valid target ring index or indices determined. Skipping extraction.")
                 bit = None


            # Записываем результат (None в случае ошибки)
            extracted_bits[i] = bit
            if bit is None:
                error_pairs += 1
            processed_pairs += 1

        except Exception as e:
            # Ловим любые другие ошибки на уровне обработки пары
            logging.error(f"Critical error processing pair {pair_num_log} in extract_watermark loop: {e}", exc_info=True)
            extracted_bits[i] = None
            error_pairs += 1
            processed_pairs += 1 # Считаем как обработанную пару, хоть и с ошибкой

    end_time = time.time()
    logging.info(
        f"Extraction finished. Pairs processed: {processed_pairs}/{max_extractable_bits}, Errors during extraction: {error_pairs}. Total time: {end_time - start_time:.2f} sec.")

    # Возвращаем список извлеченных битов (или None)
    return extracted_bits


def main():
    main_start_time = time.time()
    # Формируем имя входного файла на основе ожидаемого расширения
    input_video_base = "watermarked_video"
    input_video = input_video_base + INPUT_EXTENSION # Используем константу

    # --- Определение ожидаемой длины ВЗ ---
    expected_watermark_length = 0
    original_watermark_str = None
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r") as f:
                original_watermark_str = f.read().strip()
                if original_watermark_str and original_watermark_str.isdigit():
                    expected_watermark_length = len(original_watermark_str)
                    logging.info(f"Read original watermark ({expected_watermark_length} bits) from {ORIGINAL_WATERMARK_FILE}")
                else:
                    logging.error(f"Content of {ORIGINAL_WATERMARK_FILE} is not a valid binary string: '{original_watermark_str}'. Cannot determine length.")
                    original_watermark_str = None # Сбрасываем, чтобы не использовать для сравнения
        except IOError as e:
            logging.error(f"Could not read {ORIGINAL_WATERMARK_FILE}: {e}")
        except Exception as e:
             logging.error(f"Unexpected error reading {ORIGINAL_WATERMARK_FILE}: {e}")

    # Если не удалось прочитать из файла, используем значение по умолчанию (например, 64)
    if expected_watermark_length == 0:
        default_length = 64
        logging.warning(f"{ORIGINAL_WATERMARK_FILE} not found or invalid. Assuming default expected watermark length: {default_length}")
        expected_watermark_length = default_length
    # --- Конец определения длины ВЗ ---

    logging.info("--- Starting Extraction Main Process ---")
    # Проверяем существование входного файла
    if not os.path.exists(input_video):
        logging.critical(f"Input watermarked video not found: '{input_video}'. Exiting.")
        print(f"\nERROR: Input video file '{input_video}' not found.")
        return

    frames, input_fps = read_video(input_video)
    if not frames:
        logging.critical(f"Failed to read frames from watermarked video: {input_video}. Exiting.")
        return
    logging.info(f"Read {len(frames)} frames for extraction (Reported FPS: {input_fps:.2f})")

    # Извлекаем водяной знак
    # Параметры берутся из констант вверху файла
    extracted_bits_result = extract_watermark_from_video(
        frames=frames,
        bit_count=expected_watermark_length,
        n_rings=N_RINGS,
        ring_selection_method=RING_SELECTION_METHOD,
        ring_selection_metric=RING_SELECTION_METRIC,
        default_ring_index=DEFAULT_RING_INDEX,
        embed_component=EMBED_COMPONENT,
        num_rings_to_use=NUM_RINGS_TO_USE,
        use_saved_rings=USE_SAVED_RINGS
    )

    # Вывод результата
    valid_extracted_count = sum(1 for b in extracted_bits_result if b is not None)
    extracted_bits_str = "".join(str(b) if b is not None else '?' for b in extracted_bits_result)
    logging.info(f"Attempted to extract {expected_watermark_length} bits. Successfully extracted: {valid_extracted_count}")
    logging.info(f"Extracted watermark string ({len(extracted_bits_str)} bits): {extracted_bits_str}")
    print(f"\nExtraction Results:")
    print(f"  Attempted bits: {expected_watermark_length}")
    print(f"  Valid bits extracted: {valid_extracted_count}")
    print(f"  Extracted string ({len(extracted_bits_str)}): {extracted_bits_str}")

    # Сравнение с оригинальным ВЗ, если он был успешно загружен
    if original_watermark_str and len(original_watermark_str) == expected_watermark_length:
        print(f"  Original string ({len(original_watermark_str)}):  {original_watermark_str}")

        if len(extracted_bits_result) != expected_watermark_length:
             logging.warning("Length of extracted bits list != expected length. Cannot calculate exact BER.")
             print("\n  BER Calculation: Length mismatch, cannot calculate BER.")
        else:
            error_count = 0
            comparison_markers = []
            for i in range(expected_watermark_length):
                orig_bit = original_watermark_str[i]
                extr_bit = extracted_bits_result[i]
                if extr_bit is None:
                    error_count += 1 # Считаем None как ошибку
                    comparison_markers.append("?")
                elif str(extr_bit) != orig_bit:
                    error_count += 1
                    comparison_markers.append("X") # Явная ошибка
                else:
                    comparison_markers.append("=") # Совпадение
            comparison_str = "".join(comparison_markers)

            ber = error_count / expected_watermark_length if expected_watermark_length > 0 else 0
            logging.info(f"Bit Error Rate (BER): {ber:.4f} ({error_count}/{expected_watermark_length} errors)")
            print(f"\n  Comparison (X/? = Error):")
            # Печатаем сравнение блоками для читаемости
            block_size = 64
            for i in range(0, expected_watermark_length, block_size):
                 print(f"    Orig: {original_watermark_str[i:i+block_size]}")
                 print(f"    Extr: {extracted_bits_str[i:i+block_size]}")
                 print(f"    Comp: {comparison_str[i:i+block_size]}")

            print(f"\n  Bit Error Rate (BER): {ber:.4f} ({error_count} errors / {expected_watermark_length} bits)")
            if error_count == 0:
                print("  >>> WATERMARK MATCH <<<")
            else:
                print("  >>> !!! WATERMARK MISMATCH / ERRORS DETECTED !!! <<<")
    else:
        logging.warning("Original watermark string not available or length mismatch for BER calculation.")
        print("\n  Original watermark not available for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_script_time = time.time() - main_start_time
    logging.info(f"--- Total Extractor Script Time: {total_script_time:.2f} sec ---")
    print(f"\nExtraction finished. Check log: {LOG_FILENAME}")


# --- Запуск ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception in main (Extractor): {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}. Check the log file: {LOG_FILENAME}")