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
# Настройки адаптивной альфы
LAMBDA_PARAM: float = 0.04
ALPHA_MIN: float = 1.001  # Было 1.01, уменьшено для меньшей видимости
ALPHA_MAX: float = 1.05  # Было 1.20, уменьшено для меньшей видимости

# Общие настройки
N_RINGS: int = 8
DEFAULT_RING_INDEX: int = 4
FPS: int = 30  # Используется только для информации в логах
LOG_FILENAME: str = 'watermarking_extract.log'  # Имя лог-файла для экстрактора
SELECTED_RINGS_FILE: str = 'selected_rings.json'  # Файл для чтения выбранных колец

# --- Настройки Адаптивности (ВАЖНО: ДОЛЖНЫ СООТВЕТСТВОВАТЬ РЕЖИМУ EMBEDDING!) ---
RING_SELECTION_METHOD: str = 'deterministic'  # 'adaptive', 'deterministic', 'multi_ring'
RING_SELECTION_METRIC: str = 'entropy'  # 'entropy', 'energy', 'mean' (Метрика, которую использовал эмбеддер!)
USE_CHROMINANCE_EMBEDDING: bool = False  # Извлекать из компонента цветности вместо яркости
EMBED_COMPONENT: int = 0  # 0 = Y, 1 = Cr, 2 = Cb
NUM_RINGS_TO_USE: int = 3  # Количество колец для использования при multi_ring
USE_SAVED_RINGS: bool = True  # Использовать сохраненные кольца из файла

# --- Настройка Логирования ---
# Удаляем старые хендлеры, если они есть
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='w',  # 'w' - Перезаписывать лог при каждом запуске ЭКСТРАКТОРА
    level=logging.DEBUG,  # Максимальная детализация
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
logging.info(f"--- Запуск Скрипта Извлечения (Улучшенная версия) ---")
logging.info(
    f"Настройки: Метод выбора кольца='{RING_SELECTION_METHOD}', Метрика='{RING_SELECTION_METRIC if RING_SELECTION_METHOD == 'adaptive' else 'N/A'}', N_RINGS={N_RINGS}, Default Ring={DEFAULT_RING_INDEX}")
logging.info(f"Параметры Альфа (ожидаемые): LAMBDA={LAMBDA_PARAM}, MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(
    f"Компонент извлечения: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, Использовать сохраненные кольца: {USE_SAVED_RINGS}")


# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ИДЕНТИЧНЫ EMBEDDER) ---
# ============================================================

def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    # Идентично embedder.py
    logging.debug(f"Input shape: {signal_1d.shape}")
    result = dct(signal_1d, type=2, norm='ortho')
    logging.debug(f"Output shape: {result.shape}")
    return result


def dtcwt_transform(y_plane: np.ndarray) -> Optional[Pyramid]:
    # Идентично embedder.py
    func_start_time = time.time()
    logging.debug(
        f"Input Y plane shape: {y_plane.shape}, dtype: {y_plane.dtype}, Min: {np.min(y_plane):.4f}, Max: {np.max(y_plane):.4f}")
    if np.any(np.isnan(y_plane)): logging.warning("NaNs detected in input Y plane!")
    try:
        t = Transform2d()
        pyramid = t.forward(y_plane.astype(np.float32), nlevels=1)
        if hasattr(pyramid, 'lowpass') and pyramid.lowpass is not None:
            lp = pyramid.lowpass
            logging.debug(
                f"DTCWT lowpass shape: {lp.shape}, dtype: {lp.dtype}, Min: {np.min(lp):.4f}, Max: {np.max(lp):.4f}")
            if np.any(np.isnan(lp)): logging.warning("NaNs detected in lowpass component!")
        else:
            logging.error("DTCWT did not produce a valid lowpass!"); return None
        logging.debug(f"DTCWT transform time: {time.time() - func_start_time:.4f}s")
        return pyramid
    except Exception as e:
        logging.error(f"Exception during DTCWT transform: {e}", exc_info=True); return None


def ring_division(lowpass_subband: np.ndarray, n_rings: int = N_RINGS) -> List[List[Tuple[int, int]]]:
    # Идентично embedder.py
    """
    Векторизованное разбиение низкочастотной подполосы на кольца с подробным логированием.
    Гарантирует центральное расположение.
    """
    func_start_time = time.time()
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"Input is not a 2D numpy array! Type: {type(lowpass_subband)}")
        return [[] for _ in range(n_rings)]
    H, W = lowpass_subband.shape
    logging.info(f"Input shape H={H}, W={W}")
    if H < 2 or W < 2:
        logging.error(f"Subband dimensions too small: H={H}, W={W}.");
        return [[] for _ in range(n_rings)]
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    logging.debug(f"Calculated precise center: row={center_r:.2f}, col={center_c:.2f}")
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
    min_dist, max_dist = np.min(distances), np.max(distances)
    logging.debug(f"Distances: Min={min_dist:.4f}, Max={max_dist:.4f}")
    if max_dist < 1e-6:
        logging.warning(f"Max distance near zero.");
        delta_r = 1.0;
        ring_bins = np.array([0.0, 1.0]);
        n_rings_eff = 1
    else:
        delta_r = max_dist / n_rings; ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1); n_rings_eff = n_rings
    logging.debug(f"delta_r={delta_r:.4f}, ring_bins={np.round(ring_bins, 2)}")
    if len(ring_bins) < 2: logging.error("Invalid ring bins!"); return [[] for _ in range(n_rings)]
    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    min_idx, max_idx = np.min(ring_indices), np.max(ring_indices)
    logging.debug(f"Ring indices: Min={min_idx}, Max={max_idx}")
    rings_coords = [[] for _ in range(n_rings)];
    pixel_counts = np.zeros(n_rings, dtype=int)
    for ring_idx in range(n_rings_eff):
        coords_for_ring = np.argwhere(ring_indices == ring_idx)
        rings_coords[ring_idx] = [tuple(coord) for coord in coords_for_ring]
        pixel_counts[ring_idx] = len(rings_coords[ring_idx])
    logging.info(f"Pixel counts per ring (up to {n_rings_eff - 1}): {pixel_counts[:n_rings_eff]}")
    total_pixels_in_rings = np.sum(pixel_counts);
    total_pixels_in_subband = H * W
    if total_pixels_in_rings != total_pixels_in_subband: logging.warning(
        f"Pixel count mismatch! Rings:{total_pixels_in_rings}, Subband:{total_pixels_in_subband}")
    for i in range(n_rings):
        if not rings_coords[i]: logging.warning(f"Ring {i} is empty!")
    logging.debug(f"Ring division time: {time.time() - func_start_time:.4f}s")
    return rings_coords


def calculate_entropies(ring_vals: np.ndarray) -> Tuple[float, float]:
    # Идентично embedder.py
    eps = 1e-12
    if ring_vals.size == 0: return 0.0, 0.0
    hist, _ = np.histogram(ring_vals, bins=256, range=(0.0, 1.0), density=False)
    total_count = ring_vals.size
    if total_count == 0: return 0.0, 0.0
    probabilities = hist / total_count
    probabilities = probabilities[probabilities > eps]
    if probabilities.size == 0: return 0.0, 0.0
    visual_entropy = -np.sum(probabilities * np.log2(probabilities))
    edge_entropy = -np.sum(probabilities * np.exp(1.0 - probabilities))
    logging.debug(f"Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f} (Size: {ring_vals.size})")
    return visual_entropy, edge_entropy


def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    """
    Улучшенная версия функции с учетом локальных характеристик изображения
    """
    func_start_time = time.time()
    eps = 1e-12
    visual_entropy, edge_entropy = calculate_entropies(ring_vals)

    # Добавляем анализ локальной текстуры
    local_variance = np.var(ring_vals)
    local_mean = np.mean(ring_vals)

    # Коэффициент для снижения альфа в областях с высокой вариацией
    texture_factor = 1.0 / (1.0 + local_variance * 10.0)

    if abs(visual_entropy) < eps:
        entropy_ratio = 0.0
        logging.warning(f"Frame={frame_number}, ring={ring_index}, Visual entropy is near zero.")
    else:
        entropy_ratio = edge_entropy / visual_entropy

    # Модифицированная сигмоида с учетом текстуры
    sigmoid_ratio = 1 / (1 + np.exp(-entropy_ratio)) * texture_factor

    # Масштабируем результат в заданный диапазон [ALPHA_MIN, ALPHA_MAX]
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid_ratio

    # Ограничиваем на всякий случай
    final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)

    logging.info(f"Frame={frame_number}, ring={ring_index}, Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, " +
                 f"Var={local_variance:.4f}, Texture={texture_factor:.4f}, " +
                 f"Ratio={entropy_ratio:.4f}, Sigmoid={sigmoid_ratio:.4f} -> final_alpha={final_alpha:.4f}")

    logging.debug(f"Alpha calculation time: {time.time() - func_start_time:.4f}s")
    return final_alpha


# --- НОВАЯ Функция для детерминированного выбора кольца ---
def deterministic_ring_selection(frame: np.ndarray, n_rings: int) -> int:
    """
    Выбирает кольцо детерминированно на основе хеша содержимого кадра.
    """
    func_start_time = time.time()
    # Уменьшаем разрешение для стабильности хеша
    small_frame = cv2.resize(frame, (32, 32))
    # Преобразуем в оттенки серого для стабильности
    if small_frame.ndim == 3:
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Вычисляем перцептуальный хеш (более устойчив к небольшим изменениям)
    pil_img = Image.fromarray(small_frame)
    phash = imagehash.phash(pil_img)

    # Используем хеш для выбора кольца
    hash_str = str(phash)
    hash_int = int(hash_str, 16) if hash_str else 0
    selected_ring = hash_int % n_rings

    logging.info(
        f"Deterministic ring selection: hash={hash_str}, selected_ring={selected_ring}, time={time.time() - func_start_time:.4f}s")
    return selected_ring


# --- НОВАЯ Функция для выбора кольца на основе ключевых точек ---
def keypoint_based_ring_selection(frame: np.ndarray, n_rings: int) -> int:
    """
    Выбирает кольцо на основе ключевых точек изображения.
    """
    func_start_time = time.time()
    # Преобразуем в оттенки серого для детектора ключевых точек
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Используем FAST для быстрого обнаружения ключевых точек
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)

    # Если ключевых точек нет, используем кольцо по умолчанию
    if not keypoints:
        logging.warning("No keypoints detected, using default ring")
        return DEFAULT_RING_INDEX

    # Вычисляем среднее положение ключевых точек
    x_sum = sum(kp.pt[0] for kp in keypoints)
    y_sum = sum(kp.pt[1] for kp in keypoints)
    x_avg = x_sum / len(keypoints)
    y_avg = y_sum / len(keypoints)

    # Нормализуем положение к размеру изображения
    x_norm = x_avg / gray.shape[1]
    y_norm = y_avg / gray.shape[0]

    # Используем нормализованное положение для выбора кольца
    angle = np.arctan2(y_norm - 0.5, x_norm - 0.5)  # Угол от центра
    angle_norm = (angle + np.pi) / (2 * np.pi)  # Нормализуем к [0, 1]

    selected_ring = int(angle_norm * n_rings)
    selected_ring = max(0, min(selected_ring, n_rings - 1))  # Ограничиваем диапазон

    logging.info(
        f"Keypoint-based ring selection: keypoints={len(keypoints)}, selected_ring={selected_ring}, time={time.time() - func_start_time:.4f}s")
    return selected_ring


def select_embedding_ring(
        lowpass_subband: np.ndarray, rings_coords: List[List[Tuple[int, int]]],
        metric: str = RING_SELECTION_METRIC
) -> int:
    # Идентично embedder.py
    func_start_time = time.time()
    best_metric_value = -float('inf')
    selected_index = DEFAULT_RING_INDEX
    metric_values = []
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error("Invalid lowpass_subband input!");
        return DEFAULT_RING_INDEX
    logging.debug(f"Selecting ring using metric: '{metric}'")
    for i, coords in enumerate(rings_coords):
        current_metric = -float('inf')
        if not coords: metric_values.append(current_metric); continue
        try:
            valid_coords = coords
            if not valid_coords: logging.warning(f"Ring {i} coords empty."); metric_values.append(
                current_metric); continue
            ring_vals = np.array([lowpass_subband[r, c] for (r, c) in valid_coords], dtype=np.float32)
            if ring_vals.size == 0: metric_values.append(current_metric); continue
            if metric == 'entropy':
                visual_entropy, _ = calculate_entropies(ring_vals); current_metric = visual_entropy
            elif metric == 'energy':
                current_metric = np.sum(ring_vals ** 2)
            elif metric == 'mean':
                current_metric = np.abs(np.mean(ring_vals) - 0.5)
            else:
                if i == 0: logging.warning(f"Unknown metric '{metric}', using 'entropy'.")
                visual_entropy, _ = calculate_entropies(ring_vals);
                current_metric = visual_entropy
            metric_values.append(current_metric)
            if current_metric > best_metric_value: best_metric_value = current_metric; selected_index = i
        except Exception as e:
            logging.error(f"Error calculating metric for ring {i}: {e}", exc_info=False); metric_values.append(
                -float('inf'))
    metric_log_str = ", ".join([f"{i}:{v:.4f}" for i, v in enumerate(metric_values)])
    logging.debug(f"Ring metrics ({metric}): [{metric_log_str}]")
    logging.info(f"Selected ring: {selected_index} (Value: {best_metric_value:.4f})")
    # Проверка на пустые кольца
    if selected_index < 0 or selected_index >= len(rings_coords) or not rings_coords[selected_index]:
        logging.error(f"Selected ring {selected_index} invalid/empty! Checking default {DEFAULT_RING_INDEX}.")
        if 0 <= DEFAULT_RING_INDEX < len(rings_coords) and rings_coords[DEFAULT_RING_INDEX]:
            selected_index = DEFAULT_RING_INDEX; logging.warning(f"Using default ring {selected_index}.")
        else:
            logging.warning(
                f"Default ring {DEFAULT_RING_INDEX} also invalid/empty. Searching...");  # ... (код поиска) ...
            for idx, coords in enumerate(rings_coords):
                if coords: selected_index = idx; logging.warning(f"Using first non-empty ring {selected_index}."); break
            else:
                logging.critical("All rings are empty!"); raise ValueError("All rings are empty.")
    logging.debug(f"Ring selection time: {time.time() - func_start_time:.4f}s")
    return selected_index


# --- НОВАЯ Функция для загрузки сохраненных колец ---
def load_saved_rings() -> List[int]:
    """
    Загружает сохраненные кольца из файла.
    """
    if not os.path.exists(SELECTED_RINGS_FILE):
        logging.warning(f"Saved rings file {SELECTED_RINGS_FILE} not found.")
        return []

    try:
        with open(SELECTED_RINGS_FILE, 'r') as f:
            rings = json.load(f)
        logging.info(f"Loaded {len(rings)} saved rings from {SELECTED_RINGS_FILE}")
        return rings
    except Exception as e:
        logging.error(f"Error loading saved rings: {e}")
        return []


# ============================================================
# --- Функции Работы с Видео (Только Чтение) ---
# ============================================================

def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # Идентично embedder.py
    """Считывает видео и возвращает список кадров BGR и FPS."""
    func_start_time = time.time()
    logging.info(f"Reading video from: {video_path}")
    frames = []
    fps = float(FPS)  # Значение по умолчанию
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed to open video: {video_path}"); return frames, fps
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        if fps_read > 0:
            fps = float(fps_read); logging.info(f"Detected FPS: {fps:.2f}")
        else:
            logging.warning(f"Failed to get FPS. Using default: {fps}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Video resolution: {width}x{height}")
        frame_count = 0;
        none_frame_count = 0;
        invalid_shape_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: logging.info(f"End of video stream after {frame_count} frames."); break
            if frame is not None:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    if frame.shape[0] == height and frame.shape[1] == width:
                        frames.append(frame); frame_count += 1
                    else:
                        logging.warning(
                            f"Frame {frame_count} shape {frame.shape} != expected. Skipping."); invalid_shape_count += 1
                else:
                    logging.warning(f"Frame {frame_count} not 3-channel. Skipping."); invalid_shape_count += 1
            else:
                logging.warning(
                    f"Received None frame near index {frame_count + none_frame_count}."); none_frame_count += 1
        logging.info(
            f"Finished reading. Frames read: {len(frames)}. Skipped None: {none_frame_count}, Invalid shape: {invalid_shape_count}")
    except Exception as e:
        logging.error(f"Exception during video reading: {e}", exc_info=True)
    finally:
        if cap is not None and cap.isOpened(): cap.release(); logging.debug("Video capture released.")
    if not frames: logging.error(f"No valid frames read from {video_path}")
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
    logging.debug(f"--- Extract Start: Pair {frame_number // 2} (Frame {frame_number}), Target Ring: {ring_index} ---")
    try:
        # 1. Цвет -> YCrCb
        color_conv_start = time.time()
        try:
            if frame1 is None or frame2 is None: raise ValueError("Input frame is None")
            frame1_ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
            frame2_ycrcb = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)
        except (cv2.error, ValueError) as e:
            logging.error(f"Color conversion failed: {e}"); return None

        # Выбираем компонент для извлечения (должен совпадать с компонентом встраивания)
        if embed_component == 0:  # Y (яркость)
            comp1 = frame1_ycrcb[:, :, 0].astype(np.float32) / 255.0
            comp2 = frame2_ycrcb[:, :, 0].astype(np.float32) / 255.0
            logging.debug("Using Y component for extraction")
        elif embed_component == 1:  # Cr (красный-синий)
            comp1 = frame1_ycrcb[:, :, 1].astype(np.float32) / 255.0
            comp2 = frame2_ycrcb[:, :, 1].astype(np.float32) / 255.0
            logging.debug("Using Cr component for extraction")
        else:  # Cb (синий-желтый)
            comp1 = frame1_ycrcb[:, :, 2].astype(np.float32) / 255.0
            comp2 = frame2_ycrcb[:, :, 2].astype(np.float32) / 255.0
            logging.debug("Using Cb component for extraction")

        logging.debug(f"Extract Color conversion time: {time.time() - color_conv_start:.4f}s")

        # 2. DTCWT
        dtcwt_start = time.time()
        pyr1 = dtcwt_transform(comp1);
        pyr2 = dtcwt_transform(comp2)
        if pyr1 is None or pyr2 is None: logging.error("DTCWT failed."); return None
        L1 = pyr1.lowpass;
        L2 = pyr2.lowpass  # Не копируем
        logging.debug(f"Extract DTCWT time: {time.time() - dtcwt_start:.4f}s. L1 shape: {L1.shape}")

        # 3. Получение координат НУЖНОГО кольца
        ring_div_start = time.time()
        rings1_coords = ring_division(L1, n_rings=n_rings)
        rings2_coords = ring_division(L2, n_rings=n_rings)  # Пересчет для L2
        logging.debug(f"Extract Ring division time: {time.time() - ring_div_start:.4f}s")
        if not (0 <= ring_index < n_rings and ring_index < len(rings1_coords) and ring_index < len(rings2_coords)):
            logging.error(f"Invalid target ring_index {ring_index}.");
            return None
        coords_1 = rings1_coords[ring_index];
        coords_2 = rings2_coords[ring_index]
        if not coords_1 or not coords_2: logging.error(f"Target ring {ring_index} empty."); return None

        # 4. Извлечение значений и расчет альфы/порога
        alpha_calc_start = time.time()
        ring_vals_1 = np.array([L1[r, c] for (r, c) in coords_1], dtype=np.float32)
        ring_vals_2 = np.array([L2[r, c] for (r, c) in coords_2], dtype=np.float32)
        logging.debug(f"Ring {ring_index}: Extracted {len(ring_vals_1)} vals (F1), {len(ring_vals_2)} vals (F2).")
        if len(ring_vals_1) == 0: logging.error("Extracted empty ring_vals_1"); return None
        # ВАЖНО: Используем ту же функцию расчета альфы, что и эмбеддер!
        alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_index, frame_number)
        eps = 1e-12;
        threshold = (alpha + 1.0 / (alpha + eps)) / 2.0
        logging.debug(f"Alpha/Threshold calculation time: {time.time() - alpha_calc_start:.4f}s")
        logging.debug(f"Using alpha={alpha:.4f}, threshold={threshold:.4f}")

        # 5. DCT, SVD, Вычисление отношения
        svd_calc_start = time.time()
        dct1 = dct_1d(ring_vals_1);
        dct2 = dct_1d(ring_vals_2)
        try:
            U1, S1_vals, Vt1 = svd(dct1.reshape(-1, 1), full_matrices=False)
            U2, S2_vals, Vt2 = svd(dct2.reshape(-1, 1), full_matrices=False)
        except np.linalg.LinAlgError as e:
            logging.error(f"SVD failed: {e}."); return None
        s1 = S1_vals[0] if S1_vals.size > 0 else 0.0;
        s2 = S2_vals[0] if S2_vals.size > 0 else 0.0
        ratio = s1 / (s2 + eps)
        logging.debug(f"SVD calculation time: {time.time() - svd_calc_start:.4f}s")

        # 6. Принятие решения
        bit_extracted = 0 if ratio >= threshold else 1
        logging.info(
            f"s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f} vs threshold={threshold:.4f} -> Extracted Bit={bit_extracted}")

        total_pair_time = time.time() - func_start_time
        logging.debug(f"--- Extract Finish: Pair {frame_number // 2}. Total Time: {total_pair_time:.4f} sec ---")
        return bit_extracted

    except Exception as e:
        logging.error(f"!!! UNHANDLED EXCEPTION in extract_frame_pair (Frame {frame_number}): {e}", exc_info=True)
        return None  # Возвращаем None при любой ошибке


# --- НОВАЯ Функция для извлечения с использованием нескольких колец ---
def extract_frame_pair_multi_ring(
        frame1: np.ndarray, frame2: np.ndarray, ring_indices: List[int],
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    """
    Извлекает один бит из пары кадров, используя несколько колец и голосование.
    """
    if not ring_indices:
        logging.error("No ring indices provided for multi-ring extraction")
        return None

    # Извлекаем бит из каждого кольца
    bits = []
    for ring_idx in ring_indices:
        bit = extract_frame_pair(frame1, frame2, ring_idx, n_rings, frame_number, embed_component)
        if bit is not None:
            bits.append(bit)

    if not bits:
        logging.error("Failed to extract any bits from multiple rings")
        return None

    # Голосование большинством
    zeros = bits.count(0)
    ones = bits.count(1)
    final_bit = 0 if zeros >= ones else 1

    logging.info(f"Multi-ring extraction: rings={ring_indices}, votes: 0={zeros}, 1={ones}, final={final_bit}")
    return final_bit


def extract_watermark_from_video(
        frames: List[np.ndarray], bit_count: int, n_rings: int = N_RINGS,
        ring_selection_method: str = RING_SELECTION_METHOD,
        ring_selection_metric: str = RING_SELECTION_METRIC,
        default_ring_index: int = DEFAULT_RING_INDEX,
        embed_component: int = EMBED_COMPONENT,
        num_rings_to_use: int = NUM_RINGS_TO_USE,
        use_saved_rings: bool = USE_SAVED_RINGS
) -> List[Optional[int]]:
    """Извлекает водяной знак из видео."""
    logging.info(f"Starting extraction of {bit_count} bits. Ring Selection Method: {ring_selection_method}")
    logging.info(f"Embedding component: {['Y', 'Cr', 'Cb'][embed_component]}, Use saved rings: {use_saved_rings}")
    start_time = time.time()
    extracted_bits: List[Optional[int]] = []
    num_frames = len(frames)
    pair_count = num_frames // 2
    processed_pairs = 0
    error_pairs = 0

    # Загружаем сохраненные кольца, если нужно
    saved_rings = []
    if use_saved_rings:
        saved_rings = load_saved_rings()
        if saved_rings:
            logging.info(f"Using {len(saved_rings)} saved rings for extraction")
        else:
            logging.warning("No saved rings found, will determine rings dynamically")

    # Определяем, сколько бит МАКСИМУМ можно извлечь
    max_extractable_bits = min(pair_count, bit_count)
    logging.info(f"Attempting to extract {max_extractable_bits} bits from {pair_count} available pairs.")

    for i in range(max_extractable_bits):
        idx1 = 2 * i;
        idx2 = idx1 + 1
        logging.debug(f"Processing pair {i} (Frames {idx1}, {idx2})")
        # Проверка наличия кадров (хотя цикл уже ограничен)
        if idx2 >= num_frames: logging.warning(f"Not enough frames for pair {i}. Stopping early."); break
        f1 = frames[idx1];
        f2 = frames[idx2]
        if f1 is None or f2 is None: logging.error(
            f"Frame {idx1} or {idx2} is None. Skipping pair {i}."); extracted_bits.append(
            None); error_pairs += 1; continue

        try:
            # --- Определение Ring Index (Зеркально эмбеддеру) ---
            bit = None

            # Используем сохраненные кольца, если доступны
            if use_saved_rings and i < len(saved_rings):
                current_ring_index = saved_rings[i]
                logging.info(f"Using saved ring for pair {i}: {current_ring_index}")
                bit = extract_frame_pair(f1, f2, ring_index=current_ring_index, n_rings=n_rings,
                                         frame_number=idx1, embed_component=embed_component)

            # Если нет сохраненных колец или не удалось извлечь бит, используем выбранный метод
            if bit is None:
                if ring_selection_method == 'deterministic':
                    # Детерминированный выбор кольца на основе хеша содержимого
                    current_ring_index = deterministic_ring_selection(f1, n_rings)
                    logging.info(f"Deterministic ring selected for pair {i}: {current_ring_index}")
                    bit = extract_frame_pair(f1, f2, ring_index=current_ring_index, n_rings=n_rings,
                                             frame_number=idx1, embed_component=embed_component)

                elif ring_selection_method == 'keypoint':
                    # Выбор кольца на основе ключевых точек
                    current_ring_index = keypoint_based_ring_selection(f1, n_rings)
                    logging.info(f"Keypoint-based ring selected for pair {i}: {current_ring_index}")
                    bit = extract_frame_pair(f1, f2, ring_index=current_ring_index, n_rings=n_rings,
                                             frame_number=idx1, embed_component=embed_component)

                elif ring_selection_method == 'multi_ring':
                    # Используем несколько колец
                    rings_to_use = [i for i in range(1, n_rings - 1)]  # Исключаем крайние кольца
                    selected_rings = rings_to_use[:num_rings_to_use] if len(
                        rings_to_use) >= num_rings_to_use else rings_to_use
                    logging.info(f"Using multiple rings for pair {i}: {selected_rings}")
                    bit = extract_frame_pair_multi_ring(f1, f2, ring_indices=selected_rings, n_rings=n_rings,
                                                        frame_number=idx1, embed_component=embed_component)

                elif ring_selection_method == 'adaptive':
                    # Стандартный адаптивный выбор
                    logging.debug(f"Determining adaptive ring for frame {idx1}")
                    select_start = time.time()
                    # Получаем компонент -> DTCWT -> L1 только для выбора кольца
                    try:
                        if embed_component == 0:  # Y
                            comp1_for_select = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
                        elif embed_component == 1:  # Cr
                            comp1_for_select = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 1].astype(np.float32) / 255.0
                        else:  # Cb
                            comp1_for_select = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)[:, :, 2].astype(np.float32) / 255.0
                    except cv2.error as e:
                        logging.error(f"Color conversion failed for select (pair {i}): {e}")
                        extracted_bits.append(None);
                        error_pairs += 1;
                        continue

                    pyr1_for_select = dtcwt_transform(comp1_for_select)
                    if pyr1_for_select is None:
                        logging.error(f"DTCWT failed for select (pair {i}).")
                        extracted_bits.append(None);
                        error_pairs += 1;
                        continue

                    L1_for_select = pyr1_for_select.lowpass
                    rings_coords_for_select = ring_division(L1_for_select, n_rings=n_rings)
                    # Вызываем ту же функцию выбора с той же метрикой
                    current_ring_index = select_embedding_ring(L1_for_select, rings_coords_for_select,
                                                               metric=ring_selection_metric)
                    logging.info(
                        f"Adaptive ring selected for pair {i}: {current_ring_index}. Selection time: {time.time() - select_start:.4f}s")
                    bit = extract_frame_pair(f1, f2, ring_index=current_ring_index, n_rings=n_rings,
                                             frame_number=idx1, embed_component=embed_component)

                else:
                    # Используем фиксированный индекс
                    current_ring_index = default_ring_index
                    logging.info(f"Using fixed ring for pair {i}: {current_ring_index}")
                    bit = extract_frame_pair(f1, f2, ring_index=current_ring_index, n_rings=n_rings,
                                             frame_number=idx1, embed_component=embed_component)

            extracted_bits.append(bit)
            if bit is None: error_pairs += 1  # Считаем ошибку, если extract_frame_pair вернул None
            processed_pairs += 1

        except Exception as e:
            # Ловим любые другие ошибки на уровне обработки пары
            logging.error(f"Critical error processing pair {i} (extract_watermark): {e}", exc_info=True)
            extracted_bits.append(None)
            error_pairs += 1
            processed_pairs += 1  # Считаем как обработанную

    end_time = time.time()
    logging.info(
        f"Extraction finished. Pairs processed: {processed_pairs}, Errors during extraction: {error_pairs}. Total time: {end_time - start_time:.2f} sec.")
    # Дополняем список None, если извлечено меньше бит, чем ожидалось
    while len(extracted_bits) < bit_count:
        extracted_bits.append(None)
    return extracted_bits[:bit_count]  # Возвращаем ровно bit_count элементов


def main():
    main_start_time = time.time()
    # ВАЖНО: Укажите путь к видео, СОЗДАННОМУ ЭМБЕДДЕРОМ
    input_video = "watermarked_video.avi"  # Изменено расширение на .avi
    # Ожидаемая длина водяного знака (должна совпадать с длиной, встроенной эмбеддером)
    expected_watermark_length = 64  # Пример, измените на реальную длину

    logging.info("--- Starting Extraction Main Process ---")
    frames, input_fps = read_video(input_video)
    if not frames: logging.critical("Failed to read watermarked video. Exiting."); return
    logging.info(f"Read {len(frames)} frames for extraction (FPS: {input_fps:.2f})")

    # Извлекаем водяной знак
    # Важно использовать те же параметры n_rings, ring_selection_method, metric, default_ring, что и при встраивании!
    # Они берутся из констант вверху файла.
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
    extracted_bits_str = "".join(str(b) if b is not None else '?' for b in extracted_bits_result)
    logging.info(f"Attempted to extract {expected_watermark_length} bits.")
    logging.info(f"Extracted watermark ({len(extracted_bits_str)} bits): {extracted_bits_str}")
    print(f"\nExtracted ({len(extracted_bits_str)} bits): {extracted_bits_str}")

    # Опционально: Если оригинальный ВЗ известен (например, из лога эмбеддера или фиксированный)
    # original_watermark_str = "010101..."  # Замените на реальный
    # if len(original_watermark_str) == expected_watermark_length:
    #     print(f"Original:  {original_watermark_str}")
    #     valid_extracted_bits = [b for b in extracted_bits_result if b is not None]
    #     if len(valid_extracted_bits) == expected_watermark_length:
    #         error_count = sum(1 for i in range(expected_watermark_length) if original_watermark_str[i] != str(valid_extracted_bits[i]))
    #         ber = error_count / expected_watermark_length
    #         logging.info(f"Bit Error Rate (BER): {ber:.4f} ({error_count}/{expected_watermark_length} errors)")
    #         print(f"Bit Error Rate (BER): {ber:.4f} ({error_count}/{expected_watermark_length} errors)")
    #         if ber == 0.0: print(">>> WATERMARK MATCH <<<")
    #         else: print(">>> !!! WATERMARK MISMATCH !!! <<<")
    #     else:
    #         logging.warning("Number of valid extracted bits != expected length. Cannot calculate exact BER.")
    # else:
    #     logging.warning("Original watermark string not available or length mismatch for BER calculation.")

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
