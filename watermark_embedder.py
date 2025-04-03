# -*- coding: utf-8 -*-
# Файл: embedder.py (Версия N=2, 64bit ID, Max 5 repeats, Pad to k)
import cv2
import numpy as np
import random
import logging
import time
import concurrent.futures
import json
import os
import imagehash
from PIL import Image
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools  # Для кэширования
import uuid  # Используем только для типа UUID, не для генерации
from math import ceil

try:
    import bchlib

    BCHLIB_AVAILABLE = True
except ImportError:
    BCHLIB_AVAILABLE = False
import cProfile
import pstats

#Константы
LAMBDA_PARAM: float = 0.04
ALPHA_MIN: float = 1.005
ALPHA_MAX: float = 1.1
N_RINGS: int = 8
DEFAULT_RING_INDEX: int = 4
FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed.log'
MAX_WORKERS: Optional[int] = None
SELECTED_RINGS_FILE: str = 'selected_rings.json'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'

#Настройки Адаптивности и Multi-Ring ---
BITS_PER_PAIR: int = 2
RING_SELECTION_METHOD: str = 'multi_ring' if BITS_PER_PAIR > 1 else 'deterministic'
NUM_RINGS_TO_USE: int = BITS_PER_PAIR if BITS_PER_PAIR > 1 else 1
RING_SELECTION_METRIC: str = 'entropy'
USE_PERCEPTUAL_MASKING: bool = True
EMBED_COMPONENT: int = 1  # 0=Y, 1=Cr, 2=Cb

#Настройки Встраивания и ECC
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 5
MAX_PACKET_REPEATS: int = 5

#Настройка output
OUTPUT_CODEC: str = 'XVID'
OUTPUT_EXTENSION: str = '.avi'

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME, filemode='w', level=logging.INFO,
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
# logging.getLogger().setLevel(logging.DEBUG) # Раскомментировать для DEBUG логов

effective_use_ecc = USE_ECC and BCHLIB_AVAILABLE
logging.info(
    f"--- Запуск Скрипта Встраивания (Bits/Pair: {BITS_PER_PAIR}, Payload: {PAYLOAD_LEN_BYTES * 8}bit, ECC: {effective_use_ecc}, Max Repeats: {MAX_PACKET_REPEATS}) ---")
logging.info(
    f"Настройки: Метод кольца='{RING_SELECTION_METHOD}', Метрика='{RING_SELECTION_METRIC}', Колец исп.={NUM_RINGS_TO_USE}, N_RINGS_Total={N_RINGS}")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(f"Маскировка: {USE_PERCEPTUAL_MASKING}, Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
if effective_use_ecc:
    # Логируем параметры BCH ПОСЛЕ инициализации объекта bch в main
    pass
elif USE_ECC and not BCHLIB_AVAILABLE:
    logging.warning("ECC включен (USE_ECC=True), но bchlib не доступна! Встраивание без ECC.")
else:
    logging.info("ECC отключен (USE_ECC=False).")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG']:
    logging.warning(f"Используется lossy кодек '{OUTPUT_CODEC}'. Тщательно проверьте извлекаемость ВЗ!")
if BITS_PER_PAIR > 1 and RING_SELECTION_METHOD != 'multi_ring':
    logging.warning(
        f"BITS_PER_PAIR={BITS_PER_PAIR} требует RING_SELECTION_METHOD='multi_ring'. Метод будет '{RING_SELECTION_METHOD}'.")
if BITS_PER_PAIR > 1 and NUM_RINGS_TO_USE != BITS_PER_PAIR:
    logging.warning(
        f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) не равен BITS_PER_PAIR ({BITS_PER_PAIR}). Установлено NUM_RINGS_TO_USE = {BITS_PER_PAIR}")
    NUM_RINGS_TO_USE = BITS_PER_PAIR


# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
# ============================================================
# (dct_1d, idct_1d, dtcwt_transform, dtcwt_inverse,
# _ring_division_internal, get_ring_coords_cached, ring_division,
# calculate_entropies, compute_adaptive_alpha_entropy,
# deterministic_ring_selection, keypoint_based_ring_selection,
# calculate_perceptual_mask, calculate_spatial_weights_vectorized
# ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ)
def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    return dct(signal_1d, type=2, norm='ortho')


def idct_1d(coeffs_1d: np.ndarray) -> np.ndarray:
    return idct(coeffs_1d, type=2, norm='ortho')


def dtcwt_transform(y_plane: np.ndarray, frame_number: int = -1) -> Optional[Pyramid]:
    func_start_time = time.time();
    logging.debug(f"[F:{frame_number}] Input plane shape: {y_plane.shape}")
    if np.any(np.isnan(y_plane)): logging.warning(f"[F:{frame_number}] NaNs detected in input plane!")
    try:
        t = Transform2d();
        rows, cols = y_plane.shape;
        pad_rows = rows % 2 != 0;
        pad_cols = cols % 2 != 0
        if pad_rows or pad_cols:
            y_plane_padded = np.pad(y_plane, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        else:
            y_plane_padded = y_plane
        pyramid = t.forward(y_plane_padded.astype(np.float32), nlevels=1)
        if hasattr(pyramid, 'lowpass') and pyramid.lowpass is not None:
            lp = pyramid.lowpass; logging.debug(f"[F:{frame_number}] DTCWT lowpass shape: {lp.shape}");
        else:
            logging.error(f"[F:{frame_number}] DTCWT no valid lowpass!"); return None
        logging.debug(f"[F:{frame_number}] DTCWT transform time: {time.time() - func_start_time:.4f}s");
        pyramid.padding_info = (pad_rows, pad_cols);
        return pyramid
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception during DTCWT transform: {e}", exc_info=True); return None


def dtcwt_inverse(pyramid: Pyramid, frame_number: int = -1) -> Optional[np.ndarray]:
    func_start_time = time.time();
    if not isinstance(pyramid, Pyramid) or not hasattr(pyramid, 'lowpass'): logging.error(
        f"[F:{frame_number}] Invalid pyramid."); return None
    logging.debug(f"[F:{frame_number}] Input lowpass shape for inverse: {pyramid.lowpass.shape}")
    try:
        t = Transform2d();
        reconstructed_padded = t.inverse(pyramid).astype(np.float32)
        pad_rows, pad_cols = getattr(pyramid, 'padding_info', (False, False))
        if pad_rows or pad_cols:
            rows, cols = reconstructed_padded.shape; reconstructed_y = reconstructed_padded[
                                                                       :rows - pad_rows if pad_rows else rows,
                                                                       :cols - pad_cols if pad_cols else cols]
        else:
            reconstructed_y = reconstructed_padded
        logging.debug(f"[F:{frame_number}] DTCWT inverse output shape: {reconstructed_y.shape}");
        if np.any(np.isnan(reconstructed_y)): logging.warning(f"[F:{frame_number}] NaNs detected after inverse DTCWT!")
        logging.debug(f"[F:{frame_number}] DTCWT inverse time: {time.time() - func_start_time:.4f}s");
        return reconstructed_y
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception during DTCWT inverse: {e}", exc_info=True); return None


@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    func_start_time = time.time();
    H, W = subband_shape
    if H < 2 or W < 2: logging.error(f"_ring_division_internal: Subband too small: {H}x{W}."); return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0;
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2);
    min_dist, max_dist = np.min(distances), np.max(distances)
    if max_dist < 1e-6:
        ring_bins = np.array([0.0, 1.0]); n_rings_eff = 1
    else:
        ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1); n_rings_eff = n_rings
    if len(ring_bins) < 2: logging.error(f"_ring_division_internal: Invalid bins!"); return [None] * n_rings
    ring_indices = np.digitize(distances, ring_bins) - 1;
    ring_indices[distances < ring_bins[1]] = 0;
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    rings_coords_np: List[Optional[np.ndarray]] = [None] * n_rings;
    pixel_counts = np.zeros(n_rings, dtype=int);
    total_pixels_in_rings = 0
    for ring_idx in range(n_rings_eff):
        coords_for_ring_np = np.argwhere(ring_indices == ring_idx);
        count = coords_for_ring_np.shape[0]
        if count > 0: rings_coords_np[ring_idx] = coords_for_ring_np; pixel_counts[
            ring_idx] = count; total_pixels_in_rings += count
    total_pixels_in_subband = H * W
    if total_pixels_in_rings != total_pixels_in_subband: logging.debug(
        f"_ring_division_internal: Pixel count mismatch! Rings: {total_pixels_in_rings}, Subband: {total_pixels_in_subband}. Shape: {H}x{W}")
    logging.debug(
        f"_ring_division_internal calc time for shape {subband_shape}: {time.time() - func_start_time:.6f}s. Ring pixels: {pixel_counts[:n_rings_eff]}")
    return rings_coords_np


@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    logging.debug(f"Cache miss for ring_division shape={subband_shape}, n_rings={n_rings}. Calculating...")
    return _ring_division_internal(subband_shape, n_rings)


def ring_division(lowpass_subband: np.ndarray, n_rings: int = N_RINGS, frame_number: int = -1) -> List[
    Optional[np.ndarray]]:
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(
            f"[F:{frame_number}] Input to ring_division is not a 2D numpy array! Type: {type(lowpass_subband)}")
        return [None] * n_rings
    shape = lowpass_subband.shape
    try:
        coords_list_np = get_ring_coords_cached(shape, n_rings)
        logging.debug(
            f"[F:{frame_number}] Using cached/calculated ring coords (type: {type(coords_list_np)}) for shape {shape}")
        if not isinstance(coords_list_np, list) or not all(
                isinstance(item, (np.ndarray, type(None))) for item in coords_list_np):
            logging.error(f"[F:{frame_number}] Cached ring division result has unexpected type. Recalculating.")
            get_ring_coords_cached.cache_clear();
            coords_list_np = _ring_division_internal(shape, n_rings)
        return [arr.copy() if arr is not None else None for arr in coords_list_np]
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception in ring_division or cache lookup: {e}", exc_info=True);
        return [None] * n_rings


def calculate_entropies(ring_vals: np.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    eps = 1e-12;
    if ring_vals.size == 0: return 0.0, 0.0
    # Нормализуем значения перед гистограммой, если они не в [0, 1]
    min_v, max_v = np.min(ring_vals), np.max(ring_vals)
    if min_v < 0.0 or max_v > 1.0:
        logging.debug(
            f"[F:{frame_number}, R:{ring_index}] Ring values out of [0,1] range ({min_v:.2f}, {max_v:.2f}). Clipping.")
        ring_vals_clipped = np.clip(ring_vals, 0.0, 1.0)
    else:
        ring_vals_clipped = ring_vals

    hist, _ = np.histogram(ring_vals_clipped, bins=256, range=(0.0, 1.0), density=False)
    total_count = ring_vals_clipped.size;
    if total_count == 0: return 0.0, 0.0
    probabilities = hist / total_count;
    probabilities = probabilities[probabilities > eps]
    if probabilities.size == 0: return 0.0, 0.0
    visual_entropy = -np.sum(probabilities * np.log2(probabilities));
    edge_entropy = -np.sum(probabilities * np.exp(1.0 - probabilities));
    return visual_entropy, edge_entropy


def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    if ring_vals.size == 0: logging.warning(
        f"[F:{frame_number}, R:{ring_index}] compute_adaptive_alpha empty ring_vals."); return ALPHA_MIN
    visual_entropy, edge_entropy = calculate_entropies(ring_vals, frame_number, ring_index);
    local_variance = np.var(ring_vals)
    texture_factor = 1.0 / (1.0 + np.clip(local_variance, 0, 1) * 10.0)  # Меньший фактор для большей текстуры/вариации
    eps = 1e-12
    if abs(visual_entropy) < eps:
        entropy_ratio = 0.0; logging.debug(f"[F:{frame_number}, R:{ring_index}] Visual entropy near zero.")
    else:
        entropy_ratio = edge_entropy / visual_entropy  # Отношение Ee/Ev
    # Сигмоида для преобразования отношения в [0, 1], умноженная на текстурный фактор
    # sigmoid_input = entropy_ratio * texture_factor # Можно комбинировать здесь
    sigmoid_input = entropy_ratio  # Или использовать только отношение энтропий
    sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
    # Комбинируем с текстурным фактором - возможно, обратная зависимость нужна?
    # Если текстура высокая (фактор низкий), хотим меньшую альфа? Или наоборот?
    # Допустим, высокая текстура маскирует -> можно большую альфа.
    # Тогда final_alpha ~ sigmoid * (1 / texture_factor)? Или final_alpha ~ sigmoid + (1-texture_factor)?
    # Попробуем простой вариант: чем выше отношение Ee/Ev и чем ниже вариация, тем выше альфа.
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid_output * (
                1.0 - texture_factor * 0.5)  # Примерная формула
    final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)
    logging.debug(
        f"[F:{frame_number}, R:{ring_index}] Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, Var={local_variance:.4f}, TxtrF={texture_factor:.4f}, Ratio={entropy_ratio:.4f}, Sig={sigmoid_output:.4f} -> final_alpha={final_alpha:.4f}")
    return final_alpha


def deterministic_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    try:
        # Используем серый кадр для хеша
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray_frame = frame
        else:
            logging.error(f"[F:{frame_number}] Invalid frame dim for hashing."); return random.randrange(
                n_rings)  # Случайное при ошибке

        # Уменьшаем размер для ускорения хеширования
        small_frame = cv2.resize(gray_frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(small_frame);
        phash = imagehash.phash(pil_img);
        hash_str = str(phash)
        if not hash_str: logging.warning(f"[F:{frame_number}] Empty phash."); return random.randrange(n_rings)
        try:
            hash_int = int(hash_str, 16)
        except ValueError:
            logging.error(f"[F:{frame_number}] Invalid hash format '{hash_str}'."); return random.randrange(n_rings)
        selected_ring = hash_int % n_rings;
        logging.debug(f"[F:{frame_number}] Deterministic ring: hash={hash_str}, ring={selected_ring}");
        return selected_ring
    except Exception as e:
        logging.error(f"[F:{frame_number}] Error in deterministic_ring_selection: {e}",
                      exc_info=True); return random.randrange(n_rings)


def keypoint_based_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    # (Код без изменений)
    try:
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            logging.error(f"[F:{frame_number}] Invalid frame dim for keypoints."); return random.randrange(n_rings)
        fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True);
        keypoints = fast.detect(gray, None)
        if not keypoints: logging.warning(
            f"[F:{frame_number}] No FAST keypoints, using default."); return random.randrange(n_rings)
        num_keypoints = len(keypoints);
        x_avg = sum(kp.pt[0] for kp in keypoints) / num_keypoints;
        y_avg = sum(kp.pt[1] for kp in keypoints) / num_keypoints
        h, w = gray.shape[:2];
        x_norm = x_avg / w if w > 0 else 0.5;
        y_norm = y_avg / h if h > 0 else 0.5
        dist = np.sqrt((x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2);
        selected_ring = int((dist / 0.5) * n_rings) if dist > 0 else 0;
        selected_ring = max(0, min(selected_ring, n_rings - 1))
        logging.debug(
            f"[F:{frame_number}] Keypoint-based ring: kpts={num_keypoints}, dist={dist:.3f}, ring={selected_ring}");
        return selected_ring
    except Exception as e:
        logging.error(f"[F:{frame_number}] Error in keypoint_based_ring_selection: {e}",
                      exc_info=True); return random.randrange(n_rings)


# --- ИЗМЕНЕННАЯ select_embedding_ring (возвращает список лучших колец) ---
def select_embedding_rings(
        lowpass_subband: np.ndarray, rings_coords_np: List[Optional[np.ndarray]],
        num_to_select: int = NUM_RINGS_TO_USE, metric: str = RING_SELECTION_METRIC,
        frame_number: int = -1, min_pixels: int = 10  # Минимальное кол-во пикселей в кольце для выбора
) -> List[int]:
    """Выбирает num_to_select лучших колец на основе метрики."""
    func_start_time = time.time()
    metric_values: List[Tuple[float, int]] = []  # (значение_метрики, индекс_кольца)
    n_rings_available = len(rings_coords_np);
    known_metrics = ['entropy', 'energy', 'variance', 'mean_abs_dev']

    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Invalid lowpass_subband input for ring selection!");
        return []
    if num_to_select <= 0: return []

    metric_to_use = metric if metric in known_metrics else 'entropy'
    if metric not in known_metrics: logging.warning(
        f"[F:{frame_number}] Unknown metric '{metric}', defaulting to 'entropy'.")
    logging.debug(f"[F:{frame_number}] Selecting {num_to_select} rings using metric: '{metric_to_use}'")

    for i, coords_np in enumerate(rings_coords_np):
        current_metric = -float('inf')
        if coords_np is None or coords_np.size < min_pixels * 2:  # Проверка на мин. пиксели (*2 т.к. shape (N, 2))
            metric_values.append((current_metric, i));
            continue
        try:
            if coords_np.ndim != 2 or coords_np.shape[1] != 2:
                logging.warning(f"[F:{frame_number}, R:{i}] Invalid coords shape.");
                metric_values.append((current_metric, i));
                continue

            rows, cols = coords_np[:, 0], coords_np[:, 1];
            ring_vals = lowpass_subband[rows, cols].astype(np.float32)
            if ring_vals.size < min_pixels:
                metric_values.append((current_metric, i));
                continue

            if metric_to_use == 'entropy':
                v_e, _ = calculate_entropies(ring_vals, frame_number, i); current_metric = v_e
            elif metric_to_use == 'energy':
                current_metric = np.sum(ring_vals ** 2)
            elif metric_to_use == 'variance':
                current_metric = np.var(ring_vals)
            elif metric_to_use == 'mean_abs_dev':
                mean_val = np.mean(ring_vals); current_metric = np.mean(np.abs(ring_vals - mean_val))

            if not np.isfinite(current_metric):
                logging.warning(f"[F:{frame_number}, R:{i}] Metric is not finite ({current_metric}). Skipping.")
                current_metric = -float('inf')

            metric_values.append((current_metric, i))

        except IndexError:
            logging.error(f"[F:{frame_number}, R:{i}] IndexError calculating metric.",
                          exc_info=False); metric_values.append((-float('inf'), i))
        except Exception as e:
            logging.error(f"[F:{frame_number}, R:{i}] Error calculating metric: {e}",
                          exc_info=False); metric_values.append((-float('inf'), i))

    # Сортируем по убыванию метрики
    metric_values.sort(key=lambda x: x[0], reverse=True)

    # Отбираем лучшие num_to_select индексов с конечными метриками
    selected_indices = [idx for val, idx in metric_values if val > -float('inf')][:num_to_select]

    # Логирование
    metric_log_str = ", ".join([f"{idx}:{val:.4f}" if val > -float('inf') else f"{idx}:Err/Empty" for val, idx in
                                sorted(metric_values, key=lambda x: x[1])])
    logging.debug(f"[F:{frame_number}] Ring metrics ('{metric_to_use}'): [{metric_log_str}]")
    logging.info(f"[F:{frame_number}] Multi-ring selection result: {selected_indices}")

    # Если не удалось выбрать достаточно, дополняем дефолтными или первыми валидными
    if len(selected_indices) < num_to_select:
        logging.warning(
            f"[F:{frame_number}] Only selected {len(selected_indices)}/{num_to_select} rings. Trying to add fallbacks.")
        fallback_candidates = [idx for val, idx in metric_values if val > -float('inf') and idx not in selected_indices]
        # Добавляем кандидатов, пока не наберем нужное количество или кандидаты не кончатся
        needed = num_to_select - len(selected_indices)
        selected_indices.extend(fallback_candidates[:needed])

        # Если все еще не хватает, проверяем дефолтный
        if len(selected_indices) < num_to_select and DEFAULT_RING_INDEX not in selected_indices:
            if 0 <= DEFAULT_RING_INDEX < n_rings_available and rings_coords_np[DEFAULT_RING_INDEX] is not None and \
                    rings_coords_np[DEFAULT_RING_INDEX].size >= min_pixels * 2:
                logging.warning(f"[F:{frame_number}] Adding default ring {DEFAULT_RING_INDEX} as fallback.")
                selected_indices.append(DEFAULT_RING_INDEX)

        # Если все еще не хватает, берем первые попавшиеся валидные
        if len(selected_indices) < num_to_select:
            for idx in range(n_rings_available):
                if idx not in selected_indices and rings_coords_np[idx] is not None and rings_coords_np[
                    idx].size >= min_pixels * 2:
                    logging.warning(f"[F:{frame_number}] Adding first available ring {idx} as fallback.")
                    selected_indices.append(idx)
                    if len(selected_indices) == num_to_select: break

    if len(selected_indices) < num_to_select:
        logging.error(
            f"[F:{frame_number}] Could not select {num_to_select} valid rings! Selected only: {selected_indices}")
        # Можно либо вернуть то что есть, либо пустой список, либо вызвать исключение
        # Вернем то, что есть

    logging.debug(f"[F:{frame_number}] Ring selection process time: {time.time() - func_start_time:.4f}s")
    return selected_indices[:num_to_select]  # Возвращаем не более num_to_select


def calculate_perceptual_mask(input_plane: np.ndarray, frame_number: int = -1) -> Optional[np.ndarray]:
    if not isinstance(input_plane, np.ndarray) or input_plane.ndim != 2: logging.error(
        f"[F:{frame_number}] Invalid input for perceptual mask."); return None
    try:
        # Используем input_plane напрямую (предполагаем, что он уже 0-1)
        plane_32f = input_plane.astype(np.float32)
        gx = cv2.Sobel(plane_32f, cv2.CV_32F, 1, 0, ksize=3);
        gy = cv2.Sobel(plane_32f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        # Локальная яркость и вариация (текстура)
        local_mean = cv2.GaussianBlur(plane_32f, (11, 11), 5)
        mean_sq = cv2.GaussianBlur(plane_32f ** 2, (11, 11), 5);
        sq_mean = local_mean ** 2
        local_variance = np.maximum(mean_sq - sq_mean, 0);
        local_stddev = np.sqrt(local_variance)
        # Маска яркости (ближе к 0.5 -> менее заметно)
        brightness_mask = 1.0 - np.abs(local_mean - 0.5) * 2.0  # 1 = маскирует, 0 = заметно
        # Нормализация градиента и стандартного отклонения
        eps = 1e-9;
        max_grad = np.max(grad_mag);
        grad_norm = grad_mag / (max_grad + eps) if max_grad > eps else np.zeros_like(grad_mag)
        max_stddev = np.max(local_stddev);
        stddev_norm = local_stddev / (max_stddev + eps) if max_stddev > eps else np.zeros_like(local_stddev)
        # Комбинируем компоненты маски: градиенты и текстура маскируют лучше
        # w_grad = 0.4; w_texture = 0.4; w_brightness = 0.2
        # mask = (grad_norm * w_grad + stddev_norm * w_texture + np.clip(brightness_mask, 0, 1) * w_brightness)
        # Простая маска: чем выше градиент или текстура, тем лучше маскировка (ближе к 1)
        mask = np.maximum(grad_norm, stddev_norm)
        # Нормализуем итоговую маску
        max_mask = np.max(mask);
        mask_norm = mask / (max_mask + eps) if max_mask > eps else np.zeros_like(mask)
        mask_norm = np.clip(mask_norm, 0.0, 1.0)  # Ограничиваем [0, 1]
        # Инвертируем? Нет, маска должна быть 1 там, где можно сильно менять, 0 - где нельзя.
        logging.debug(
            f"[F:{frame_number}] Perceptual mask calculated. Shape: {mask_norm.shape}, Min: {np.min(mask_norm):.4f}, Max: {np.max(mask_norm):.4f}, Mean: {np.mean(mask_norm):.4f}")
        return mask_norm.astype(np.float32)
    except Exception as e:
        logging.error(f"[F:{frame_number}] Error calculating perceptual mask: {e}", exc_info=True); return np.ones_like(
            input_plane, dtype=np.float32)  # Возвращаем 1, чтобы не блокировать встраивание


def calculate_spatial_weights_vectorized(shape: Tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = shape;
    center_r, center_c = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = np.sqrt(center_r ** 2 + center_c ** 2) + 1e-9
    dists = np.sqrt((rows - center_r) ** 2 + (cols - center_c) ** 2)
    norm_dists = dists / max_dist
    # Веса: от 0.5 в центре до 1.0 на краях (больше меняем на краях)
    weights = 0.5 + 0.5 * norm_dists
    return np.clip(weights, 0.5, 1.0)


# --- ИЗМЕНЕННАЯ add_ecc (работает с битами, паддинг до k) ---
def add_ecc(data_bits: np.ndarray, bch: 'bchlib.BCH') -> Optional[np.ndarray]:
    """
    Добавляет ECC к массиву битов данных, дополняя их до bch.k нулями.
    Возвращает numpy массив битов [padded_data + ecc].
    """
    if not BCHLIB_AVAILABLE or bch is None:
        logging.warning("bchlib не доступна или не инициализирована, ECC не добавляется.")
        return data_bits  # Возвращаем исходные биты

    k = bch.k  # Длина данных, ожидаемая BCH
    n = bch.n  # Полная длина блока BCH
    ecc_len = bch.ecc_bits  # Длина ECC части

    if data_bits.size > k:
        logging.error(f"Размер данных ({data_bits.size} бит) больше, чем вместимость BCH k={k} бит.")
        return None
    elif data_bits.size < k:
        # Дополняем нулями до k бит
        padding_len = k - data_bits.size
        padded_data_bits = np.pad(data_bits, (0, padding_len), 'constant', constant_values=0).astype(np.uint8)
        logging.debug(f"Данные ({data_bits.size} бит) дополнены {padding_len} нулями до {k} бит.")
    else:
        padded_data_bits = data_bits.astype(np.uint8)

    try:
        # Конвертируем дополненные данные в байты для bchlib
        data_bytes = np.packbits(padded_data_bits).tobytes()

        # Кодируем
        ecc_computed_bytes = bch.encode(data_bytes)

        # Конвертируем ECC байты в биты
        ecc_computed_bits = np.unpackbits(np.frombuffer(ecc_computed_bytes, dtype=np.uint8))
        # Убедимся, что ECC имеет правильную длину (отсекаем лишние нули от packbits)
        ecc_computed_bits = ecc_computed_bits[:ecc_len]

        # Соединяем дополненные данные и ECC
        packet_bits = np.concatenate((padded_data_bits, ecc_computed_bits)).astype(np.uint8)

        # Проверка итоговой длины
        if packet_bits.size != n:
            logging.warning(
                f"Итоговая длина пакета ({packet_bits.size}) не совпадает с ожидаемой n={n}. Это может быть проблемой.")
            # Обрежем или дополним до n? Лучше пока выдать предупреждение.

        logging.info(
            f"ECC: К {data_bits.size} битам данных (дополнено до {k}) добавлено {ecc_len} бит ECC. Итого пакет: {packet_bits.size} бит.")
        return packet_bits

    except Exception as e:
        logging.error(f"Ошибка при кодировании ECC: {e}", exc_info=True)
        return None


# ============================================================
# --- Функции Работы с Видео (I/O) ---
# ============================================================
# (read_video и write_video без изменений)
def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # (Код read_video без изменений)
    func_start_time = time.time();
    logging.info(f"Reading video from: {video_path}")
    frames = [];
    fps = float(FPS);
    cap = None;
    expected_height, expected_width = -1, -1
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed to open video: {video_path}"); return frames, fps
        fps_read = cap.get(cv2.CAP_PROP_FPS);
        if fps_read > 0:
            fps = float(fps_read); logging.info(f"Detected FPS: {fps:.2f}")
        else:
            logging.warning(f"Failed to get FPS. Using default: {fps}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
        logging.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, ~{frame_count_prop} frames")
        expected_height, expected_width = height, width;
        frame_index = 0;
        read_count = 0;
        none_frame_count = 0;
        invalid_shape_count = 0
        while True:
            ret, frame = cap.read();
            frame_number_log = frame_index + 1
            if not ret: logging.info(f"End of stream after reading {read_count} frames (index {frame_index})."); break
            if frame is None: logging.warning(
                f"Received None frame at index {frame_index}. Skipping."); none_frame_count += 1; frame_index += 1; continue
            if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
                current_h, current_w = frame.shape[:2]
                if current_h == expected_height and current_w == expected_width:
                    frames.append(frame); read_count += 1;
                else:
                    logging.warning(
                        f"Frame {frame_number_log} shape mismatch ({current_w}x{current_h} vs {expected_width}x{expected_height}). Skipping."); invalid_shape_count += 1
            else:
                logging.warning(
                    f"Frame {frame_number_log} not valid BGR (ndim={frame.ndim}, dtype={frame.dtype}). Skipping."); invalid_shape_count += 1
            frame_index += 1
        logging.info(
            f"Finished reading. Valid frames: {len(frames)}. Skipped None: {none_frame_count}, Invalid shape: {invalid_shape_count}")
    except Exception as e:
        logging.error(f"Exception during video reading: {e}", exc_info=True)
    finally:
        if cap is not None and cap.isOpened(): cap.release(); logging.debug("Video capture released.")
    if not frames: logging.error(f"No valid frames were read from {video_path}")
    logging.debug(f"Read video time: {time.time() - func_start_time:.4f}s")
    return frames, fps


def write_video(frames: List[np.ndarray], out_path: str, fps: float, codec: str = OUTPUT_CODEC):
    # (Код write_video без изменений)
    func_start_time = time.time();
    if not frames: logging.error("No frames to write."); return
    logging.info(f"Starting video writing to: {out_path} (FPS: {fps:.2f}, Codec: {codec})")
    writer = None
    try:
        # Ищем первый валидный кадр для определения размера
        first_valid = None
        for f in frames:
            if f is not None and f.ndim == 3 and f.shape[2] == 3 and f.dtype == np.uint8:
                first_valid = f
                break
        if first_valid is None: logging.error("No valid frames found to determine output size."); return

        h, w, c = first_valid.shape;
        logging.info(f"Output resolution: {w}x{h}")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        base, _ = os.path.splitext(out_path);
        out_path_corrected = base + OUTPUT_EXTENSION
        if out_path_corrected != out_path: logging.info(
            f"Correcting output extension to '{OUTPUT_EXTENSION}'. New path: {out_path_corrected}"); out_path = out_path_corrected
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h));
        writer_codec_used = codec
        if not writer.isOpened():
            logging.error(f"Failed to create VideoWriter with codec '{codec}'.")
            # Попытка использовать MJPG как fallback для AVI
            if OUTPUT_EXTENSION.lower() == '.avi' and codec.upper() != 'MJPG':
                fallback_codec = 'MJPG';
                logging.warning(f"Trying fallback codec '{fallback_codec}'.")
                fourcc_fallback = cv2.VideoWriter_fourcc(*fallback_codec);
                writer = cv2.VideoWriter(out_path, fourcc_fallback, fps, (w, h))
                if writer.isOpened():
                    logging.info(f"Using fallback codec '{fallback_codec}'."); writer_codec_used = fallback_codec
                else:
                    logging.critical(f"Fallback codec '{fallback_codec}' also failed."); return
            else:
                logging.critical(f"Cannot initialize VideoWriter."); return

        written_count = 0;
        skipped_count = 0;
        start_write_loop = time.time()
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)  # Создаем черный кадр заранее
        for i, frame in enumerate(frames):
            if frame is not None and frame.shape == (h, w, c) and frame.dtype == np.uint8:
                writer.write(frame);
                written_count += 1;
            else:
                shape_info = frame.shape if frame is not None else 'None';
                dtype_info = frame.dtype if frame is not None else 'N/A'
                logging.warning(
                    f"Skipping invalid frame #{i + 1}. Shape:{shape_info}, Dtype:{dtype_info}. Writing black frame instead.");
                writer.write(black_frame);
                skipped_count += 1
        logging.debug(f"Write loop time: {time.time() - start_write_loop:.4f}s")
        logging.info(
            f"Finished writing with codec '{writer_codec_used}'. Frames written: {written_count}, Skipped/Replaced with black: {skipped_count}")
    except Exception as e:
        logging.error(f"Exception during video writing: {e}", exc_info=True)
    finally:
        if writer is not None: writer.release(); logging.debug("Video writer released.")
    logging.debug(f"Write video total time: {time.time() - func_start_time:.4f}s")


# ============================================================
# --- ЛОГИКА ВСТРАИВАНИЯ (Embed) ---
# ============================================================

# --- ИЗМЕНЕННАЯ embed_frame_pair (для N=2) ---
def embed_frame_pair(
        frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bits: List[int],
        selected_ring_indices: List[int],  # Теперь ожидает список выбранных колец
        n_rings: int = N_RINGS,
        frame_number: int = 0,
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Встраивает несколько бит (len(bits)) в соответствующие кольца (selected_ring_indices)."""
    func_start_time = time.time();
    pair_num_log = frame_number // 2
    logging.debug(
        f"--- Embed Start: Pair {pair_num_log} (F:{frame_number}), Bits: {bits}, Rings: {selected_ring_indices} ---")

    if len(bits) != len(selected_ring_indices):
        logging.error(
            f"[P:{pair_num_log}] Mismatch between number of bits ({len(bits)}) and rings ({len(selected_ring_indices)}).")
        return None, None
    if not bits or not selected_ring_indices:
        logging.warning(f"[P:{pair_num_log}] Empty bits or rings list. Skipping embedding for this pair.")
        return frame1_bgr, frame2_bgr  # Возвращаем оригиналы

    try:
        if frame1_bgr is None or frame2_bgr is None: logging.error(
            f"[P:{pair_num_log}] Input frame None."); return None, None
        if frame1_bgr.shape != frame2_bgr.shape: logging.error(
            f"[P:{pair_num_log}] Frame shapes mismatch."); return None, None

        try:
            frame1_ycrcb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb); frame2_ycrcb = cv2.cvtColor(frame2_bgr,
                                                                                                      cv2.COLOR_BGR2YCrCb)
        except cv2.error as e:
            logging.error(f"[P:{pair_num_log}] Color conversion failed: {e}"); return None, None

        comp_name = ['Y', 'Cr', 'Cb'][embed_component];
        logging.debug(f"[P:{pair_num_log}] Using {comp_name}")
        try:
            # Извлекаем нужный компонент и нормализуем
            comp1 = frame1_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
            comp2 = frame2_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
            # Сохраняем остальные компоненты для сборки
            Y1_orig = frame1_ycrcb[:, :, 0];
            Cr1 = frame1_ycrcb[:, :, 1];
            Cb1 = frame1_ycrcb[:, :, 2]
            Y2_orig = frame2_ycrcb[:, :, 0];
            Cr2 = frame2_ycrcb[:, :, 1];
            Cb2 = frame2_ycrcb[:, :, 2]
        except IndexError:
            logging.error(f"[P:{pair_num_log}] Invalid component index."); return None, None

        # DTCWT
        pyr1 = dtcwt_transform(comp1, frame_number=frame_number);
        pyr2 = dtcwt_transform(comp2, frame_number=frame_number + 1)
        if pyr1 is None or pyr2 is None or pyr1.lowpass is None or pyr2.lowpass is None:
            logging.error(f"[P:{pair_num_log}] DTCWT failed.");
            return None, None
        L1 = pyr1.lowpass.copy();
        L2 = pyr2.lowpass.copy()  # Работаем с копиями

        # Деление на кольца (координаты)
        rings1_coords_np = ring_division(L1, n_rings=n_rings, frame_number=frame_number)
        rings2_coords_np = ring_division(L2, n_rings=n_rings, frame_number=frame_number + 1)

        # Перцептуальная маска (если используется)
        perceptual_mask = None
        if use_perceptual_masking:
            perceptual_mask = calculate_perceptual_mask(comp1, frame_number=frame_number)
            if perceptual_mask is not None and perceptual_mask.shape != L1.shape:
                perceptual_mask = cv2.resize(perceptual_mask, (L1.shape[1], L1.shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
            elif perceptual_mask is None:
                logging.warning(f"[P:{pair_num_log}] Failed perceptual mask calculation. Proceeding without masking.")

        # --- Цикл встраивания по выбранным кольцам ---
        modifications_applied_count = 0
        for ring_idx, bit_to_embed in zip(selected_ring_indices, bits):
            logging.debug(f"[P:{pair_num_log}] Processing Ring {ring_idx} for Bit {bit_to_embed}")
            if not (0 <= ring_idx < n_rings):
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Invalid ring index.");
                continue
            try:
                coords_1_np = rings1_coords_np[ring_idx];
                coords_2_np = rings2_coords_np[ring_idx]
                if coords_1_np is None or coords_1_np.size == 0 or coords_2_np is None or coords_2_np.size == 0:
                    logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Ring coordinates invalid/empty.");
                    continue

                rows1, cols1 = coords_1_np[:, 0], coords_1_np[:, 1]
                rows2, cols2 = coords_2_np[:, 0], coords_2_np[:, 1]
                ring_vals_1 = L1[rows1, cols1].astype(np.float32)
                ring_vals_2 = L2[rows2, cols2].astype(np.float32)
            except IndexError:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] IndexError extracting ring values.",
                              exc_info=False); continue
            except Exception as e:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Error extracting ring values: {e}",
                              exc_info=False); continue

            if ring_vals_1.size == 0 or ring_vals_2.size == 0:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Extracted empty ring values.");
                continue

            # Адаптивная альфа для текущего кольца
            alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_idx, frame_number)
            logging.info(f"[P:{pair_num_log}, R:{ring_idx}] Embedding Bit={bit_to_embed}, Alpha={alpha:.4f}")

            # DCT -> SVD
            dct1 = dct_1d(ring_vals_1);
            dct2 = dct_1d(ring_vals_2)
            try:
                U1, S1, Vt1 = svd(dct1.reshape(-1, 1), full_matrices=False); U2, S2, Vt2 = svd(dct2.reshape(-1, 1),
                                                                                               full_matrices=False)
            except np.linalg.LinAlgError as e:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] SVD failed: {e}. Skipping ring."); continue

            s1 = S1[0] if S1.size > 0 else 0.0;
            s2 = S2[0] if S2.size > 0 else 0.0
            eps = 1e-12;
            ratio = s1 / (s2 + eps)
            new_s1, new_s2 = s1, s2;
            modified = False
            alpha_sq = alpha * alpha;
            inv_alpha = 1.0 / (alpha + eps)

            # Модификация SVD значений
            if bit_to_embed == 0:
                if ratio < alpha: new_s1 = (s1 * alpha_sq + alpha * s2) / (alpha_sq + 1.0); new_s2 = (
                                                                                                                 alpha * s1 + s2) / (
                                                                                                                 alpha_sq + 1.0); modified = True
            else:  # bit_to_embed == 1
                if ratio >= inv_alpha: new_s1 = (s1 + alpha * s2) / (1.0 + alpha_sq); new_s2 = (
                                                                                                           alpha * s1 + alpha_sq * s2) / (
                                                                                                           1.0 + alpha_sq); modified = True

            log_lvl = logging.INFO if modified else logging.DEBUG
            logging.log(log_lvl,
                        f"[P:{pair_num_log}, R:{ring_idx}] SVD Mod Applied: {modified}. Orig s1={s1:.4f}, s2={s2:.4f}. New s1={new_s1:.4f}, s2={new_s2:.4f}. Target bit={bit_to_embed}.")

            if modified:
                modifications_applied_count += 1
                new_S1 = np.array([[new_s1]]) if S1.size > 0 else np.zeros((1, 1))
                new_S2 = np.array([[new_s2]]) if S2.size > 0 else np.zeros((1, 1))

                # IDCT
                dct1_mod = (U1 @ new_S1 @ Vt1).flatten();
                dct2_mod = (U2 @ new_S2 @ Vt2).flatten()
                ring_vals_1_mod = idct_1d(dct1_mod);
                ring_vals_2_mod = idct_1d(dct2_mod)

                # Применение изменений к L1, L2 с маскировкой
                if len(ring_vals_1_mod) == len(rows1):
                    delta1 = ring_vals_1_mod - ring_vals_1;
                    mod_factors1 = np.ones_like(delta1)
                    if perceptual_mask is not None:
                        # Применяем маску (1=сильно менять, 0=слабо)
                        mask_vals = perceptual_mask[rows1, cols1]
                        mod_factors1 *= (LAMBDA_PARAM + (1.0 - LAMBDA_PARAM) * mask_vals)  # От LAMBDA до 1.0
                    # Пространственное взвешивание (если нужно)
                    # spatial_weights1 = calculate_spatial_weights_vectorized(L1.shape, rows1, cols1); mod_factors1 *= spatial_weights1
                    L1[rows1, cols1] += delta1 * mod_factors1
                else:
                    logging.error(
                        f"[P:{pair_num_log}, R:{ring_idx}] Length mismatch L1. Skipping update for this ring."); continue

                if len(ring_vals_2_mod) == len(rows2):
                    delta2 = ring_vals_2_mod - ring_vals_2;
                    mod_factors2 = np.ones_like(delta2)
                    if perceptual_mask is not None:
                        mask_vals = perceptual_mask[rows2, cols2]
                        mod_factors2 *= (LAMBDA_PARAM + (1.0 - LAMBDA_PARAM) * mask_vals)
                    # spatial_weights2 = calculate_spatial_weights_vectorized(L2.shape, rows2, cols2); mod_factors2 *= spatial_weights2
                    L2[rows2, cols2] += delta2 * mod_factors2
                else:
                    logging.error(
                        f"[P:{pair_num_log}, R:{ring_idx}] Length mismatch L2. Skipping update for this ring."); continue
        # --- Конец цикла по кольцам ---

        if modifications_applied_count == 0:
            logging.warning(
                f"[P:{pair_num_log}] No SVD modifications were applied for bits {bits} in rings {selected_ring_indices}.")
            # Все равно продолжаем, чтобы вернуть кадры (возможно, не измененные)

        # Обратный DTCWT
        pyr1.lowpass = L1;
        pyr2.lowpass = L2
        comp1_mod = dtcwt_inverse(pyr1, frame_number=frame_number);
        comp2_mod = dtcwt_inverse(pyr2, frame_number=frame_number + 1)
        if comp1_mod is None or comp2_mod is None:
            logging.error(f"[P:{pair_num_log}] Inverse DTCWT failed.");
            return None, None

        # Проверка и изменение размера, если нужно
        target_shape = (Y1_orig.shape[0], Y1_orig.shape[1])
        if comp1_mod.shape != target_shape:
            comp1_mod = cv2.resize(comp1_mod, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR);
            logging.warning(f"[P:{pair_num_log}] Resized comp1 after inverse")
        if comp2_mod.shape != target_shape:
            comp2_mod = cv2.resize(comp2_mod, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR);
            logging.warning(f"[P:{pair_num_log}] Resized comp2 after inverse")

        # Масштабирование и клиппинг
        comp1_mod_scaled = np.clip(comp1_mod * 255.0, 0, 255).astype(np.uint8)
        comp2_mod_scaled = np.clip(comp2_mod * 255.0, 0, 255).astype(np.uint8)

        # Сборка кадров YCrCb
        new_ycrcb1 = np.stack((Y1_orig, Cr1, Cb1), axis=-1);
        new_ycrcb2 = np.stack((Y2_orig, Cr2, Cb2), axis=-1)
        # Вставка модифицированного компонента
        new_ycrcb1[:, :, embed_component] = comp1_mod_scaled
        new_ycrcb2[:, :, embed_component] = comp2_mod_scaled

        # Обратное преобразование в BGR
        frame1_mod_bgr = cv2.cvtColor(new_ycrcb1, cv2.COLOR_YCrCb2BGR);
        frame2_mod_bgr = cv2.cvtColor(new_ycrcb2, cv2.COLOR_YCrCb2BGR)

        total_pair_time = time.time() - func_start_time
        logging.info(
            f"--- Embed Finish: Pair {pair_num_log}. Bits: {bits}, Rings: {selected_ring_indices}. Mods: {modifications_applied_count}. Time: {total_pair_time:.4f} sec ---")
        return frame1_mod_bgr, frame2_mod_bgr

    except Exception as e:
        pair_num_log_err = frame_number // 2 if frame_number >= 0 else -1
        logging.error(f"!!! UNHANDLED EXCEPTION in embed_frame_pair (Pair {pair_num_log_err}): {e}", exc_info=True)
        return None, None


# --- ИЗМЕНЕННЫЙ _embed_frame_pair_worker (для N=2) ---
def _embed_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """Воркер для встраивания N бит в N лучших колец."""
    idx1 = args['idx1'];
    pair_num_log = idx1 // 2
    frame_number = args['frame_number']
    bits_to_embed: List[int] = args['bits']
    n_rings: int = args['n_rings']
    ring_selection_method: str = args['ring_selection_method']
    ring_selection_metric: str = args['ring_selection_metric']
    num_rings_to_use: int = args['num_rings_to_use']
    embed_component: int = args['embed_component']
    frame1_orig = args['frame1']
    frame2_orig = args['frame2']
    selected_rings_indices = []  # Список выбранных колец для возврата

    try:
        logging.debug(f"[Worker P:{pair_num_log}] Received bits: {bits_to_embed}")
        if len(bits_to_embed) != num_rings_to_use:
            raise ValueError(f"Worker expected {num_rings_to_use} bits, but received {len(bits_to_embed)}")

        # 1. Преобразование и выбор колец (делаем здесь, чтобы не передавать lowpass)
        try:
            comp1 = frame1_orig[:, :, embed_component].astype(np.float32) / 255.0
            comp2 = frame2_orig[:, :, embed_component].astype(np.float32) / 255.0
        except IndexError:
            logging.error(f"[Worker P:{pair_num_log}] Invalid component index."); return idx1, None, None, []
        except Exception as e:
            logging.error(f"[Worker P:{pair_num_log}] Error getting component: {e}"); return idx1, None, None, []

        pyr1 = dtcwt_transform(comp1, frame_number=frame_number)
        if pyr1 is None or pyr1.lowpass is None:
            logging.error(f"[Worker P:{pair_num_log}] DTCWT failed for frame {frame_number}.");
            return idx1, None, None, []
        L1 = pyr1.lowpass
        rings_coords_np = ring_division(L1, n_rings=n_rings, frame_number=frame_number)

        # Выбираем ЛУЧШИЕ кольца
        if ring_selection_method == 'multi_ring':
            selected_rings_indices = select_embedding_rings(
                L1, rings_coords_np, num_to_select=num_rings_to_use,
                metric=ring_selection_metric, frame_number=frame_number
            )
        elif ring_selection_method == 'deterministic':
            # Для детерминированного выбираем одно и дублируем (не лучший вариант, но для совместимости)
            ring_idx = deterministic_ring_selection(frame1_orig, n_rings, frame_number)
            selected_rings_indices = [ring_idx] * num_rings_to_use
            logging.warning(
                f"[Worker P:{pair_num_log}] Deterministic selection used for multi-bit embedding. Using ring {ring_idx} multiple times.")
        # Добавить другие методы если нужно, или выдать ошибку
        else:
            logging.error(
                f"[Worker P:{pair_num_log}] Unsupported ring selection method '{ring_selection_method}' for multi-bit embedding.")
            return idx1, None, None, []

        if len(selected_rings_indices) < num_rings_to_use:
            logging.error(
                f"[Worker P:{pair_num_log}] Failed to select enough valid rings ({len(selected_rings_indices)}/{num_rings_to_use}).")
            # Попытка встроить в то, что есть? Или пропустить? Пропустим.
            return idx1, None, None, selected_rings_indices  # Возвращаем то, что выбрали, но кадры не будут изменены

        # 2. Вызов embed_frame_pair с выбранными кольцами и битами
        f1_mod, f2_mod = embed_frame_pair(
            frame1_orig, frame2_orig, bits=bits_to_embed,
            selected_ring_indices=selected_rings_indices,  # Передаем список колец
            n_rings=n_rings, frame_number=frame_number,
            use_perceptual_masking=args.get('use_perceptual_masking', USE_PERCEPTUAL_MASKING),
            embed_component=embed_component,
        )

        # Возвращаем результат и список колец, которые *фактически* были использованы
        return idx1, f1_mod, f2_mod, selected_rings_indices

    except Exception as e:
        logging.error(f"Exception in worker for pair {pair_num_log} (Frame {idx1}): {e}", exc_info=True)
        return idx1, None, None, []  # Возвращаем пустой список колец при ошибке


# --- ИЗМЕНЕННАЯ embed_watermark_in_video (для N=2 и Max Repeats) ---
def embed_watermark_in_video(
        frames: List[np.ndarray], packet_bits: np.ndarray,  # Теперь numpy массив
        n_rings: int = N_RINGS,
        ring_selection_method: str = RING_SELECTION_METHOD, ring_selection_metric: str = RING_SELECTION_METRIC,
        num_rings_to_use: int = NUM_RINGS_TO_USE, bits_per_pair: int = BITS_PER_PAIR,
        max_packet_repeats: int = MAX_PACKET_REPEATS,
        default_ring_index: int = DEFAULT_RING_INDEX,  # Все еще нужен для select_embedding_rings
        fps: float = FPS, max_workers: Optional[int] = MAX_WORKERS,
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT
):
    num_frames = len(frames);
    total_pairs = num_frames // 2
    packet_len_bits = packet_bits.size

    if total_pairs == 0 or packet_len_bits == 0:
        logging.warning("Нет пар кадров или нулевая длина пакета ВЗ. Встраивание не выполняется.")
        return frames[:]  # Возвращаем копию оригинальных кадров

    # Расчет количества пар, необходимых для N повторов
    pairs_needed_for_repeats = ceil(max_packet_repeats * packet_len_bits / bits_per_pair)

    # Количество пар для обработки ограничено и видео, и макс. повторами
    num_pairs_to_process = min(total_pairs, pairs_needed_for_repeats)

    # Общее количество бит для встраивания
    total_bits_to_embed = num_pairs_to_process * bits_per_pair

    # Создаем полный поток бит для встраивания
    repeats_needed = ceil(total_bits_to_embed / packet_len_bits) if packet_len_bits > 0 else 1
    bits_for_embedding_flat = np.tile(packet_bits, repeats_needed)[:total_bits_to_embed]

    logging.info(
        f"Starting embedding: {total_bits_to_embed} total bits ({bits_per_pair} bits/pair) across {num_pairs_to_process} frame pairs.")
    logging.info(
        f"Packet length: {packet_len_bits} bits. Target repeats: up to {max_packet_repeats} (requires {pairs_needed_for_repeats} pairs).")
    logging.info(f"Actual bits embedded correspond to {total_bits_to_embed / packet_len_bits:.2f} packets.")

    start_time = time.time();
    watermarked_frames = frames[:]  # Работаем с копией
    if num_pairs_to_process == 0: return watermarked_frames

    tasks_args = []
    for pair_idx in range(num_pairs_to_process):
        idx1 = 2 * pair_idx;
        idx2 = idx1 + 1
        if idx2 >= num_frames or frames[idx1] is None or frames[idx2] is None:
            logging.warning(
                f"Skipping pair {pair_idx} (Frames {idx1}, {idx2}) due to missing frames within processing range.")
            continue

        # Извлекаем нужные биты для этой пары
        start_bit_idx = pair_idx * bits_per_pair
        end_bit_idx = start_bit_idx + bits_per_pair
        current_bits = bits_for_embedding_flat[start_bit_idx:end_bit_idx].tolist()  # Преобразуем в список int

        if len(current_bits) != bits_per_pair:
            logging.error(
                f"Logic error: Failed to get {bits_per_pair} bits for pair {pair_idx}. Got {len(current_bits)}. Skipping.")
            continue

        args = {
            'idx1': idx1, 'frame1': frames[idx1], 'frame2': frames[idx2],
            'bits': current_bits,  # Передаем список бит
            'n_rings': n_rings, 'ring_selection_method': ring_selection_method,
            'ring_selection_metric': ring_selection_metric,
            'num_rings_to_use': num_rings_to_use,  # Теперь это BITS_PER_PAIR
            'default_ring_index': default_ring_index,
            'frame_number': idx1,
            'use_perceptual_masking': use_perceptual_masking,
            'embed_component': embed_component
        }
        tasks_args.append(args)

    if not tasks_args:
        logging.error("No valid tasks created for embedding.")
        return watermarked_frames

    results: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    # Теперь храним список списков индексов колец
    selected_rings_log: Dict[int, List[int]] = {}  # {pair_idx: [ring1, ring2]}
    processed_count = 0;
    error_count = 0;
    task_count = len(tasks_args)

    try:
        logging.info(f"Submitting {task_count} embedding tasks to ThreadPoolExecutor (max_workers={max_workers})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # В future_to_idx1 храним индекс первого кадра пары
            future_to_idx1 = {executor.submit(_embed_frame_pair_worker, arg): arg['idx1'] for arg in tasks_args}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_idx1)):
                idx1 = future_to_idx1[future];
                pair_idx = idx1 // 2  # Получаем индекс пары
                try:
                    # Воркер возвращает (idx1, f1_mod, f2_mod, selected_rings_list)
                    _, f1_mod, f2_mod, selected_rings_list = future.result()

                    if selected_rings_list:  # Записываем только если список не пустой
                        selected_rings_log[pair_idx] = selected_rings_list
                    else:
                        logging.warning(f"Pair {pair_idx} worker returned empty ring list.")

                    if f1_mod is not None and f2_mod is not None:
                        results[idx1] = (f1_mod, f2_mod);
                        processed_count += 1;
                        logging.debug(
                            f"Pair {pair_idx} completed ({i + 1}/{task_count}). Rings used: {selected_rings_list}")
                    else:
                        error_count += 1;
                        logging.error(
                            f"Pair {pair_idx} (Frame {idx1}) failed embedding (returned None frame). Rings attempted: {selected_rings_list}")
                except Exception as exc:
                    error_count += 1;
                    logging.error(f'Pair {pair_idx} (Frame {idx1}) generated exception: {exc}', exc_info=True);
    except Exception as e:
        logging.critical(f"CRITICAL ERROR during ThreadPoolExecutor: {e}", exc_info=True);
        return frames[:]

    logging.info(
        f"ThreadPoolExecutor finished. Successful pairs processed: {processed_count}, Failed pairs: {error_count}.")

    # Применение результатов
    update_count = 0
    for idx1, (f1_mod, f2_mod) in results.items():
        idx2 = idx1 + 1
        if idx1 < len(watermarked_frames): watermarked_frames[idx1] = f1_mod; update_count += 1
        if idx2 < len(watermarked_frames): watermarked_frames[idx2] = f2_mod; update_count += 1
    logging.info(f"Applied results from {len(results)} pairs to frames.")  # Логируем кол-во пар

    # Сохранение лога колец (списки для каждой пары)
    if selected_rings_log:
        try:
            # Преобразуем ключи в строки для JSON
            serializable_rings_log = {str(k): v for k, v in selected_rings_log.items()}
            with open(SELECTED_RINGS_FILE, 'w') as f:
                json.dump(serializable_rings_log, f, indent=4)
            logging.info(f"Saved selected ring indices for {len(selected_rings_log)} pairs to {SELECTED_RINGS_FILE}")
        except IOError as e:
            logging.error(f"Could not save selected rings to {SELECTED_RINGS_FILE}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error saving rings JSON: {e}", exc_info=True)
    else:
        logging.warning("No valid selected rings recorded to save.")

    end_time = time.time();
    logging.info(f"Embedding process finished. Total time: {end_time - start_time:.2f} sec.")
    return watermarked_frames


# ============================================================
# --- ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ ---
# ============================================================
def main():
    main_start_time = time.time()
    input_video = "input.mp4";
    base_output_name = "watermarked_video_n2";
    output_video = base_output_name + OUTPUT_EXTENSION
    logging.info("--- Starting Embedding Main Process ---")
    frames, input_fps = read_video(input_video)
    if not frames: logging.critical("Failed to read input video. Exiting."); return
    fps_to_use = float(FPS) if input_fps <= 0 else input_fps
    if input_fps <= 0: logging.warning(f"Using default FPS={fps_to_use} for writing.")
    total_pairs_available = len(frames) // 2
    if total_pairs_available == 0: logging.error("Not enough frames for any pairs."); return

    # 1. Генерация Payload (64-bit ID)
    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES)  # 8 байт
    original_id_hex = original_id_bytes.hex()
    logging.info(f"Generated Payload ID (Hex): {original_id_hex}")
    logging.info(f"Payload size: {PAYLOAD_LEN_BYTES} bytes ({PAYLOAD_LEN_BYTES * 8} bits)")

    # 2. Добавление ECC (если включено)
    packet_bits: Optional[np.ndarray] = None
    bch = None
    local_use_ecc = USE_ECC and BCHLIB_AVAILABLE

    if local_use_ecc:
        try:
            bch = bchlib.BCH(m=BCH_M, t=BCH_T)
            # Логируем параметры BCH здесь, после успешной инициализации
            k_bits = bch.k
            n_bits = bch.n
            ecc_bits = bch.ecc_bits
            logging.info(
                f"BCH initialized (m={BCH_M}, t={BCH_T}). Params: n={n_bits}, k={k_bits}, ecc={ecc_bits} bits.")

            # Конвертируем ID в биты
            payload_bits_np = np.unpackbits(np.frombuffer(original_id_bytes, dtype=np.uint8))

            # Добавляем ECC (функция сама сделает паддинг до k)
            packet_bits = add_ecc(payload_bits_np, bch)

            if packet_bits is None:
                raise RuntimeError("add_ecc returned None")

        except Exception as e:
            logging.error(f"Failed to initialize/use BCH: {e}. Proceeding without ECC.", exc_info=True)
            local_use_ecc = False  # Отключаем ECC только для этого запуска
    else:
        logging.info("ECC is disabled or bchlib not available.")

    # Если ECC не использовался или не удался, используем просто биты payload
    if packet_bits is None:
        packet_bits = np.unpackbits(np.frombuffer(original_id_bytes, dtype=np.uint8))
        logging.info(f"Using raw payload bits (No ECC). Packet length: {packet_bits.size} bits.")
    else:
        logging.info(f"Final packet length (Payload + Padding + ECC): {packet_bits.size} bits.")

    # 3. Сохранение оригинального ID
    try:
        with open(ORIGINAL_WATERMARK_FILE, "w") as f:
            f.write(original_id_hex)
        logging.info(f"Original ID (Hex) saved to {ORIGINAL_WATERMARK_FILE}")
    except IOError as e:
        logging.error(f"Could not save original ID: {e}")

    # 4. Запуск встраивания
    watermarked_frames = embed_watermark_in_video(
        frames=frames, packet_bits=packet_bits, n_rings=N_RINGS,
        ring_selection_method=RING_SELECTION_METHOD, ring_selection_metric=RING_SELECTION_METRIC,
        num_rings_to_use=NUM_RINGS_TO_USE, bits_per_pair=BITS_PER_PAIR,
        max_packet_repeats=MAX_PACKET_REPEATS,
        default_ring_index=DEFAULT_RING_INDEX,  # Передаем для select_rings
        fps=fps_to_use, max_workers=MAX_WORKERS,
        use_perceptual_masking=USE_PERCEPTUAL_MASKING,
        embed_component=EMBED_COMPONENT
    )

    # 5. Запись результата
    if watermarked_frames and len(watermarked_frames) == len(frames):
        write_video(watermarked_frames, output_video, fps=fps_to_use, codec=OUTPUT_CODEC)
        logging.info(f"Watermarked video saved to: {output_video}")
        try:
            if os.path.exists(output_video):
                file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                logging.info(f"Output file size: {file_size_mb:.2f} MB")
            else:
                logging.error(f"Output file {output_video} not created.")
        except OSError as e:
            logging.error(f"Could not get file size: {e}")
    else:
        logging.error("Embedding failed or frame count mismatch. Output not saved.")

    logging.info("--- Embedding Main Process Finished ---")
    total_script_time = time.time() - main_start_time
    logging.info(f"--- Total Embedder Script Time: {total_script_time:.2f} sec ---")
    print(f"\nEmbedding finished. Output: {output_video}")
    print(f"Logs: {LOG_FILENAME}")
    print(f"Watermark (Original ID): {ORIGINAL_WATERMARK_FILE}")
    print(f"Rings Log: {SELECTED_RINGS_FILE}")
    print("\nRun extractor to verify.")


# --- Запуск с Профилированием ---
if __name__ == "__main__":
    if USE_ECC and not BCHLIB_AVAILABLE:
        print("\nERROR: USE_ECC is True, but bchlib library is not installed.")
        print("Please install it using: pip install bchlib")
        print("ECC will be disabled for this run.")

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    except ValueError as ve:
        logging.critical(f"Value Error: {ve}"); print(f"\nERROR: {ve}.")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True); print(
            f"\nCRITICAL ERROR: {e}. See {LOG_FILENAME}")
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        profile_file = "profile_stats_embed_n2.txt"  # Изменено имя файла профиля
        try:
            with open(profile_file, "w") as f:
                stats_file = pstats.Stats(profiler, stream=f)
                stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved to {profile_file}")
            print(f"Profiling stats saved to {profile_file}")
        except IOError as e:
            logging.error(f"Could not save profiling stats to {profile_file}: {e}")
