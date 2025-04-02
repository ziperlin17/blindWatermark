# -*- coding: utf-8 -*-
# Файл: embedder.py (улучшенная версия с оптимизированным ring_division)

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
import functools # Для кэширования
import cProfile
import pstats

# --- Константы (Общие) ---
LAMBDA_PARAM: float = 0.04
ALPHA_MIN: float = 1.005
ALPHA_MAX: float = 1.1
N_RINGS: int = 8
DEFAULT_RING_INDEX: int = 4
FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed.log'
MAX_WORKERS: Optional[int] = None # None для автоматического определения (обычно по числу ядер CPU)
SELECTED_RINGS_FILE: str = 'selected_rings.json'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark.txt'

# --- Настройки Адаптивности ---
RING_SELECTION_METHOD: str = 'deterministic' # 'deterministic', 'adaptive', 'keypoint', 'multi_ring', 'fixed'
RING_SELECTION_METRIC: str = 'entropy' # 'entropy', 'energy', 'variance', 'mean_abs_dev'
USE_PERCEPTUAL_MASKING: bool = True
EMBED_COMPONENT: int = 1 # 0=Y, 1=Cr, 2=Cb
NUM_RINGS_TO_USE: int = 3 # Для 'multi_ring'

# --- Настройка Видео Выхода ---
OUTPUT_CODEC: str = 'XVID'
OUTPUT_EXTENSION: str = '.avi'

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME, filemode='w', level=logging.INFO,
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
# logging.getLogger().setLevel(logging.DEBUG) # Раскомментируйте для детальной отладки

logging.info(f"--- Запуск Скрипта Встраивания (Версия с {OUTPUT_CODEC} кодеком, Оптимизированные Кольца) ---")
logging.info(f"Настройки: Метод кольца='{RING_SELECTION_METHOD}', Метрика='{RING_SELECTION_METRIC if RING_SELECTION_METHOD in ['adaptive', 'multi_ring'] else 'N/A'}', N_RINGS={N_RINGS}")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(f"Маскировка: {USE_PERCEPTUAL_MASKING}, Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG']:
    logging.warning(f"Используется lossy кодек '{OUTPUT_CODEC}'. Тщательно проверьте извлекаемость ВЗ! Может потребоваться УВЕЛИЧЕНИЕ ALPHA_MIN/ALPHA_MAX.")

# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
# ============================================================

def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    return dct(signal_1d, type=2, norm='ortho')

def idct_1d(coeffs_1d: np.ndarray) -> np.ndarray:
    return idct(coeffs_1d, type=2, norm='ortho')

def dtcwt_transform(y_plane: np.ndarray, frame_number: int = -1) -> Optional[Pyramid]:
    # (Код dtcwt_transform остается без изменений)
    func_start_time = time.time(); logging.debug(f"[F:{frame_number}] Input plane shape: {y_plane.shape}")
    if np.any(np.isnan(y_plane)): logging.warning(f"[F:{frame_number}] NaNs detected in input plane!")
    try:
        t = Transform2d(); rows, cols = y_plane.shape; pad_rows = rows % 2 != 0; pad_cols = cols % 2 != 0
        if pad_rows or pad_cols: y_plane_padded = np.pad(y_plane, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        else: y_plane_padded = y_plane
        pyramid = t.forward(y_plane_padded.astype(np.float32), nlevels=1)
        if hasattr(pyramid, 'lowpass') and pyramid.lowpass is not None: lp = pyramid.lowpass; logging.debug(f"[F:{frame_number}] DTCWT lowpass shape: {lp.shape}");
        else: logging.error(f"[F:{frame_number}] DTCWT no valid lowpass!"); return None
        logging.debug(f"[F:{frame_number}] DTCWT transform time: {time.time() - func_start_time:.4f}s"); pyramid.padding_info = (pad_rows, pad_cols); return pyramid
    except Exception as e: logging.error(f"[F:{frame_number}] Exception during DTCWT transform: {e}", exc_info=True); return None

def dtcwt_inverse(pyramid: Pyramid, frame_number: int = -1) -> Optional[np.ndarray]:
    # (Код dtcwt_inverse остается без изменений)
    func_start_time = time.time();
    if not isinstance(pyramid, Pyramid) or not hasattr(pyramid, 'lowpass'): logging.error(f"[F:{frame_number}] Invalid pyramid."); return None
    logging.debug(f"[F:{frame_number}] Input lowpass shape for inverse: {pyramid.lowpass.shape}")
    try:
        t = Transform2d(); reconstructed_padded = t.inverse(pyramid).astype(np.float32)
        pad_rows, pad_cols = getattr(pyramid, 'padding_info', (False, False))
        if pad_rows or pad_cols: rows, cols = reconstructed_padded.shape; reconstructed_y = reconstructed_padded[:rows - pad_rows if pad_rows else rows, :cols - pad_cols if pad_cols else cols]
        else: reconstructed_y = reconstructed_padded
        logging.debug(f"[F:{frame_number}] DTCWT inverse output shape: {reconstructed_y.shape}");
        if np.any(np.isnan(reconstructed_y)): logging.warning(f"[F:{frame_number}] NaNs detected after inverse DTCWT!")
        logging.debug(f"[F:{frame_number}] DTCWT inverse time: {time.time() - func_start_time:.4f}s"); return reconstructed_y
    except Exception as e: logging.error(f"[F:{frame_number}] Exception during DTCWT inverse: {e}", exc_info=True); return None

# --- ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ РАБОТЫ С КОЛЬЦАМИ (НОВЫЕ) ---
@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    """
    Вычисляет координаты колец для заданного размера подполосы.
    Возвращает список NumPy массивов координат [N, 2] или None для пустых колец.
    """
    func_start_time = time.time()
    H, W = subband_shape
    if H < 2 or W < 2:
        logging.error(f"_ring_division_internal: Subband too small: {H}x{W}.")
        return [None] * n_rings

    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
    min_dist, max_dist = np.min(distances), np.max(distances)

    if max_dist < 1e-6:
        ring_bins = np.array([0.0, 1.0])
        n_rings_eff = 1
    else:
        ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1)
        n_rings_eff = n_rings

    if len(ring_bins) < 2:
        logging.error(f"_ring_division_internal: Invalid bins!")
        return [None] * n_rings

    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)

    rings_coords_np: List[Optional[np.ndarray]] = [None] * n_rings
    pixel_counts = np.zeros(n_rings, dtype=int)
    total_pixels_in_rings = 0

    for ring_idx in range(n_rings_eff):
        # Эта операция может быть узким местом, если колец много
        coords_for_ring_np = np.argwhere(ring_indices == ring_idx)
        count = coords_for_ring_np.shape[0]
        if count > 0:
            rings_coords_np[ring_idx] = coords_for_ring_np
            pixel_counts[ring_idx] = count
            total_pixels_in_rings += count

    total_pixels_in_subband = H * W
    if total_pixels_in_rings != total_pixels_in_subband:
         logging.debug(f"_ring_division_internal: Pixel count mismatch! Rings: {total_pixels_in_rings}, Subband: {total_pixels_in_subband}. Shape: {H}x{W}")

    logging.debug(f"_ring_division_internal calc time for shape {subband_shape}: {time.time() - func_start_time:.6f}s. Ring pixels: {pixel_counts[:n_rings_eff]}")
    return rings_coords_np

@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    logging.debug(f"Cache miss for ring_division shape={subband_shape}, n_rings={n_rings}. Calculating...")
    return _ring_division_internal(subband_shape, n_rings)

def ring_division(lowpass_subband: np.ndarray, n_rings: int = N_RINGS, frame_number: int = -1) -> List[Optional[np.ndarray]]:
    """
    Разбивает подполосу на кольца, используя кэширование.
    Возвращает список NumPy массивов координат [N, 2] или None для пустых колец.
    """
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Input to ring_division is not a 2D numpy array! Type: {type(lowpass_subband)}")
        return [None] * n_rings

    shape = lowpass_subband.shape
    try:
        coords_list_np = get_ring_coords_cached(shape, n_rings)
        # Важно! Кэш возвращает тот же объект. Если кто-то изменит массив из списка, кэш будет испорчен.
        # Для безопасности можно возвращать копии, но это замедлит.
        # return [arr.copy() if arr is not None else None for arr in coords_list_np]
        logging.debug(f"[F:{frame_number}] Using cached/calculated ring coords (type: {type(coords_list_np)}) for shape {shape}")
        if not isinstance(coords_list_np, list) or not all(isinstance(item, (np.ndarray, type(None))) for item in coords_list_np):
             logging.error(f"[F:{frame_number}] Cached ring division result has unexpected type. Recalculating.")
             get_ring_coords_cached.cache_clear()
             coords_list_np = _ring_division_internal(shape, n_rings)
        return coords_list_np
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception in ring_division or cache lookup: {e}", exc_info=True)
        return [None] * n_rings
# --- Конец оптимизированных функций колец ---

def calculate_entropies(ring_vals: np.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    # (Код calculate_entropies без изменений)
    eps=1e-12
    if ring_vals.size==0: return 0.0, 0.0
    hist, _ =np.histogram(np.clip(ring_vals,0.0,1.0), bins=256, range=(0.0,1.0), density=False)
    total_count=ring_vals.size
    if total_count == 0: return 0.0, 0.0
    probabilities=hist/total_count
    probabilities=probabilities[probabilities>eps]
    if probabilities.size==0: return 0.0, 0.0
    visual_entropy=-np.sum(probabilities*np.log2(probabilities))
    edge_entropy=-np.sum(probabilities*np.exp(1.0-probabilities))
    return visual_entropy, edge_entropy

def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    # (Код compute_adaptive_alpha_entropy без изменений)
    if ring_vals.size == 0: logging.warning(f"[F:{frame_number}, R:{ring_index}] compute_adaptive_alpha empty ring_vals."); return ALPHA_MIN
    visual_entropy, edge_entropy = calculate_entropies(ring_vals, frame_number, ring_index)
    local_variance = np.var(ring_vals)
    texture_factor = 1.0 / (1.0 + np.clip(local_variance, 0, 1) * 10.0) # Оригинальный factor
    eps = 1e-12
    if abs(visual_entropy) < eps:
        entropy_ratio = 0.0
        logging.debug(f"[F:{frame_number}, R:{ring_index}] Visual entropy near zero.")
    else:
        entropy_ratio = edge_entropy / visual_entropy
    sigmoid_input = entropy_ratio
    sigmoid_ratio = 1.0 / (1.0 + np.exp(-sigmoid_input)) * texture_factor
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid_ratio
    final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)
    logging.info(f"[F:{frame_number}, R:{ring_index}] Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, Var={local_variance:.4f}, Factor={texture_factor:.4f}, Ratio={entropy_ratio:.4f} -> final_alpha={final_alpha:.4f}")
    return final_alpha

def deterministic_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    # (Код deterministic_ring_selection без изменений)
    try:
        small_frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        if small_frame.ndim == 3 and small_frame.shape[2] == 3: gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        elif small_frame.ndim == 2: gray_frame = small_frame
        else: logging.error(f"[F:{frame_number}] Invalid frame dim for hashing."); return DEFAULT_RING_INDEX
        pil_img = Image.fromarray(gray_frame); phash = imagehash.phash(pil_img); hash_str = str(phash)
        if not hash_str: logging.warning(f"[F:{frame_number}] Empty phash."); return DEFAULT_RING_INDEX
        try: hash_int = int(hash_str, 16)
        except ValueError: logging.error(f"[F:{frame_number}] Invalid hash format '{hash_str}'."); return DEFAULT_RING_INDEX
        selected_ring = hash_int % n_rings; logging.info(f"[F:{frame_number}] Deterministic ring: hash={hash_str}, ring={selected_ring}"); return selected_ring
    except Exception as e: logging.error(f"[F:{frame_number}] Error in deterministic_ring_selection: {e}", exc_info=True); return DEFAULT_RING_INDEX

def keypoint_based_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    # (Код keypoint_based_ring_selection без изменений)
    try:
        if frame.ndim == 3 and frame.shape[2] == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2: gray = frame
        else: logging.error(f"[F:{frame_number}] Invalid frame dim for keypoints."); return DEFAULT_RING_INDEX
        fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True); keypoints = fast.detect(gray, None)
        if not keypoints: logging.warning(f"[F:{frame_number}] No FAST keypoints, using default."); return DEFAULT_RING_INDEX
        num_keypoints = len(keypoints); x_avg = sum(kp.pt[0] for kp in keypoints)/num_keypoints; y_avg = sum(kp.pt[1] for kp in keypoints)/num_keypoints
        h, w = gray.shape[:2]; x_norm = x_avg/w if w>0 else 0.5; y_norm = y_avg/h if h>0 else 0.5
        dist = np.sqrt((x_norm-0.5)**2 + (y_norm-0.5)**2); selected_ring = int((dist/0.5)*n_rings) if dist>0 else 0; selected_ring = max(0, min(selected_ring, n_rings-1))
        logging.info(f"[F:{frame_number}] Keypoint-based ring: kpts={num_keypoints}, dist={dist:.3f}, ring={selected_ring}"); return selected_ring
    except Exception as e: logging.error(f"[F:{frame_number}] Error in keypoint_based_ring_selection: {e}", exc_info=True); return DEFAULT_RING_INDEX

# --- АДАПТИРОВАННАЯ select_embedding_ring ---
def select_embedding_ring(
        lowpass_subband: np.ndarray, rings_coords_np: List[Optional[np.ndarray]],
        metric: str = RING_SELECTION_METRIC, frame_number: int = -1
) -> int:
    """
    Выбирает наиболее подходящее кольцо для встраивания на основе метрики.
    Адаптировано для работы с NumPy массивами координат.
    """
    func_start_time = time.time()
    best_metric_value = -float('inf')
    selected_index = DEFAULT_RING_INDEX
    metric_values = []
    n_rings_available = len(rings_coords_np)
    known_metrics = ['entropy', 'energy', 'variance', 'mean_abs_dev']

    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Invalid lowpass_subband input for select_embedding_ring!")
        return DEFAULT_RING_INDEX

    metric_to_use = metric if metric in known_metrics else 'entropy'
    if metric not in known_metrics:
         logging.warning(f"[F:{frame_number}] Unknown metric '{metric}', defaulting to 'entropy'.")

    logging.debug(f"[F:{frame_number}] Selecting ring using metric: '{metric_to_use}'")

    for i, coords_np in enumerate(rings_coords_np):
        current_metric = -float('inf')
        if coords_np is None or coords_np.size == 0:
            metric_values.append(current_metric)
            continue
        try:
            if coords_np.ndim != 2 or coords_np.shape[1] != 2:
                 logging.warning(f"[F:{frame_number}, R:{i}] Invalid coords shape {coords_np.shape} for metric calc.")
                 metric_values.append(current_metric); continue

            # Используем NumPy индексацию
            rows, cols = coords_np[:, 0], coords_np[:, 1]
            ring_vals = lowpass_subband[rows, cols].astype(np.float32)

            if ring_vals.size == 0:
                 metric_values.append(current_metric); continue

            # Вычисление метрики
            if metric_to_use == 'entropy':
                visual_entropy, _ = calculate_entropies(ring_vals, frame_number, i)
                current_metric = visual_entropy
            elif metric_to_use == 'energy':
                current_metric = np.sum(ring_vals ** 2)
            elif metric_to_use == 'variance':
                current_metric = np.var(ring_vals)
            elif metric_to_use == 'mean_abs_dev':
                 mean_val = np.mean(ring_vals)
                 current_metric = np.mean(np.abs(ring_vals - mean_val))

            metric_values.append(current_metric)
            if current_metric > best_metric_value:
                 best_metric_value = current_metric
                 selected_index = i
        except IndexError:
             logging.error(f"[F:{frame_number}, R:{i}] IndexError calculating metric '{metric_to_use}'.", exc_info=False)
             metric_values.append(-float('inf'))
        except Exception as e:
            logging.error(f"[F:{frame_number}, R:{i}] Error calculating metric '{metric_to_use}': {e}", exc_info=False)
            metric_values.append(-float('inf'))

    metric_log_str = ", ".join([f"{i}:{v:.4f}" if v > -float('inf') else f"{i}:Err/Empty" for i, v in enumerate(metric_values)])
    logging.debug(f"[F:{frame_number}] Ring metrics calculated using '{metric_to_use}': [{metric_log_str}]")
    logging.info(f"[F:{frame_number}] Adaptive ring selection result: Ring={selected_index} (Value: {best_metric_value:.4f})")

    # Проверка валидности выбранного кольца
    if not (0 <= selected_index < n_rings_available and \
            rings_coords_np[selected_index] is not None and \
            rings_coords_np[selected_index].size > 0):
        logging.error(f"[F:{frame_number}] Selected ring {selected_index} is invalid or empty! Checking default {DEFAULT_RING_INDEX}.")
        if 0 <= DEFAULT_RING_INDEX < n_rings_available and \
           rings_coords_np[DEFAULT_RING_INDEX] is not None and \
           rings_coords_np[DEFAULT_RING_INDEX].size > 0:
            selected_index = DEFAULT_RING_INDEX
            logging.warning(f"[F:{frame_number}] Using default ring {selected_index}.")
        else:
            logging.warning(f"[F:{frame_number}] Default ring {DEFAULT_RING_INDEX} also invalid/empty. Searching...")
            found_non_empty = False
            for idx, coords_np_check in enumerate(rings_coords_np):
                if coords_np_check is not None and coords_np_check.size > 0:
                    selected_index = idx
                    logging.warning(f"[F:{frame_number}] Using first non-empty ring {selected_index}.")
                    found_non_empty = True
                    break
            if not found_non_empty:
                logging.critical(f"[F:{frame_number}] All rings are empty!")
                # ВАЖНО: Возвращаем дефолт, но это может вызвать ошибку позже, если и он пуст
                return DEFAULT_RING_INDEX
                # raise ValueError(f"Frame {frame_number}: All rings are empty, cannot select a ring.")

    logging.debug(f"[F:{frame_number}] Ring selection process time: {time.time() - func_start_time:.4f}s")
    return selected_index
# --- Конец адапт. select_embedding_ring ---

def calculate_perceptual_mask(input_plane: np.ndarray, frame_number: int = -1) -> Optional[np.ndarray]:
    # (Код calculate_perceptual_mask без изменений)
    if not isinstance(input_plane, np.ndarray) or input_plane.ndim != 2:
        logging.error(f"[F:{frame_number}] Invalid input for perceptual mask calculation.")
        return None
    try:
        plane_32f = input_plane.astype(np.float32)
        gx = cv2.Sobel(plane_32f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane_32f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        local_brightness = cv2.GaussianBlur(plane_32f, (11, 11), 5)
        brightness_mask = 1.0 - np.abs(local_brightness - 0.5) * 2.0
        mean_sq = cv2.GaussianBlur(plane_32f**2, (11, 11), 5)
        sq_mean = local_brightness**2
        local_variance = np.maximum(mean_sq - sq_mean, 0)
        local_stddev = np.sqrt(local_variance)
        eps = 1e-9
        max_grad = np.max(grad_mag)
        grad_norm = grad_mag / (max_grad + eps) if max_grad > eps else np.zeros_like(grad_mag)
        max_stddev = np.max(local_stddev)
        stddev_norm = local_stddev / (max_stddev + eps) if max_stddev > eps else np.zeros_like(local_stddev)
        w_grad = 0.4; w_texture = 0.4; w_brightness = 0.2
        mask = (grad_norm * w_grad + stddev_norm * w_texture + np.clip(brightness_mask, 0, 1) * w_brightness)
        max_mask = np.max(mask)
        mask_norm = mask / (max_mask + eps) if max_mask > eps else np.zeros_like(mask)
        logging.debug(f"[F:{frame_number}] Perceptual mask calculated. Shape: {mask_norm.shape}, Max: {np.max(mask_norm):.4f}")
        return mask_norm.astype(np.float32)
    except Exception as e:
        logging.error(f"[F:{frame_number}] Error calculating perceptual mask: {e}", exc_info=True)
        return np.ones_like(input_plane, dtype=np.float32)

# --- НОВАЯ ВЕКТОРИЗОВАННАЯ функция пространственного веса ---
def calculate_spatial_weights_vectorized(shape: Tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Вычисляет пространственные веса для набора координат."""
    h, w = shape
    center_r, center_c = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = np.sqrt(center_r**2 + center_c**2) + 1e-9
    dists = np.sqrt((rows - center_r)**2 + (cols - center_c)**2)
    norm_dists = dists / max_dist
    weights = 0.5 + 0.5 * norm_dists
    return np.clip(weights, 0.5, 1.0)
# --- Конец ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---

# ============================================================
# --- Функции Работы с Видео (I/O) ---
# ============================================================
# (Код read_video и write_video без изменений)
def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    func_start_time=time.time(); logging.info(f"Reading video from: {video_path}")
    frames=[]; fps=float(FPS); cap=None; expected_height,expected_width=-1,-1
    try:
        cap=cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed to open video: {video_path}"); return frames,fps
        fps_read=cap.get(cv2.CAP_PROP_FPS)
        if fps_read>0: fps=float(fps_read); logging.info(f"Detected FPS: {fps:.2f}")
        else: logging.warning(f"Failed to get FPS. Using default: {fps}")
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_prop=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); logging.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, ~{frame_count_prop} frames")
        expected_height,expected_width=height,width
        frame_index=0; read_count=0; none_frame_count=0; invalid_shape_count=0
        while True:
            ret,frame=cap.read(); frame_number_log=frame_index+1
            if not ret: logging.info(f"End of stream after reading {read_count} frames."); break
            if frame is None: logging.warning(f"Received None frame at index {frame_index}. Skipping."); none_frame_count+=1; frame_index+=1; continue
            if frame.ndim==3 and frame.shape[2]==3 and frame.dtype==np.uint8:
                 current_h,current_w=frame.shape[:2]
                 if current_h==expected_height and current_w==expected_width: frames.append(frame); read_count+=1;
                 else: logging.warning(f"Frame {frame_number_log} shape {(current_h,current_w)} != expected. Skipping."); invalid_shape_count+=1
            else: logging.warning(f"Frame {frame_number_log} not valid BGR uint8. Skipping."); invalid_shape_count+=1
            frame_index+=1
            # if frame_index > frame_count_prop * 1.5 and frame_count_prop > 0: logging.error(f"Read too many frames. Stopping."); break
        logging.info(f"Finished reading. Valid frames: {len(frames)}. Skipped None: {none_frame_count}, Invalid shape: {invalid_shape_count}")
    except Exception as e: logging.error(f"Exception during video reading: {e}",exc_info=True)
    finally:
        if cap is not None and cap.isOpened(): cap.release(); logging.debug("Video capture released.")
    if not frames: logging.error(f"No valid frames read from {video_path}")
    logging.debug(f"Read video time: {time.time()-func_start_time:.4f}s")
    return frames,fps

def write_video(frames: List[np.ndarray], out_path: str, fps: float, codec: str = OUTPUT_CODEC):
    func_start_time=time.time();
    if not frames: logging.error("No frames to write."); return
    logging.info(f"Starting video writing to: {out_path} (FPS: {fps:.2f}, Codec: {codec})")
    writer=None
    try:
        valid_frames_gen = (f for f in frames if f is not None and f.ndim==3 and f.shape[2]==3 and f.dtype==np.uint8)
        first_valid = next(valid_frames_gen, None)
        if first_valid is None: logging.error("No valid frames found to determine output size."); return
        h,w,c=first_valid.shape; logging.info(f"Output resolution: {w}x{h}")
        fourcc=cv2.VideoWriter_fourcc(*codec)
        base, _=os.path.splitext(out_path); out_path_corrected=base+OUTPUT_EXTENSION
        if out_path_corrected!=out_path: logging.info(f"Correcting output extension to '{OUTPUT_EXTENSION}'. New path: {out_path_corrected}"); out_path=out_path_corrected
        writer=cv2.VideoWriter(out_path, fourcc, fps, (w, h)); writer_codec_used=codec
        if not writer.isOpened():
            logging.error(f"Failed to create VideoWriter with codec '{codec}'.")
            if OUTPUT_EXTENSION.lower()=='.avi' and codec.upper()!='MJPG':
                 fallback_codec='MJPG'; logging.warning(f"Trying fallback codec '{fallback_codec}'.")
                 fourcc_fallback=cv2.VideoWriter_fourcc(*fallback_codec); writer=cv2.VideoWriter(out_path, fourcc_fallback, fps, (w,h))
                 if writer.isOpened(): logging.info(f"Using fallback codec '{fallback_codec}'."); writer_codec_used=fallback_codec
                 else: logging.critical(f"Fallback codec '{fallback_codec}' also failed."); return
            else: logging.critical(f"Cannot initialize VideoWriter."); return
        written_count=0; skipped_count=0; start_write_loop=time.time()
        black_frame = np.zeros((h,w,3),dtype=np.uint8)
        for i,frame in enumerate(frames):
            if frame is not None and frame.shape==(h,w,c) and frame.dtype==np.uint8:
                 writer.write(frame); written_count+=1;
            else:
                shape_info=frame.shape if frame is not None else 'None'; dtype_info=frame.dtype if frame is not None else 'N/A'
                logging.warning(f"Skipping invalid frame #{i+1}. Shape:{shape_info}, Dtype:{dtype_info}. Writing black frame instead.");
                writer.write(black_frame); skipped_count+=1
        logging.debug(f"Write loop time: {time.time()-start_write_loop:.4f}s")
        logging.info(f"Finished writing with codec '{writer_codec_used}'. Frames written: {written_count}, Skipped/Replaced with black: {skipped_count}")
    except Exception as e: logging.error(f"Exception during video writing: {e}",exc_info=True)
    finally:
        if writer is not None: writer.release(); logging.debug("Video writer released.")
    logging.debug(f"Write video total time: {time.time()-func_start_time:.4f}s")
# --- Конец Функций Работы с Видео ---

# ============================================================
# --- ЛОГИКА ВСТРАИВАНИЯ (Embed) ---
# ============================================================

# --- АДАПТИРОВАННАЯ embed_frame_pair ---
def embed_frame_pair(
        frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bit: int,
        n_rings: int = N_RINGS, ring_selection_method: str = RING_SELECTION_METHOD,
        ring_selection_metric: str = RING_SELECTION_METRIC,
        default_ring_index: int = DEFAULT_RING_INDEX, frame_number: int = 0,
        visualize_mask: bool = False, use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT, num_rings_to_use: int = NUM_RINGS_TO_USE
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Встраивает бит в пару кадров.
    Адаптировано для работы с NumPy массивами координат.
    """
    func_start_time = time.time(); pair_num_log = frame_number // 2
    logging.debug(f"--- Embed Start: Pair {pair_num_log} (Frame {frame_number}) ---")
    try:
        # 1. Проверки и конвертация YCrCb
        if frame1_bgr is None or frame2_bgr is None: logging.error(f"[P:{pair_num_log}] Input frame None."); return None, None, default_ring_index
        if frame1_bgr.shape != frame2_bgr.shape: logging.error(f"[P:{pair_num_log}] Frame shapes mismatch: {frame1_bgr.shape} vs {frame2_bgr.shape}"); return None, None, default_ring_index
        try:
            frame1_ycrcb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb)
            frame2_ycrcb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2YCrCb)
        except cv2.error as e: logging.error(f"[P:{pair_num_log}] Color conversion failed: {e}"); return None, None, default_ring_index

        comp_name = ['Y', 'Cr', 'Cb'][embed_component]; logging.debug(f"[P:{pair_num_log}] Using {comp_name} component")
        try:
            Y1_orig=frame1_ycrcb[:,:,0]; Cr1=frame1_ycrcb[:,:,1]; Cb1=frame1_ycrcb[:,:,2]
            Y2_orig=frame2_ycrcb[:,:,0]; Cr2=frame2_ycrcb[:,:,1]; Cb2=frame2_ycrcb[:,:,2]
            comp1 = frame1_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
            comp2 = frame2_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
        except IndexError: logging.error(f"[P:{pair_num_log}] Invalid embed_component index."); return None, None, default_ring_index

        # 2. DTCWT
        pyr1 = dtcwt_transform(comp1, frame_number=frame_number); pyr2 = dtcwt_transform(comp2, frame_number=frame_number + 1)
        if pyr1 is None or pyr2 is None or pyr1.lowpass is None or pyr2.lowpass is None: logging.error(f"[P:{pair_num_log}] DTCWT failed."); return None, None, default_ring_index
        L1 = pyr1.lowpass.copy(); L2 = pyr2.lowpass.copy()

        # 3. Получение координат колец (ИЗМЕНЕНО)
        rings1_coords_np = ring_division(L1, n_rings=n_rings, frame_number=frame_number)
        rings2_coords_np = ring_division(L2, n_rings=n_rings, frame_number=frame_number + 1)

        # 4. Выбор кольца/колец (АДАПТИРОВАНО)
        selected_rings_indices = []
        current_ring_index = default_ring_index

        if ring_selection_method == 'deterministic':
            current_ring_index = deterministic_ring_selection(frame1_bgr, n_rings, frame_number=frame_number)
            selected_rings_indices = [current_ring_index]
        elif ring_selection_method == 'keypoint':
             current_ring_index = keypoint_based_ring_selection(frame1_bgr, n_rings, frame_number=frame_number)
             selected_rings_indices = [current_ring_index]
        elif ring_selection_method == 'multi_ring':
             metric_values = []
             temp_metric = ring_selection_metric if ring_selection_metric in ['entropy', 'energy', 'variance', 'mean_abs_dev'] else 'entropy'
             for i_mr, coords_np_mr in enumerate(rings1_coords_np):
                 if coords_np_mr is not None and coords_np_mr.size > 0:
                     try:
                         rows_mr, cols_mr = coords_np_mr[:, 0], coords_np_mr[:, 1]
                         ring_vals_mr = L1[rows_mr, cols_mr].astype(np.float32)
                         if ring_vals_mr.size > 0:
                             if temp_metric == 'entropy': v_e_mr, _ = calculate_entropies(ring_vals_mr, frame_number, i_mr); metric_values.append((v_e_mr, i_mr))
                             elif temp_metric == 'energy': metric_values.append((np.sum(ring_vals_mr**2), i_mr))
                             elif temp_metric == 'variance': metric_values.append((np.var(ring_vals_mr), i_mr))
                             elif temp_metric == 'mean_abs_dev': mean_val_mr = np.mean(ring_vals_mr); metric_values.append((np.mean(np.abs(ring_vals_mr - mean_val_mr)), i_mr))
                             else: v_e_mr, _ = calculate_entropies(ring_vals_mr, frame_number, i_mr); metric_values.append((v_e_mr, i_mr))
                         else: metric_values.append((-float('inf'), i_mr))
                     except IndexError: metric_values.append((-float('inf'), i_mr))
                 else: metric_values.append((-float('inf'), i_mr))
             metric_values.sort(key=lambda x: x[0], reverse=True)
             selected_rings_indices = [idx for val, idx in metric_values[:num_rings_to_use] if val > -float('inf')]
             if not selected_rings_indices:
                 selected_rings_indices = [DEFAULT_RING_INDEX]
             current_ring_index = selected_rings_indices[0]
             logging.info(f"[P:{pair_num_log}] Multi-ring selected (metric '{temp_metric}'): {selected_rings_indices}")
        elif ring_selection_method == 'adaptive':
             # Используем адаптивную функцию, которая тоже должна быть обновлена
             current_ring_index = select_embedding_ring(L1, rings1_coords_np, metric=ring_selection_metric, frame_number=frame_number)
             selected_rings_indices = [current_ring_index]
        elif ring_selection_method == 'fixed':
             current_ring_index = default_ring_index
             selected_rings_indices = [current_ring_index]
             logging.info(f"[P:{pair_num_log}] Using fixed ring: {current_ring_index}")
        else:
             current_ring_index = default_ring_index
             selected_rings_indices = [current_ring_index]
             logging.error(f"Unknown ring selection method '{ring_selection_method}', using default.")

        # 5. Валидация выбранных колец (АДАПТИРОВАНО)
        valid_selected_rings = []
        for ring_idx in selected_rings_indices:
            if 0 <= ring_idx < n_rings and \
               ring_idx < len(rings1_coords_np) and rings1_coords_np[ring_idx] is not None and rings1_coords_np[ring_idx].size > 0 and \
               ring_idx < len(rings2_coords_np) and rings2_coords_np[ring_idx] is not None and rings2_coords_np[ring_idx].size > 0:
                valid_selected_rings.append(ring_idx)
            else:
                logging.warning(f"[P:{pair_num_log}] Selected ring {ring_idx} is invalid or empty in L1/L2. Removing.")

        # Логика Fallback (АДАПТИРОВАНО)
        if not valid_selected_rings:
            logging.error(f"[P:{pair_num_log}] No valid rings left from selection {selected_rings_indices}! Trying default {DEFAULT_RING_INDEX}.")
            ring_idx = DEFAULT_RING_INDEX
            if 0 <= ring_idx < n_rings and \
               ring_idx < len(rings1_coords_np) and rings1_coords_np[ring_idx] is not None and rings1_coords_np[ring_idx].size > 0 and \
               ring_idx < len(rings2_coords_np) and rings2_coords_np[ring_idx] is not None and rings2_coords_np[ring_idx].size > 0:
                 valid_selected_rings = [DEFAULT_RING_INDEX]
                 current_ring_index = DEFAULT_RING_INDEX
                 logging.warning(f"Using default ring {DEFAULT_RING_INDEX} as fallback.")
            else:
                 logging.warning(f"Default ring {DEFAULT_RING_INDEX} also invalid/empty. Searching...")
                 found_available = False
                 for idx in range(n_rings):
                      if idx < len(rings1_coords_np) and rings1_coords_np[idx] is not None and rings1_coords_np[idx].size > 0 and \
                         idx < len(rings2_coords_np) and rings2_coords_np[idx] is not None and rings2_coords_np[idx].size > 0:
                           valid_selected_rings = [idx]
                           current_ring_index = idx
                           logging.warning(f"Using first available ring {idx} as fallback.")
                           found_available = True
                           break
                 if not found_available:
                      logging.critical(f"[P:{pair_num_log}] No valid rings found at all. Cannot embed.");
                      return None, None, default_ring_index

        if current_ring_index not in valid_selected_rings:
            current_ring_index = valid_selected_rings[0] if valid_selected_rings else default_ring_index

        logging.info(f"[P:{pair_num_log}] Final rings for embedding: {valid_selected_rings} (Primary/Log Ring: {current_ring_index})")

        # 6. Вычисление перцептуальной маски
        perceptual_mask = None
        if use_perceptual_masking:
            # Используем comp1 (исходный компонент) для расчета маски
            perceptual_mask = calculate_perceptual_mask(comp1, frame_number=frame_number)
            if perceptual_mask is not None and perceptual_mask.shape != L1.shape:
                 perceptual_mask = cv2.resize(perceptual_mask, (L1.shape[1], L1.shape[0]), interpolation=cv2.INTER_LINEAR)
            elif perceptual_mask is None:
                 logging.warning(f"[P:{pair_num_log}] Failed to calculate perceptual mask, disabling.")

        # 7. Цикл встраивания по валидным кольцам (АДАПТИРОВАНО)
        for ring_idx in valid_selected_rings:
            try:
                coords_1_np = rings1_coords_np[ring_idx]
                coords_2_np = rings2_coords_np[ring_idx]
                rows1, cols1 = coords_1_np[:, 0], coords_1_np[:, 1]
                rows2, cols2 = coords_2_np[:, 0], coords_2_np[:, 1]
                ring_vals_1 = L1[rows1, cols1].astype(np.float32)
                ring_vals_2 = L2[rows2, cols2].astype(np.float32)
            except IndexError:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] IndexError extracting values for embedding.", exc_info=False); continue
            except Exception as e:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Error extracting vals for embedding: {e}", exc_info=False); continue

            if ring_vals_1.size == 0 or ring_vals_2.size == 0:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Empty ring values for embedding. Skipping."); continue

            alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_idx, frame_number)
            logging.info(f"[P:{pair_num_log}, R:{ring_idx}] Embedding Bit={bit}, Alpha={alpha:.4f}")

            dct1 = dct_1d(ring_vals_1); dct2 = dct_1d(ring_vals_2)
            try:
                U1, S1, Vt1 = svd(dct1.reshape(-1, 1), full_matrices=False)
                U2, S2, Vt2 = svd(dct2.reshape(-1, 1), full_matrices=False)
            except np.linalg.LinAlgError as e:
                logging.error(f"[P:{pair_num_log}, R:{ring_idx}] SVD failed: {e}. Skipping ring."); continue

            s1 = S1[0] if S1.size > 0 else 0.0; s2 = S2[0] if S2.size > 0 else 0.0
            eps = 1e-12; ratio = s1 / (s2 + eps); new_s1, new_s2 = s1, s2; modified = False
            alpha_sq = alpha * alpha; inv_alpha = 1.0 / (alpha + eps)

            if bit == 0: # Target: ratio' >= alpha
                if ratio < alpha:
                    new_s1 = (s1 * alpha_sq + alpha * s2) / (alpha_sq + 1.0)
                    new_s2 = (alpha * s1 + s2) / (alpha_sq + 1.0)
                    modified = True
            else: # bit == 1, Target: ratio' < 1/alpha
                if ratio >= inv_alpha:
                    new_s1 = (s1 + alpha * s2) / (1.0 + alpha_sq)
                    new_s2 = (alpha * s1 + alpha_sq * s2) / (1.0 + alpha_sq)
                    modified = True

            log_lvl = logging.INFO if modified else logging.DEBUG
            logging.log(log_lvl, f"[P:{pair_num_log}, R:{ring_idx}] SVD Mod Applied: {modified}. Orig s1={s1:.4f}, s2={s2:.4f}. New s1={new_s1:.4f}, s2={new_s2:.4f}. Target bit={bit}.")

            new_S1 = np.array([[new_s1]]) if S1.size > 0 else np.zeros((1, 1))
            new_S2 = np.array([[new_s2]]) if S2.size > 0 else np.zeros((1, 1))
            dct1_mod = (U1 @ new_S1 @ Vt1).flatten()
            dct2_mod = (U2 @ new_S2 @ Vt2).flatten()
            ring_vals_1_mod = idct_1d(dct1_mod)
            ring_vals_2_mod = idct_1d(dct2_mod)

            # Применение модификаций к L1 и L2 (ВЕКТОРИЗОВАНО)
            if len(ring_vals_1_mod) == len(rows1):
                delta1 = ring_vals_1_mod - ring_vals_1
                mod_factors1 = np.ones_like(delta1)
                if perceptual_mask is not None:
                    mask_vals1 = perceptual_mask[rows1, cols1]
                    mod_factors1 *= mask_vals1
                spatial_weights1 = calculate_spatial_weights_vectorized(L1.shape, rows1, cols1)
                mod_factors1 *= spatial_weights1
                L1[rows1, cols1] += delta1 * mod_factors1
            else: logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Length mismatch applying modif. to L1"); continue

            if len(ring_vals_2_mod) == len(rows2):
                delta2 = ring_vals_2_mod - ring_vals_2
                mod_factors2 = np.ones_like(delta2)
                if perceptual_mask is not None:
                     mask_vals2 = perceptual_mask[rows2, cols2]
                     mod_factors2 *= mask_vals2
                spatial_weights2 = calculate_spatial_weights_vectorized(L2.shape, rows2, cols2)
                mod_factors2 *= spatial_weights2
                L2[rows2, cols2] += delta2 * mod_factors2
            else: logging.error(f"[P:{pair_num_log}, R:{ring_idx}] Length mismatch applying modif. to L2"); continue
        # --- Конец цикла по кольцам ---

        # 8. Обратный DTCWT
        pyr1.lowpass = L1; pyr2.lowpass = L2
        comp1_mod = dtcwt_inverse(pyr1, frame_number=frame_number)
        comp2_mod = dtcwt_inverse(pyr2, frame_number=frame_number + 1)
        if comp1_mod is None or comp2_mod is None: logging.error(f"[P:{pair_num_log}] Inverse DTCWT failed."); return None, None, current_ring_index

        # 9. Проверка размера и ресайз
        target_shape = (Y1_orig.shape[0], Y1_orig.shape[1])
        if comp1_mod.shape != target_shape: comp1_mod = cv2.resize(comp1_mod, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR); logging.warning(f"[P:{pair_num_log}] Resized comp1 after inverse DTCWT")
        if comp2_mod.shape != target_shape: comp2_mod = cv2.resize(comp2_mod, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR); logging.warning(f"[P:{pair_num_log}] Resized comp2 after inverse DTCWT")

        # 10. Сборка BGR
        comp1_mod_scaled = np.clip(comp1_mod * 255.0, 0, 255).astype(np.uint8)
        comp2_mod_scaled = np.clip(comp2_mod * 255.0, 0, 255).astype(np.uint8)
        new_ycrcb1 = np.stack((Y1_orig, Cr1, Cb1), axis=-1)
        new_ycrcb2 = np.stack((Y2_orig, Cr2, Cb2), axis=-1)
        new_ycrcb1[:, :, embed_component] = comp1_mod_scaled
        new_ycrcb2[:, :, embed_component] = comp2_mod_scaled
        frame1_mod_bgr = cv2.cvtColor(new_ycrcb1, cv2.COLOR_YCrCb2BGR)
        frame2_mod_bgr = cv2.cvtColor(new_ycrcb2, cv2.COLOR_YCrCb2BGR)

        total_pair_time = time.time() - func_start_time
        logging.info(f"--- Embed Finish: Pair {pair_num_log}. Total Time: {total_pair_time:.4f} sec ---")
        return frame1_mod_bgr, frame2_mod_bgr, current_ring_index

    except Exception as e:
        pair_num_log_err = frame_number // 2 if frame_number >= 0 else -1
        logging.error(f"!!! UNHANDLED EXCEPTION in embed_frame_pair (Pair {pair_num_log_err}): {e}", exc_info=True)
        return None, None, default_ring_index

# --- _embed_frame_pair_worker --- (без изменений)
def _embed_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], int]:
    idx1 = args['idx1']; pair_num_log = idx1 // 2
    try:
        # Передаем все параметры как есть
        f1_mod, f2_mod, selected_ring = embed_frame_pair(
            args['frame1'], args['frame2'], args['bit'], args['n_rings'],
            args['ring_selection_method'], args['ring_selection_metric'],
            args['default_ring_index'], args['frame_number'],
            visualize_mask=args.get('visualize_mask', False),
            use_perceptual_masking=args.get('use_perceptual_masking', USE_PERCEPTUAL_MASKING),
            embed_component=args.get('embed_component', EMBED_COMPONENT),
            num_rings_to_use=args.get('num_rings_to_use', NUM_RINGS_TO_USE)
            )
        return idx1, f1_mod, f2_mod, selected_ring
    except Exception as e:
        # Логируем ошибку в воркере
        logging.error(f"Exception in worker for pair {pair_num_log} (Frame {idx1}): {e}", exc_info=True)
        # Возвращаем None для кадров и дефолтный индекс кольца
        return idx1, None, None, args['default_ring_index']

# --- embed_watermark_in_video --- (без изменений)
def embed_watermark_in_video(
        frames: List[np.ndarray], watermark_bits: List[int], n_rings: int = N_RINGS,
        ring_selection_method: str = RING_SELECTION_METHOD, ring_selection_metric: str = RING_SELECTION_METRIC,
        default_ring_index: int = DEFAULT_RING_INDEX, fps: float = FPS, max_workers: Optional[int] = MAX_WORKERS,
        visualize_masks: bool = False, use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT, num_rings_to_use: int = NUM_RINGS_TO_USE ):
    num_frames = len(frames); pair_count = num_frames // 2; bits_to_embed = len(watermark_bits)
    # Определяем количество пар для обработки
    num_pairs_to_process = min(pair_count, bits_to_embed)
    logging.info(f"Starting embedding of {bits_to_embed} bits into {num_pairs_to_process} frame pairs.")
    start_time = time.time()
    # Создаем копию списка кадров для модификации
    watermarked_frames = frames[:] # Важно создать копию!

    if num_pairs_to_process == 0:
        logging.warning("No frame pairs available or no bits to embed.")
        return watermarked_frames # Возвращаем оригинальные кадры

    tasks_args = []
    for pair_idx in range(num_pairs_to_process):
        idx1 = 2 * pair_idx
        idx2 = idx1 + 1
        # Проверяем наличие обоих кадров в паре
        if idx2 >= num_frames or frames[idx1] is None or frames[idx2] is None:
            logging.warning(f"Skipping pair {pair_idx} (Frames {idx1}, {idx2}) due to missing frames.")
            continue
        # Формируем аргументы для воркера
        args = {
            'idx1': idx1,
            'frame1': frames[idx1],
            'frame2': frames[idx2],
            'bit': watermark_bits[pair_idx],
            'n_rings': n_rings,
            'ring_selection_method': ring_selection_method,
            'ring_selection_metric': ring_selection_metric,
            'default_ring_index': default_ring_index,
            'frame_number': idx1,
            'visualize_mask': visualize_masks,
            'use_perceptual_masking': use_perceptual_masking,
            'embed_component': embed_component,
            'num_rings_to_use': num_rings_to_use
        }
        tasks_args.append(args)

    if not tasks_args:
        logging.error("No valid tasks created for embedding.")
        return watermarked_frames

    # Словарь для хранения результатов {индекс_первого_кадра: (кадр1_мод, кадр2_мод)}
    results: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    # Список для хранения выбранных колец (для сохранения в JSON)
    selected_rings_log: List[int] = [-1] * num_pairs_to_process # Инициализируем -1
    processed_count = 0; error_count = 0; task_count = len(tasks_args)

    try:
        logging.info(f"Submitting {task_count} tasks to ThreadPoolExecutor (max_workers={max_workers})...")
        # Используем ThreadPoolExecutor для параллельной обработки пар
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем словарь {future: индекс_первого_кадра}
            future_to_idx1 = {executor.submit(_embed_frame_pair_worker, arg): arg['idx1'] for arg in tasks_args}

            # Обрабатываем результаты по мере их поступления
            for i, future in enumerate(concurrent.futures.as_completed(future_to_idx1)):
                idx1 = future_to_idx1[future]
                pair_num_log = idx1 // 2
                try:
                    # Получаем результат из future
                    _, f1_mod, f2_mod, selected_ring = future.result()

                    # Записываем выбранное кольцо в лог
                    if 0 <= pair_num_log < len(selected_rings_log):
                        selected_rings_log[pair_num_log] = selected_ring
                    else:
                        logging.warning(f"Pair index {pair_num_log} out of bounds for selected_rings_log.")

                    # Сохраняем результат, если он валидный
                    if f1_mod is not None and f2_mod is not None:
                        results[idx1] = (f1_mod, f2_mod)
                        processed_count += 1
                        logging.debug(f"Pair {pair_num_log} completed ({i + 1}/{task_count}). Selected ring: {selected_ring}")
                    else:
                        # Если воркер вернул None, считаем это ошибкой
                        error_count += 1
                        logging.error(f"Pair {pair_num_log} (Frame {idx1}) failed processing (returned None).")
                except Exception as exc:
                    # Ловим исключения, возникшие внутри воркера
                    error_count += 1
                    logging.error(f'Pair {pair_num_log} (Frame {idx1}) generated exception: {exc}', exc_info=True)

    except Exception as e:
        # Ловим критические ошибки при работе с Executor
        logging.critical(f"CRITICAL ERROR during ThreadPoolExecutor execution: {e}", exc_info=True)
        # В этом случае лучше вернуть оригинальные кадры
        return frames[:]

    logging.info(f"ThreadPoolExecutor finished. Successful pairs processed: {processed_count}, Failed pairs: {error_count}.")

    # Применяем успешные результаты к списку кадров
    update_count = 0
    for idx1, (f1_mod, f2_mod) in results.items():
        idx2 = idx1 + 1
        # Проверяем, что индексы не выходят за пределы списка
        if idx1 < len(watermarked_frames):
            watermarked_frames[idx1] = f1_mod
            update_count += 1
        if idx2 < len(watermarked_frames):
            watermarked_frames[idx2] = f2_mod
            update_count += 1 # Считаем оба кадра как обновленные

    logging.info(f"Applied results from {len(results)} pairs to {update_count} frames in the list.")

    # --- Сохранение выбранных колец ---
    # Сохраняем только валидные индексы (>= 0)
    final_selected_rings = [r for r in selected_rings_log if r >= 0]
    if final_selected_rings:
        try:
            with open(SELECTED_RINGS_FILE, 'w') as f:
                json.dump(final_selected_rings, f, indent=4) # Добавим indent для читаемости
            logging.info(f"Saved {len(final_selected_rings)} selected ring indices to {SELECTED_RINGS_FILE}")
        except IOError as e:
            logging.error(f"Could not save selected rings to {SELECTED_RINGS_FILE}: {e}")
        except Exception as e:
             logging.error(f"Unexpected error saving rings JSON: {e}", exc_info=True)
    else:
        logging.warning("No valid selected rings recorded to save.")
    # --- Конец сохранения колец ---

    if visualize_masks:
        logging.info("Mask visualization was enabled, destroying windows...")
        cv2.destroyAllWindows()

    end_time = time.time()
    logging.info(f"Embedding process finished. Total time: {end_time - start_time:.2f} sec.")
    return watermarked_frames
# --- Конец Логики Встраивания ---


# ============================================================
# --- ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ ---
# ============================================================
def main():
    # (Код main без изменений)
    main_start_time = time.time()
    input_video = "input.mp4"
    base_output_name = "watermarked_video"
    output_video = base_output_name + OUTPUT_EXTENSION
    logging.info("--- Starting Embedding Main Process ---")

    frames, input_fps = read_video(input_video)
    if not frames:
        logging.critical("Failed to read input video. Exiting.")
        return

    fps_to_use = float(FPS) if input_fps <= 0 else input_fps
    if input_fps <= 0:
        logging.warning(f"Using default FPS={fps_to_use} for writing.")

    num_pairs = len(frames) // 2
    watermark_length_target = min(num_pairs, 128)
    if num_pairs == 0:
        logging.error("Not enough frames for any pairs.")
        return

    watermark_length = watermark_length_target if num_pairs >= watermark_length_target else num_pairs
    if watermark_length < watermark_length_target:
        logging.warning(
            f"Embedding {watermark_length} bits instead of {watermark_length_target} due to frame pairs limit.")

    watermark_bits = [random.randint(0, 1) for _ in range(watermark_length)]
    watermark_str = ''.join(map(str, watermark_bits))
    logging.info(f"Using watermark ({watermark_length} bits): {watermark_str}")

    try:
        with open(ORIGINAL_WATERMARK_FILE, "w") as f:
            f.write(watermark_str)
        logging.info(f"Original watermark saved to {ORIGINAL_WATERMARK_FILE}")
    except IOError as e:
        logging.error(f"Could not save original watermark: {e}")

    enable_mask_visualization = False
    watermarked_frames = embed_watermark_in_video(
        frames=frames,
        watermark_bits=watermark_bits,
        n_rings=N_RINGS,
        ring_selection_method=RING_SELECTION_METHOD,
        ring_selection_metric=RING_SELECTION_METRIC,
        default_ring_index=DEFAULT_RING_INDEX,
        fps=fps_to_use,
        max_workers=MAX_WORKERS,
        visualize_masks=enable_mask_visualization,
        use_perceptual_masking=USE_PERCEPTUAL_MASKING,
        embed_component=EMBED_COMPONENT,
        num_rings_to_use=NUM_RINGS_TO_USE
    )

    if watermarked_frames and len(watermarked_frames) == len(frames):
        write_video(watermarked_frames, output_video, fps=fps_to_use, codec=OUTPUT_CODEC)
        logging.info(f"Watermarked video saved to: {output_video}")
        try:
            if os.path.exists(output_video):
                file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                logging.info(f"Output file size: {file_size_mb:.2f} MB")
                if file_size_mb > 100 and OUTPUT_CODEC in ['HFYU', 'FFV1']:
                    logging.warning(f"Output file size large due to lossless codec '{OUTPUT_CODEC}'.")
                elif file_size_mb < 1 and OUTPUT_CODEC in ['XVID', 'mp4v']:
                    logging.warning(
                        f"Output file size very small with lossy codec '{OUTPUT_CODEC}'. Check quality/robustness.")
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
    print(f"Watermark: {ORIGINAL_WATERMARK_FILE}")
    print(f"Rings: {SELECTED_RINGS_FILE}")
    print("\nIMPORTANT: Please run the extractor and check the Bit Error Rate (BER).")
    if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG']:
        print(
            f"Since lossy codec '{OUTPUT_CODEC}' was used, you might need to increase ALPHA_MIN/ALPHA_MAX in the script if BER is high.")
# --- Конец Main ---

# --- Запуск с Профилированием ---
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    except ValueError as ve:
        logging.critical(f"Value Error: {ve}")
        print(f"\nERROR: {ve}.")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}. See {LOG_FILENAME}")
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        # Сохранение статистики в файл
        profile_file = "profile_stats_embed.txt"
        try:
            with open(profile_file, "w") as f:
                stats_file = pstats.Stats(profiler, stream=f)
                stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved to {profile_file}")
            print(f"Profiling stats saved to {profile_file}")
        except IOError as e:
            logging.error(f"Could not save profiling stats to {profile_file}: {e}")
# --- Конец Запуска ---