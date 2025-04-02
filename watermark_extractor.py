# -*- coding: utf-8 -*-
# Файл: extractor.py (улучшенная версия с оптимизированным ring_division и параллелизмом)

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
import functools # Для кэширования
import concurrent.futures # Для параллелизма
import cProfile
import pstats

# --- Константы (ДОЛЖНЫ СОВПАДАТЬ С EMBEDDER!) ---
LAMBDA_PARAM: float = 0.04
ALPHA_MIN: float = 1.005
ALPHA_MAX: float = 1.1
N_RINGS: int = 8
DEFAULT_RING_INDEX: int = 4
FPS: int = 30
LOG_FILENAME: str = 'watermarking_extract.log'
SELECTED_RINGS_FILE: str = 'selected_rings.json'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark.txt'
MAX_WORKERS_EXTRACT: Optional[int] = None # None для автоопределения, можно задать число

# --- Настройки Адаптивности (ВАЖНО: ДОЛЖНЫ СООТВЕТСТВОВАТЬ РЕЖИМУ EMBEDDING!) ---
RING_SELECTION_METHOD: str = 'deterministic' # 'deterministic', 'adaptive', 'keypoint', 'multi_ring', 'fixed'
RING_SELECTION_METRIC: str = 'entropy' # 'entropy', 'energy', 'variance', 'mean_abs_dev'
EMBED_COMPONENT: int = 1 # 0=Y, 1=Cr, 2=Cb
NUM_RINGS_TO_USE: int = 3 # Для 'multi_ring' при динамическом выборе
USE_SAVED_RINGS: bool = True  # Использовать сохраненные кольца из файла

# --- Настройка Видео Входа ---
INPUT_EXTENSION: str = '.avi'

# --- Настройка Логирования ---
# (Код настройки логирования без изменений)
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME, filemode='w', level=logging.INFO,
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
# logging.getLogger().setLevel(logging.DEBUG)

logging.info(f"--- Запуск Скрипта Извлечения (Оптимизированные Кольца, Параллельный) ---")
logging.info(f"Ожидаемые настройки эмбеддера: Метод='{RING_SELECTION_METHOD}', Метрика='{RING_SELECTION_METRIC if RING_SELECTION_METHOD in ['adaptive', 'multi_ring'] else 'N/A'}', N_RINGS={N_RINGS}")
logging.info(f"Альфа (ожид.): MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, Исп. сохр. кольца: {USE_SAVED_RINGS}")

# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Идентичные Embedder) ---
# ============================================================

def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    return dct(signal_1d, type=2, norm='ortho')

def dtcwt_transform(y_plane: np.ndarray, frame_number: int = -1) -> Optional[Pyramid]:
    # (Код dtcwt_transform - идентичен embedder.py)
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

# --- ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ РАБОТЫ С КОЛЬЦАМИ (Идентичные Embedder) ---
@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
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
        ring_bins = np.array([0.0, 1.0]); n_rings_eff = 1
    else:
        ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1); n_rings_eff = n_rings
    if len(ring_bins) < 2:
        logging.error(f"_ring_division_internal: Invalid bins!"); return [None] * n_rings
    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    rings_coords_np: List[Optional[np.ndarray]] = [None] * n_rings
    pixel_counts = np.zeros(n_rings, dtype=int)
    total_pixels_in_rings = 0
    for ring_idx in range(n_rings_eff):
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
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Input to ring_division is not a 2D numpy array! Type: {type(lowpass_subband)}")
        return [None] * n_rings
    shape = lowpass_subband.shape
    try:
        coords_list_np = get_ring_coords_cached(shape, n_rings)
        logging.debug(f"[F:{frame_number}] Using cached/calculated ring coords (type: {type(coords_list_np)}) for shape {shape}")
        if not isinstance(coords_list_np, list) or not all(isinstance(item, (np.ndarray, type(None))) for item in coords_list_np):
             logging.error(f"[F:{frame_number}] Cached ring division result has unexpected type. Recalculating.")
             get_ring_coords_cached.cache_clear()
             coords_list_np = _ring_division_internal(shape, n_rings)
        # Возвращаем копии, чтобы избежать изменения кэша извне (небольшое замедление)
        return [arr.copy() if arr is not None else None for arr in coords_list_np]
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception in ring_division or cache lookup: {e}", exc_info=True)
        return [None] * n_rings
# --- Конец оптимизированных функций колец ---

def calculate_entropies(ring_vals: np.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    # (Код calculate_entropies - идентичен embedder.py)
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
    # (Код compute_adaptive_alpha_entropy - идентичен embedder.py)
    if ring_vals.size == 0: logging.warning(f"[F:{frame_number}, R:{ring_index}] compute_adaptive_alpha empty ring_vals."); return ALPHA_MIN
    visual_entropy, edge_entropy = calculate_entropies(ring_vals, frame_number, ring_index)
    local_variance = np.var(ring_vals)
    texture_factor = 1.0 / (1.0 + np.clip(local_variance, 0, 1) * 10.0)
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
    # (Код deterministic_ring_selection - идентичен embedder.py)
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
    # (Код keypoint_based_ring_selection - идентичен embedder.py)
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

# --- АДАПТИРОВАННАЯ select_embedding_ring (Идентична Embedder) ---
def select_embedding_ring(
        lowpass_subband: np.ndarray, rings_coords_np: List[Optional[np.ndarray]],
        metric: str = RING_SELECTION_METRIC, frame_number: int = -1
) -> int:
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
            metric_values.append(current_metric); continue
        try:
            if coords_np.ndim != 2 or coords_np.shape[1] != 2:
                 logging.warning(f"[F:{frame_number}, R:{i}] Invalid coords shape {coords_np.shape} for metric calc.")
                 metric_values.append(current_metric); continue
            rows, cols = coords_np[:, 0], coords_np[:, 1]
            ring_vals = lowpass_subband[rows, cols].astype(np.float32)
            if ring_vals.size == 0:
                 metric_values.append(current_metric); continue
            if metric_to_use == 'entropy': v_e, _ = calculate_entropies(ring_vals, frame_number, i); current_metric = v_e
            elif metric_to_use == 'energy': current_metric = np.sum(ring_vals ** 2)
            elif metric_to_use == 'variance': current_metric = np.var(ring_vals)
            elif metric_to_use == 'mean_abs_dev': mean_val = np.mean(ring_vals); current_metric = np.mean(np.abs(ring_vals - mean_val))
            metric_values.append(current_metric)
            if current_metric > best_metric_value: best_metric_value = current_metric; selected_index = i
        except IndexError: logging.error(f"[F:{frame_number}, R:{i}] IndexError calculating metric '{metric_to_use}'.", exc_info=False); metric_values.append(-float('inf'))
        except Exception as e: logging.error(f"[F:{frame_number}, R:{i}] Error calculating metric '{metric_to_use}': {e}", exc_info=False); metric_values.append(-float('inf'))
    metric_log_str = ", ".join([f"{i}:{v:.4f}" if v > -float('inf') else f"{i}:Err/Empty" for i, v in enumerate(metric_values)])
    logging.debug(f"[F:{frame_number}] Ring metrics calculated using '{metric_to_use}': [{metric_log_str}]")
    logging.info(f"[F:{frame_number}] Adaptive ring selection result: Ring={selected_index} (Value: {best_metric_value:.4f})")
    if not (0 <= selected_index < n_rings_available and rings_coords_np[selected_index] is not None and rings_coords_np[selected_index].size > 0):
        logging.error(f"[F:{frame_number}] Selected ring {selected_index} is invalid or empty! Checking default {DEFAULT_RING_INDEX}.")
        if 0 <= DEFAULT_RING_INDEX < n_rings_available and rings_coords_np[DEFAULT_RING_INDEX] is not None and rings_coords_np[DEFAULT_RING_INDEX].size > 0:
            selected_index = DEFAULT_RING_INDEX; logging.warning(f"[F:{frame_number}] Using default ring {selected_index}.")
        else:
            logging.warning(f"[F:{frame_number}] Default ring {DEFAULT_RING_INDEX} also invalid/empty. Searching...")
            found_non_empty = False
            for idx, coords_np_check in enumerate(rings_coords_np):
                if coords_np_check is not None and coords_np_check.size > 0:
                    selected_index = idx; logging.warning(f"[F:{frame_number}] Using first non-empty ring {selected_index}."); found_non_empty = True; break
            if not found_non_empty: logging.critical(f"[F:{frame_number}] All rings are empty!"); return DEFAULT_RING_INDEX
    logging.debug(f"[F:{frame_number}] Ring selection process time: {time.time() - func_start_time:.4f}s")
    return selected_index
# --- Конец адапт. select_embedding_ring ---

def load_saved_rings() -> List[int]:
    # (Код load_saved_rings - идентичен embedder.py)
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
    except json.JSONDecodeError as e: logging.error(f"Error decoding JSON from {SELECTED_RINGS_FILE}: {e}"); return []
    except IOError as e: logging.error(f"Error reading saved rings file {SELECTED_RINGS_FILE}: {e}"); return []
    except Exception as e: logging.error(f"Unexpected error loading saved rings: {e}", exc_info=True); return []
# --- Конец ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ---

# ============================================================
# --- Функции Работы с Видео (Только Чтение) ---
# ============================================================
def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # (Код read_video - идентичен embedder.py)
    func_start_time = time.time(); logging.info(f"Reading video from: {video_path}")
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
            if not ret: logging.info(f"End of stream after reading {read_count} frames (index {frame_index})."); break
            if frame is None: logging.warning(f"Received None frame at index {frame_index}. Skipping."); none_frame_count+=1; frame_index+=1; continue
            if frame.ndim==3 and frame.shape[2]==3 and frame.dtype==np.uint8:
                 current_h,current_w=frame.shape[:2]
                 if current_h==expected_height and current_w==expected_width: frames.append(frame); read_count+=1;
                 else: logging.warning(f"Frame {frame_number_log} shape {(current_h,current_w)} != expected ({expected_height},{expected_width}). Skipping."); invalid_shape_count+=1
            else: logging.warning(f"Frame {frame_number_log} is not valid BGR image (shape: {frame.shape}, dtype: {frame.dtype}). Skipping."); invalid_shape_count+=1
            frame_index+=1
            # if frame_index > frame_count_prop * 1.5 and frame_count_prop > 0: logging.error(f"Read too many frames ({frame_index} vs {frame_count_prop}). Stopping."); break
        logging.info(f"Finished reading. Valid frames: {len(frames)}. Total processed: {frame_index}. Skipped None: {none_frame_count}, Invalid shape: {invalid_shape_count}")
    except Exception as e: logging.error(f"Exception during video reading: {e}", exc_info=True)
    finally:
        if cap is not None and cap.isOpened(): cap.release(); logging.debug("Video capture released.")
    if not frames: logging.error(f"No valid frames were read from {video_path}")
    logging.debug(f"Read video time: {time.time() - func_start_time:.4f}s")
    return frames, fps
# --- Конец Функций Работы с Видео ---

# ============================================================
# --- ЛОГИКА ИЗВЛЕЧЕНИЯ (Extract) ---
# ============================================================

# --- АДАПТИРОВАННАЯ extract_frame_pair ---
def extract_frame_pair(
        frame1: np.ndarray, frame2: np.ndarray, ring_index: int,
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    """
    Извлекает один бит из пары кадров, используя ЗАДАННЫЙ ring_index.
    Адаптировано для работы с NumPy массивами координат.
    """
    func_start_time = time.time()
    pair_num_log = frame_number // 2
    logging.debug(f"--- Extract Start: Pair {pair_num_log} (Frame {frame_number}), Target Ring: {ring_index} ---")
    try:
        # 1. Проверка входных данных
        if frame1 is None or frame2 is None: logging.error(f"[P:{pair_num_log}, F:{frame_number}] Input frame is None."); return None
        if frame1.shape != frame2.shape: logging.error(f"[P:{pair_num_log}, F:{frame_number}] Frame shapes mismatch: {frame1.shape} vs {frame2.shape}"); return None
        if frame1.dtype != np.uint8: frame1 = np.clip(frame1, 0, 255).astype(np.uint8); logging.warning(f"[P:{pair_num_log}] Frame 1 type was {frame1.dtype}, converted.")
        if frame2.dtype != np.uint8: frame2 = np.clip(frame2, 0, 255).astype(np.uint8); logging.warning(f"[P:{pair_num_log}] Frame 2 type was {frame2.dtype}, converted.")

        # 2. Цвет -> YCrCb
        try:
            frame1_ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
            frame2_ycrcb = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)
        except cv2.error as e: logging.error(f"[P:{pair_num_log}, F:{frame_number}] Color conversion failed: {e}"); return None

        # 3. Выбор компонента и нормализация
        comp_name = ['Y', 'Cr', 'Cb'][embed_component]
        logging.debug(f"[P:{pair_num_log}] Using {comp_name} component")
        try:
            comp1 = frame1_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
            comp2 = frame2_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
        except IndexError: logging.error(f"[P:{pair_num_log}, F:{frame_number}] Invalid embed_component index."); return None

        # 4. DTCWT
        pyr1 = dtcwt_transform(comp1, frame_number=frame_number)
        pyr2 = dtcwt_transform(comp2, frame_number=frame_number + 1)
        if pyr1 is None or pyr2 is None or pyr1.lowpass is None or pyr2.lowpass is None:
            logging.error(f"[P:{pair_num_log}, F:{frame_number}] DTCWT failed."); return None
        L1 = pyr1.lowpass
        L2 = pyr2.lowpass

        # 5. Получение координат НУЖНОГО кольца (ИЗМЕНЕНО)
        rings1_coords_np = ring_division(L1, n_rings=n_rings, frame_number=frame_number)
        rings2_coords_np = ring_division(L2, n_rings=n_rings, frame_number=frame_number + 1)

        # Проверка валидности целевого кольца
        if not (0 <= ring_index < n_rings and ring_index < len(rings1_coords_np) and ring_index < len(rings2_coords_np)):
            logging.error(f"[P:{pair_num_log}, F:{frame_number}] Invalid target ring_index {ring_index} (n_rings={n_rings})."); return None
        coords_1_np = rings1_coords_np[ring_index]
        coords_2_np = rings2_coords_np[ring_index]
        if coords_1_np is None or coords_1_np.size == 0 or coords_2_np is None or coords_2_np.size == 0:
             logging.error(f"[P:{pair_num_log}, F:{frame_number}] Target ring {ring_index} is empty/invalid."); return None
        if coords_1_np.ndim != 2 or coords_1_np.shape[1] != 2 or \
           coords_2_np.ndim != 2 or coords_2_np.shape[1] != 2:
             logging.error(f"[P:{pair_num_log}, R:{ring_index}] Invalid coordinate array shape."); return None

        # 6. Извлечение значений и расчет альфы/порога (ИЗМЕНЕНО)
        try:
             rows1, cols1 = coords_1_np[:, 0], coords_1_np[:, 1]
             ring_vals_1 = L1[rows1, cols1].astype(np.float32)
             rows2, cols2 = coords_2_np[:, 0], coords_2_np[:, 1]
             ring_vals_2 = L2[rows2, cols2].astype(np.float32)
        except IndexError:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] IndexError extracting ring values.", exc_info=False); return None
        except Exception as e:
             logging.error(f"[P:{pair_num_log}, R:{ring_index}] Error extracting ring values: {e}", exc_info=False); return None

        logging.debug(f"[P:{pair_num_log}, R:{ring_index}] Extracted {ring_vals_1.size} vals (F1), {ring_vals_2.size} vals (F2).")
        if ring_vals_1.size == 0 or ring_vals_2.size == 0:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] Extracted empty ring values."); return None

        # Расчет альфы и порога
        alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_index, frame_number)
        eps = 1e-12
        threshold = (alpha + 1.0 / (alpha + eps)) / 2.0
        logging.debug(f"[P:{pair_num_log}, R:{ring_index}] Using alpha={alpha:.4f}, threshold={threshold:.4f}")

        # 7. DCT, SVD, Вычисление отношения
        dct1 = dct_1d(ring_vals_1)
        dct2 = dct_1d(ring_vals_2)
        try:
            S1_vals = svd(dct1.reshape(-1, 1), compute_uv=False)
            S2_vals = svd(dct2.reshape(-1, 1), compute_uv=False)
        except np.linalg.LinAlgError as e:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] SVD failed: {e}."); return None

        s1 = S1_vals[0] if S1_vals.size > 0 else 0.0
        s2 = S2_vals[0] if S2_vals.size > 0 else 0.0
        ratio = s1 / (s2 + eps)

        # 8. Принятие решения
        bit_extracted = 0 if ratio >= threshold else 1
        logging.info(
            f"[P:{pair_num_log}, R:{ring_index}] s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f} vs threshold={threshold:.4f} -> Extracted Bit={bit_extracted}")

        total_pair_time = time.time() - func_start_time
        logging.debug(f"--- Extract Finish: Pair {pair_num_log}. Total Time: {total_pair_time:.4f} sec ---")
        return bit_extracted

    except Exception as e:
        pair_num_log_err = frame_number // 2 if frame_number >= 0 else -1
        logging.error(f"!!! UNHANDLED EXCEPTION in extract_frame_pair (Pair {pair_num_log_err}, Frame {frame_number}, Ring {ring_index}): {e}", exc_info=True)
        return None
# --- Конец extract_frame_pair ---

# --- Функция для multi-ring extraction (без изменений, использует новую extract_frame_pair) ---
def extract_frame_pair_multi_ring(
        frame1: np.ndarray, frame2: np.ndarray, ring_indices: List[int],
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    pair_num_log = frame_number // 2
    if not ring_indices:
        logging.error(f"[P:{pair_num_log}] No ring indices provided for multi-ring extraction")
        return None
    logging.info(f"[P:{pair_num_log}] Multi-ring extraction started for rings: {ring_indices}")
    bits = []
    for ring_idx in ring_indices:
        bit = extract_frame_pair(frame1, frame2, ring_idx, n_rings, frame_number, embed_component)
        if bit is not None:
            bits.append(bit)
            logging.debug(f"[P:{pair_num_log}] Multi-ring: Extracted bit={bit} from ring {ring_idx}")
        else:
             logging.warning(f"[P:{pair_num_log}] Multi-ring: Failed to extract bit from ring {ring_idx}. Skipping vote.")
    if not bits:
        logging.error(f"[P:{pair_num_log}] Multi-ring: Failed to extract any bits from rings {ring_indices}")
        return None
    zeros = bits.count(0); ones = bits.count(1)
    final_bit = 0 if zeros >= ones else 1
    logging.info(f"[P:{pair_num_log}] Multi-ring extraction result: votes: 0={zeros}, 1={ones} -> final_bit={final_bit}")
    return final_bit

# --- НОВЫЙ Воркер для параллельного извлечения ---
def _extract_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, Optional[int]]:
    """Обрабатывает одну пару кадров для извлечения бита."""
    pair_idx = args['pair_idx']
    frame1 = args['frame1']
    frame2 = args['frame2']
    n_rings = args['n_rings']
    embed_component = args['embed_component']
    effective_use_saved_rings = args['effective_use_saved_rings']
    ring_selection_method = args['ring_selection_method']
    ring_selection_metric = args['ring_selection_metric']
    default_ring_index = args['default_ring_index']
    num_rings_to_use = args['num_rings_to_use']
    saved_rings = args['saved_rings'] # List of saved rings
    frame_number = 2 * pair_idx # Base frame number for logging etc.

    bit_extracted = None
    try:
        target_ring_index = -1
        target_ring_indices = []

        # --- Определение кольца/колец ---
        if effective_use_saved_rings:
            if pair_idx < len(saved_rings):
                saved_ring = saved_rings[pair_idx]
                if 0 <= saved_ring < n_rings:
                    target_ring_index = saved_ring
                    target_ring_indices = [target_ring_index]
                    logging.info(f"[P:{pair_idx}] Worker using saved ring: {target_ring_index}")
                else:
                    logging.error(f"[P:{pair_idx}] Worker: Invalid saved ring index {saved_ring}. Skipping.")
                    return pair_idx, None
            else:
                logging.error(f"[P:{pair_idx}] Worker: Index out of bounds for saved rings. Skipping.")
                return pair_idx, None
        else:
            # Динамический выбор
            logging.debug(f"[P:{pair_idx}] Worker determining ring dynamically ('{ring_selection_method}').")
            try:
                # Получаем компонент и L1 для выбора кольца
                if embed_component == 0: comp1_sel = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
                elif embed_component == 1: comp1_sel = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 1].astype(np.float32) / 255.0
                else: comp1_sel = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 2].astype(np.float32) / 255.0

                pyr1_sel = dtcwt_transform(comp1_sel, frame_number=frame_number)
                if pyr1_sel is None or pyr1_sel.lowpass is None: raise RuntimeError("DTCWT failed for dynamic selection")
                L1_sel = pyr1_sel.lowpass
                rings_coords_sel_np = ring_division(L1_sel, n_rings=n_rings, frame_number=frame_number)

                # Выбираем кольцо(а) на основе метода
                if ring_selection_method == 'deterministic':
                    target_ring_index = deterministic_ring_selection(frame1, n_rings, frame_number=frame_number)
                    target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'keypoint':
                    target_ring_index = keypoint_based_ring_selection(frame1, n_rings, frame_number=frame_number)
                    target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'multi_ring':
                     metric_values = []
                     temp_metric = ring_selection_metric # Используем переданную метрику
                     for ring_i, coords_np in enumerate(rings_coords_sel_np):
                         if coords_np is not None and coords_np.size > 0:
                             try: # Добавим try-except для безопасности
                                 rows, cols = coords_np[:, 0], coords_np[:, 1]; ring_vals = L1_sel[rows, cols].astype(np.float32)
                                 if ring_vals.size > 0:
                                     if temp_metric == 'entropy': v_e, _ = calculate_entropies(ring_vals, frame_number, ring_i); metric_values.append((v_e, ring_i))
                                     elif temp_metric == 'energy': metric_values.append((np.sum(ring_vals**2), ring_i))
                                     elif temp_metric == 'variance': metric_values.append((np.var(ring_vals), ring_i))
                                     elif temp_metric == 'mean_abs_dev': mean_val = np.mean(ring_vals); metric_values.append((np.mean(np.abs(ring_vals - mean_val)), ring_i))
                                     else: v_e, _ = calculate_entropies(ring_vals, frame_number, ring_i); metric_values.append((v_e, ring_i)) # По умолчанию энтропия
                                 else: metric_values.append((-float('inf'), ring_i))
                             except IndexError: metric_values.append((-float('inf'), ring_i))
                         else: metric_values.append((-float('inf'), ring_i))
                     metric_values.sort(key=lambda x: x[0], reverse=True)
                     target_ring_indices = [idx for val, idx in metric_values[:num_rings_to_use] if val > -float('inf')]
                     if not target_ring_indices: target_ring_indices = [default_ring_index]
                     logging.info(f"[P:{pair_idx}] Worker multi-ring dynamic selection (metric '{temp_metric}'): {target_ring_indices}")
                elif ring_selection_method == 'adaptive':
                    # Используем обновленную select_embedding_ring
                    target_ring_index = select_embedding_ring(L1_sel, rings_coords_sel_np, metric=ring_selection_metric, frame_number=frame_number)
                    target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'fixed':
                    target_ring_index = default_ring_index
                    target_ring_indices = [target_ring_index]
                else:
                     logging.error(f"[P:{pair_idx}] Worker: Unknown dynamic method '{ring_selection_method}'. Using default.")
                     target_ring_index = default_ring_index
                     target_ring_indices = [target_ring_index]

            except Exception as sel_err:
                 logging.error(f"[P:{pair_idx}] Worker: Error during dynamic ring selection: {sel_err}. Using default.", exc_info=True)
                 target_ring_index = default_ring_index
                 target_ring_indices = [target_ring_index]
        # --- Конец определения кольца ---

        # --- Выполнение Извлечения ---
        if ring_selection_method == 'multi_ring' and not effective_use_saved_rings:
            # Вызываем multi-ring функцию, если метод multi-ring и кольца выбирались динамически
            bit_extracted = extract_frame_pair_multi_ring(frame1, frame2, ring_indices=target_ring_indices, n_rings=n_rings,
                                                frame_number=frame_number, embed_component=embed_component)
        elif target_ring_index != -1:
             # Вызываем single-ring функцию, если индекс определен (сохраненный или single-dynamic)
             bit_extracted = extract_frame_pair(frame1, frame2, ring_index=target_ring_index, n_rings=n_rings,
                                          frame_number=frame_number, embed_component=embed_component)
        elif target_ring_indices:
             # Случай, когда target_ring_indices не пуст, но target_ring_index остался -1
             # Это может быть multi-ring с сохраненными кольцами, ИЛИ если динамический multi-ring выбрал >1 кольца
             if len(target_ring_indices) == 1:
                 # Если динамический multi-ring выбрал только одно кольцо
                 target_ring_index = target_ring_indices[0]
                 bit_extracted = extract_frame_pair(frame1, frame2, ring_index=target_ring_index, n_rings=n_rings,
                                               frame_number=frame_number, embed_component=embed_component)
             else:
                 # Если колец несколько (либо сохраненные для multi-ring, либо динамические multi-ring)
                 logging.debug(f"[P:{pair_idx}] Calling multi-ring extraction for indices: {target_ring_indices}")
                 bit_extracted = extract_frame_pair_multi_ring(frame1, frame2, ring_indices=target_ring_indices, n_rings=n_rings,
                                                              frame_number=frame_number, embed_component=embed_component)
        else:
             # Не должно происходить из-за fallback логики, но на всякий случай
             logging.error(f"[P:{pair_idx}] Worker: No valid target ring determined.")
             bit_extracted = None
        # --- Конец Извлечения ---

        return pair_idx, bit_extracted

    except Exception as e:
        # Ловим любые другие ошибки внутри воркера
        logging.error(f"Exception in worker for pair {pair_idx}: {e}", exc_info=True)
        return pair_idx, None
# --- Конец Воркера ---

# --- ПАРАЛЛЕЛИЗОВАННАЯ extract_watermark_from_video ---
def extract_watermark_from_video(
        frames: List[np.ndarray], bit_count: int, n_rings: int = N_RINGS,
        ring_selection_method: str = RING_SELECTION_METHOD,
        ring_selection_metric: str = RING_SELECTION_METRIC,
        default_ring_index: int = DEFAULT_RING_INDEX,
        embed_component: int = EMBED_COMPONENT,
        num_rings_to_use: int = NUM_RINGS_TO_USE,
        use_saved_rings: bool = USE_SAVED_RINGS
) -> List[Optional[int]]:
    """
    Извлекает водяной знак из видео параллельно, используя ThreadPoolExecutor.
    """
    logging.info(f"Starting parallel extraction of {bit_count} bits.")
    logging.info(f"Ring Selection Method (dynamic): '{ring_selection_method}', Metric: '{ring_selection_metric}'")
    logging.info(f"Embedding component: {['Y', 'Cr', 'Cb'][embed_component]}, Use saved rings: {use_saved_rings}")
    start_time = time.time()
    extracted_bits: List[Optional[int]] = [None] * bit_count
    num_frames = len(frames)
    pair_count = num_frames // 2
    processed_pairs = 0; error_pairs = 0

    # Загрузка сохраненных колец
    saved_rings: List[int] = []
    effective_use_saved_rings = use_saved_rings
    if use_saved_rings:
        saved_rings = load_saved_rings()
        if saved_rings:
            logging.info(f"Successfully loaded {len(saved_rings)} saved rings.")
            if len(saved_rings) < bit_count: logging.warning(f"Saved rings count ({len(saved_rings)}) < expected bits ({bit_count}).")
            elif len(saved_rings) > bit_count: logging.warning(f"Saved rings count ({len(saved_rings)}) > expected bits ({bit_count}). Using first {bit_count}."); saved_rings = saved_rings[:bit_count]
        else:
            logging.warning("USE_SAVED_RINGS is True, but failed to load rings. Switching to dynamic selection.")
            effective_use_saved_rings = False

    max_extractable_bits = min(pair_count, bit_count)
    if max_extractable_bits < bit_count:
        logging.warning(f"Not enough frame pairs ({pair_count}) for {bit_count} bits. Extracting max {max_extractable_bits}.")

    logging.info(f"Attempting to extract {max_extractable_bits} bits from {pair_count} available pairs.")

    # Подготовка задач для параллельного выполнения
    tasks_args = []
    skipped_pairs_indices = [] # Индексы пар, пропущенных до запуска executor
    for i in range(max_extractable_bits):
        idx1 = 2 * i; idx2 = idx1 + 1
        if idx2 >= num_frames or frames[idx1] is None or frames[idx2] is None:
            logging.error(f"Frame {idx1} or {idx2} missing. Skipping pair {i} before submission.")
            if i < len(extracted_bits): extracted_bits[i] = None # Отмечаем как None
            skipped_pairs_indices.append(i)
            error_pairs += 1 # Считаем ошибкой
            continue

        args = {
            'pair_idx': i,
            'frame1': frames[idx1],
            'frame2': frames[idx2],
            'n_rings': n_rings,
            'embed_component': embed_component,
            'effective_use_saved_rings': effective_use_saved_rings,
            'ring_selection_method': ring_selection_method,
            'ring_selection_metric': ring_selection_metric,
            'default_ring_index': default_ring_index,
            'num_rings_to_use': num_rings_to_use,
            'saved_rings': saved_rings # Передаем список целиком
        }
        tasks_args.append(args)

    if not tasks_args:
        logging.error("No valid tasks created for parallel extraction.")
        # Заполняем оставшиеся None, если биты ожидались
        for i in range(max_extractable_bits):
             if i not in skipped_pairs_indices and i < len(extracted_bits):
                 extracted_bits[i] = None
        return extracted_bits

    # Параллельное выполнение
    # Можно попробовать ProcessPoolExecutor, если ThreadPoolExecutor не дает ускорения
    executor_class = concurrent.futures.ThreadPoolExecutor
    logging.info(f"Submitting {len(tasks_args)} extraction tasks to {executor_class.__name__} (max_workers={MAX_WORKERS_EXTRACT})...")

    try:
        with executor_class(max_workers=MAX_WORKERS_EXTRACT) as executor:
            future_to_pair_idx = {executor.submit(_extract_frame_pair_worker, arg): arg['pair_idx'] for arg in tasks_args}

            for future in concurrent.futures.as_completed(future_to_pair_idx):
                pair_idx = future_to_pair_idx[future]
                try:
                    _, bit_result = future.result() # Получаем результат (pair_idx, bit_extracted)
                    if 0 <= pair_idx < len(extracted_bits):
                        extracted_bits[pair_idx] = bit_result
                        if bit_result is None:
                            error_pairs += 1 # Считаем None как ошибку извлечения
                        processed_pairs += 1
                        logging.debug(f"Pair {pair_idx} completed. Result: {bit_result} ({processed_pairs}/{len(tasks_args)})")
                    else:
                         logging.error(f"Received result for out-of-bounds pair index {pair_idx}.")
                         error_pairs += 1
                except Exception as exc:
                    logging.error(f'Pair {pair_idx} generated exception in executor: {exc}', exc_info=True)
                    if 0 <= pair_idx < len(extracted_bits):
                        extracted_bits[pair_idx] = None # Помечаем как None при любой ошибке
                    error_pairs += 1
                    processed_pairs += 1 # Считаем обработанной, хоть и с ошибкой

    except Exception as e:
         logging.critical(f"CRITICAL ERROR during Executor execution: {e}", exc_info=True)
         # В случае критической ошибки, заполняем оставшиеся None
         for i in range(max_extractable_bits):
             if i not in skipped_pairs_indices and extracted_bits[i] is None: # Только те, что не были обработаны
                 extracted_bits[i] = None

    total_processed = processed_pairs + len(skipped_pairs_indices)
    end_time = time.time()
    logging.info(
        f"Parallel extraction finished. Pairs submitted: {len(tasks_args)}. "
        f"Pairs processed successfully (incl. skipped): {total_processed}. "
        f"Extraction errors (incl. skipped & None results): {error_pairs}. "
        f"Total time: {end_time - start_time:.2f} sec.")

    return extracted_bits
# --- Конец Логики Извлечения ---


# ============================================================
# --- ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ ---
# ============================================================
def main():
    # (Код main без изменений, кроме добавления MAX_WORKERS_EXTRACT в вызов extract...)
    main_start_time = time.time()
    input_video_base = "watermarked_video"
    input_video = input_video_base + INPUT_EXTENSION

    # Определение длины ВЗ (без изменений)
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
                logging.error(f"Content of {ORIGINAL_WATERMARK_FILE} is invalid: '{original_watermark_str}'."); original_watermark_str = None
        except IOError as e: logging.error(f"Could not read {ORIGINAL_WATERMARK_FILE}: {e}")
        except Exception as e: logging.error(f"Unexpected error reading {ORIGINAL_WATERMARK_FILE}: {e}")
    if expected_watermark_length == 0:
        default_length = 128 # Используем 128 как в эмбеддере
        logging.warning(f"{ORIGINAL_WATERMARK_FILE} not found/invalid. Assuming default length: {default_length}")
        expected_watermark_length = default_length

    logging.info("--- Starting Extraction Main Process ---")
    if not os.path.exists(input_video):
        logging.critical(f"Input video not found: '{input_video}'. Exiting."); print(f"\nERROR: Input video '{input_video}' not found."); return

    frames, input_fps = read_video(input_video)
    if not frames: logging.critical(f"Failed to read frames from {input_video}. Exiting."); return
    logging.info(f"Read {len(frames)} frames for extraction (Reported FPS: {input_fps:.2f})")

    # Извлекаем водяной знак (используя параллельную функцию)
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
        # max_workers_extract передается через константу MAX_WORKERS_EXTRACT в сам ThreadPoolExecutor
    )

    # Вывод результата (без изменений)
    valid_extracted_count = sum(1 for b in extracted_bits_result if b is not None)
    extracted_bits_str = "".join(str(b) if b is not None else '?' for b in extracted_bits_result)
    logging.info(f"Attempted to extract {expected_watermark_length} bits. Successfully extracted: {valid_extracted_count}")
    logging.info(f"Extracted watermark string ({len(extracted_bits_str)} bits): {extracted_bits_str}")
    print(f"\nExtraction Results:")
    print(f"  Attempted bits: {expected_watermark_length}")
    print(f"  Valid bits extracted: {valid_extracted_count}")
    print(f"  Extracted string ({len(extracted_bits_str)}): {extracted_bits_str}")

    # Сравнение с оригинальным ВЗ (без изменений)
    if original_watermark_str and len(original_watermark_str) == expected_watermark_length:
        print(f"  Original string ({len(original_watermark_str)}):  {original_watermark_str}")
        if len(extracted_bits_result) != expected_watermark_length:
             logging.warning("Length mismatch, cannot calculate BER.")
             print("\n  BER Calculation: Length mismatch.")
        else:
            error_count = 0; comparison_markers = []
            for i in range(expected_watermark_length):
                orig_bit = original_watermark_str[i]
                extr_bit = extracted_bits_result[i]
                if extr_bit is None: error_count += 1; comparison_markers.append("?")
                elif str(extr_bit) != orig_bit: error_count += 1; comparison_markers.append("X")
                else: comparison_markers.append("=")
            comparison_str = "".join(comparison_markers)
            ber = error_count / expected_watermark_length if expected_watermark_length > 0 else 0
            logging.info(f"Bit Error Rate (BER): {ber:.4f} ({error_count}/{expected_watermark_length} errors)")
            print(f"\n  Comparison (X/? = Error):")
            block_size = 64
            for i in range(0, expected_watermark_length, block_size):
                 print(f"    Orig: {original_watermark_str[i:i+block_size]}")
                 print(f"    Extr: {extracted_bits_str[i:i+block_size]}")
                 print(f"    Comp: {comparison_str[i:i+block_size]}")
            print(f"\n  Bit Error Rate (BER): {ber:.4f} ({error_count} errors / {expected_watermark_length} bits)")
            if error_count == 0: print("  >>> WATERMARK MATCH <<<")
            else: print("  >>> !!! WATERMARK MISMATCH / ERRORS DETECTED !!! <<<")
    else:
        logging.warning("Original watermark unavailable for BER calculation.")
        print("\n  Original watermark not available for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_script_time = time.time() - main_start_time
    logging.info(f"--- Total Extractor Script Time: {total_script_time:.2f} sec ---")
    print(f"\nExtraction finished. Check log: {LOG_FILENAME}")
# --- Конец Main ---


# --- Запуск с Профилированием ---
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception in main (Extractor): {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}. Check the log file: {LOG_FILENAME}")
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        # Сохранение статистики в файл
        profile_file = "profile_stats_extract.txt"
        try:
            with open(profile_file, "w") as f:
                stats_file = pstats.Stats(profiler, stream=f)
                stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved to {profile_file}")
            print(f"Profiling stats saved to {profile_file}")
        except IOError as e:
             logging.error(f"Could not save profiling stats to {profile_file}: {e}")
# --- Конец Запуска ---