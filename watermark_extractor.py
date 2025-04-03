# -*- coding: utf-8 -*-
# Файл: extractor.py (Версия N=2, 64bit ID, Max 5 repeats, Packet Voting)
import random

import cv2
import numpy as np
import logging
import time
import json
import os
import imagehash
from PIL import Image
from scipy.fftpack import dct
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools
import concurrent.futures
import uuid # Используем только для типа UUID при конвертации
from math import ceil
try:
    import bchlib
    BCHLIB_AVAILABLE = True
except ImportError:
    BCHLIB_AVAILABLE = False
    # Логирование ошибки произойдет при проверке
import cProfile
import pstats
from collections import Counter # Для голосования

# --- Константы (ДОЛЖНЫ СОВПАДАТЬ С EMBEDDER!) ---
LAMBDA_PARAM: float = 0.04 # Используется ли LAMBDA в extractor? Обычно нет.
ALPHA_MIN: float = 1.005
ALPHA_MAX: float = 1.1
N_RINGS: int = 8
DEFAULT_RING_INDEX: int = 4
FPS: int = 30
LOG_FILENAME: str = 'watermarking_extract.log'
SELECTED_RINGS_FILE: str = 'selected_rings.json'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt' # Изменено имя
MAX_WORKERS_EXTRACT: Optional[int] = None

# --- Настройки Адаптивности (ВАЖНО: ДОЛЖНЫ СООТВЕТСТВОВАТЬ EMBEDDER!) ---
BITS_PER_PAIR: int = 2 # <<<< ИЗМЕНЕНО
RING_SELECTION_METHOD: str = 'multi_ring' if BITS_PER_PAIR > 1 else 'deterministic'
NUM_RINGS_TO_USE: int = BITS_PER_PAIR if BITS_PER_PAIR > 1 else 1
RING_SELECTION_METRIC: str = 'entropy' # Метрика для динамического выбора (если USE_SAVED_RINGS = False)
EMBED_COMPONENT: int = 1 # 0=Y, 1=Cr, 2=Cb
USE_SAVED_RINGS: bool = True # Предпочтительно True для надежности

# --- Настройки Извлечения и ECC (ВАЖНО: ДОЛЖНЫ СООТВЕТСТВОВАТЬ EMBEDDER!) ---
PAYLOAD_LEN_BYTES: int = 8 # <<<< ИЗМЕНЕНО: 64-битный ID
USE_ECC: bool = True # Ожидается ли ECC?
BCH_M: int = 8
BCH_T: int = 5
MAX_PACKET_REPEATS: int = 5 # <<<< Используется для определения, сколько пар *ожидать*

# --- Настройка Видео Входа ---
INPUT_EXTENSION: str = '.avi'

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME, filemode='w', level=logging.INFO,
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
# logging.getLogger().setLevel(logging.DEBUG) # Раскомментировать для DEBUG

effective_use_ecc = USE_ECC and BCHLIB_AVAILABLE
logging.info(f"--- Запуск Скрипта Извлечения (Bits/Pair: {BITS_PER_PAIR}, Payload: {PAYLOAD_LEN_BYTES*8}bit, ECC Ожид.: {USE_ECC}, Доступно: {BCHLIB_AVAILABLE}, Max Repeats: {MAX_PACKET_REPEATS}) ---")
logging.info(f"Ожид. настройки эмбеддера: Метод='{RING_SELECTION_METHOD}', Метрика='{RING_SELECTION_METRIC}', Колец исп.={NUM_RINGS_TO_USE}, N_RINGS_Total={N_RINGS}")
logging.info(f"Альфа (ожид.): MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, Исп. сохр. кольца: {USE_SAVED_RINGS}")
if effective_use_ecc:
    # Логируем параметры BCH ПОСЛЕ инициализации объекта bch
    pass
elif USE_ECC and not BCHLIB_AVAILABLE:
    logging.warning("ECC ожидается (USE_ECC=True), но bchlib не доступна! Декодирование ECC будет невозможно.")
else:
     logging.info("ECC не ожидается (USE_ECC=False).")


# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Идентичные Embedder + load/decode) ---
# ============================================================
# (dct_1d, dtcwt_transform, _ring_division_internal,
# get_ring_coords_cached, ring_division, calculate_entropies,
# compute_adaptive_alpha_entropy, deterministic_ring_selection,
# keypoint_based_ring_selection, select_embedding_rings - ИЗМЕНЕНА)
def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    return dct(signal_1d, type=2, norm='ortho')

def dtcwt_transform(y_plane: np.ndarray, frame_number: int = -1) -> Optional[Pyramid]:
    # (Код без изменений)
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

@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    # (Код без изменений)
    func_start_time = time.time(); H, W = subband_shape
    if H < 2 or W < 2: logging.error(f"_ring_division_internal: Subband too small: {H}x{W}."); return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0; rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2); min_dist, max_dist = np.min(distances), np.max(distances)
    if max_dist < 1e-6: ring_bins = np.array([0.0, 1.0]); n_rings_eff = 1
    else: ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1); n_rings_eff = n_rings
    if len(ring_bins) < 2: logging.error(f"_ring_division_internal: Invalid bins!"); return [None] * n_rings
    ring_indices = np.digitize(distances, ring_bins) - 1; ring_indices[distances < ring_bins[1]] = 0; ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    rings_coords_np: List[Optional[np.ndarray]] = [None] * n_rings; pixel_counts = np.zeros(n_rings, dtype=int); total_pixels_in_rings = 0
    for ring_idx in range(n_rings_eff):
        coords_for_ring_np = np.argwhere(ring_indices == ring_idx); count = coords_for_ring_np.shape[0]
        if count > 0: rings_coords_np[ring_idx] = coords_for_ring_np; pixel_counts[ring_idx] = count; total_pixels_in_rings += count
    total_pixels_in_subband = H * W
    if total_pixels_in_rings != total_pixels_in_subband: logging.debug(f"_ring_division_internal: Pixel count mismatch! Rings: {total_pixels_in_rings}, Subband: {total_pixels_in_subband}. Shape: {H}x{W}")
    logging.debug(f"_ring_division_internal calc time for shape {subband_shape}: {time.time() - func_start_time:.6f}s. Ring pixels: {pixel_counts[:n_rings_eff]}")
    return rings_coords_np

@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    # (Код без изменений)
    logging.debug(f"Cache miss for ring_division shape={subband_shape}, n_rings={n_rings}. Calculating...")
    return _ring_division_internal(subband_shape, n_rings)

def ring_division(lowpass_subband: np.ndarray, n_rings: int = N_RINGS, frame_number: int = -1) -> List[Optional[np.ndarray]]:
    # (Код без изменений)
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Input to ring_division is not a 2D numpy array! Type: {type(lowpass_subband)}")
        return [None] * n_rings
    shape = lowpass_subband.shape
    try:
        coords_list_np = get_ring_coords_cached(shape, n_rings)
        logging.debug(f"[F:{frame_number}] Using cached/calculated ring coords (type: {type(coords_list_np)}) for shape {shape}")
        if not isinstance(coords_list_np, list) or not all(isinstance(item, (np.ndarray, type(None))) for item in coords_list_np):
             logging.error(f"[F:{frame_number}] Cached ring division result has unexpected type. Recalculating.")
             get_ring_coords_cached.cache_clear(); coords_list_np = _ring_division_internal(shape, n_rings)
        return [arr.copy() if arr is not None else None for arr in coords_list_np] # Возвращаем копии
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception in ring_division or cache lookup: {e}", exc_info=True); return [None] * n_rings

def calculate_entropies(ring_vals: np.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    # (Код без изменений)
    eps=1e-12;
    if ring_vals.size==0: return 0.0, 0.0
    min_v, max_v = np.min(ring_vals), np.max(ring_vals)
    if min_v < 0.0 or max_v > 1.0:
        logging.debug(f"[F:{frame_number}, R:{ring_index}] Ring values out of [0,1] range ({min_v:.2f}, {max_v:.2f}). Clipping.")
        ring_vals_clipped = np.clip(ring_vals, 0.0, 1.0)
    else:
        ring_vals_clipped = ring_vals
    hist, _ =np.histogram(ring_vals_clipped, bins=256, range=(0.0,1.0), density=False)
    total_count=ring_vals_clipped.size;
    if total_count == 0: return 0.0, 0.0
    probabilities=hist/total_count; probabilities=probabilities[probabilities>eps]
    if probabilities.size==0: return 0.0, 0.0
    visual_entropy=-np.sum(probabilities*np.log2(probabilities)); edge_entropy=-np.sum(probabilities*np.exp(1.0-probabilities)); return visual_entropy, edge_entropy

def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    # (Код без изменений - но используется в extract_frame_pair для порога)
    if ring_vals.size == 0: logging.warning(f"[F:{frame_number}, R:{ring_index}] compute_adaptive_alpha empty ring_vals."); return ALPHA_MIN
    visual_entropy, edge_entropy = calculate_entropies(ring_vals, frame_number, ring_index); local_variance = np.var(ring_vals)
    texture_factor = 1.0 / (1.0 + np.clip(local_variance, 0, 1) * 10.0)
    eps = 1e-12
    if abs(visual_entropy) < eps: entropy_ratio = 0.0; logging.debug(f"[F:{frame_number}, R:{ring_index}] Visual entropy near zero.")
    else: entropy_ratio = edge_entropy / visual_entropy
    sigmoid_input = entropy_ratio; sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid_output * (1.0 - texture_factor * 0.5)
    final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)
    logging.debug(f"[F:{frame_number}, R:{ring_index}] Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, Var={local_variance:.4f}, TxtrF={texture_factor:.4f}, Ratio={entropy_ratio:.4f}, Sig={sigmoid_output:.4f} -> final_alpha={final_alpha:.4f}")
    return final_alpha

def deterministic_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    # (Код без изменений)
    try:
        if frame.ndim == 3 and frame.shape[2] == 3: gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2: gray_frame = frame
        else: logging.error(f"[F:{frame_number}] Invalid frame dim for hashing."); return random.randrange(n_rings)
        small_frame = cv2.resize(gray_frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(small_frame); phash = imagehash.phash(pil_img); hash_str = str(phash)
        if not hash_str: logging.warning(f"[F:{frame_number}] Empty phash."); return random.randrange(n_rings)
        try: hash_int = int(hash_str, 16)
        except ValueError: logging.error(f"[F:{frame_number}] Invalid hash format '{hash_str}'."); return random.randrange(n_rings)
        selected_ring = hash_int % n_rings; logging.debug(f"[F:{frame_number}] Deterministic ring: hash={hash_str}, ring={selected_ring}"); return selected_ring
    except Exception as e: logging.error(f"[F:{frame_number}] Error in deterministic_ring_selection: {e}", exc_info=True); return random.randrange(n_rings)

def keypoint_based_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
    # (Код без изменений)
    try:
        if frame.ndim == 3 and frame.shape[2] == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2: gray = frame
        else: logging.error(f"[F:{frame_number}] Invalid frame dim for keypoints."); return random.randrange(n_rings)
        fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True); keypoints = fast.detect(gray, None)
        if not keypoints: logging.warning(f"[F:{frame_number}] No FAST keypoints, using default."); return random.randrange(n_rings)
        num_keypoints = len(keypoints); x_avg = sum(kp.pt[0] for kp in keypoints)/num_keypoints; y_avg = sum(kp.pt[1] for kp in keypoints)/num_keypoints
        h, w = gray.shape[:2]; x_norm = x_avg/w if w>0 else 0.5; y_norm = y_avg/h if h>0 else 0.5
        dist = np.sqrt((x_norm-0.5)**2 + (y_norm-0.5)**2); selected_ring = int((dist/0.5)*n_rings) if dist>0 else 0; selected_ring = max(0, min(selected_ring, n_rings-1))
        logging.debug(f"[F:{frame_number}] Keypoint-based ring: kpts={num_keypoints}, dist={dist:.3f}, ring={selected_ring}"); return selected_ring
    except Exception as e: logging.error(f"[F:{frame_number}] Error in keypoint_based_ring_selection: {e}", exc_info=True); return random.randrange(n_rings)

# --- ИЗМЕНЕННАЯ select_embedding_rings (как в embedder.py) ---
def select_embedding_rings(
        lowpass_subband: np.ndarray, rings_coords_np: List[Optional[np.ndarray]],
        num_to_select: int = NUM_RINGS_TO_USE, metric: str = RING_SELECTION_METRIC,
        frame_number: int = -1, min_pixels: int = 10
) -> List[int]:
    # (Код идентичен embedder.py)
    func_start_time = time.time()
    metric_values: List[Tuple[float, int]] = [] # (значение_метрики, индекс_кольца)
    n_rings_available = len(rings_coords_np); known_metrics = ['entropy', 'energy', 'variance', 'mean_abs_dev']
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"[F:{frame_number}] Invalid lowpass_subband input for ring selection!"); return []
    if num_to_select <= 0: return []
    metric_to_use = metric if metric in known_metrics else 'entropy'
    if metric not in known_metrics: logging.warning(f"[F:{frame_number}] Unknown metric '{metric}', defaulting to 'entropy'.")
    logging.debug(f"[F:{frame_number}] Selecting {num_to_select} rings using metric: '{metric_to_use}'")
    for i, coords_np in enumerate(rings_coords_np):
        current_metric = -float('inf')
        if coords_np is None or coords_np.size < min_pixels * 2: metric_values.append((current_metric, i)); continue
        try:
            if coords_np.ndim != 2 or coords_np.shape[1] != 2: logging.warning(f"[F:{frame_number}, R:{i}] Invalid coords shape."); metric_values.append((current_metric, i)); continue
            rows, cols = coords_np[:, 0], coords_np[:, 1]; ring_vals = lowpass_subband[rows, cols].astype(np.float32)
            if ring_vals.size < min_pixels: metric_values.append((current_metric, i)); continue
            if metric_to_use == 'entropy': v_e, _ = calculate_entropies(ring_vals, frame_number, i); current_metric = v_e
            elif metric_to_use == 'energy': current_metric = np.sum(ring_vals ** 2)
            elif metric_to_use == 'variance': current_metric = np.var(ring_vals)
            elif metric_to_use == 'mean_abs_dev': mean_val = np.mean(ring_vals); current_metric = np.mean(np.abs(ring_vals - mean_val))
            if not np.isfinite(current_metric): logging.warning(f"[F:{frame_number}, R:{i}] Metric is not finite ({current_metric}). Skipping."); current_metric = -float('inf')
            metric_values.append((current_metric, i))
        except IndexError: logging.error(f"[F:{frame_number}, R:{i}] IndexError calculating metric.", exc_info=False); metric_values.append((-float('inf'), i))
        except Exception as e: logging.error(f"[F:{frame_number}, R:{i}] Error calculating metric: {e}", exc_info=False); metric_values.append((-float('inf'), i))
    metric_values.sort(key=lambda x: x[0], reverse=True)
    selected_indices = [idx for val, idx in metric_values if val > -float('inf')][:num_to_select]
    metric_log_str = ", ".join([f"{idx}:{val:.4f}" if val > -float('inf') else f"{idx}:Err/Empty" for val, idx in sorted(metric_values, key=lambda x: x[1])])
    logging.debug(f"[F:{frame_number}] Ring metrics ('{metric_to_use}'): [{metric_log_str}]")
    logging.info(f"[F:{frame_number}] Multi-ring selection result: {selected_indices}")
    if len(selected_indices) < num_to_select:
        logging.warning(f"[F:{frame_number}] Only selected {len(selected_indices)}/{num_to_select} rings. Trying to add fallbacks.")
        fallback_candidates = [idx for val, idx in metric_values if val > -float('inf') and idx not in selected_indices]
        needed = num_to_select - len(selected_indices); selected_indices.extend(fallback_candidates[:needed])
        if len(selected_indices) < num_to_select and DEFAULT_RING_INDEX not in selected_indices:
             if 0 <= DEFAULT_RING_INDEX < n_rings_available and rings_coords_np[DEFAULT_RING_INDEX] is not None and rings_coords_np[DEFAULT_RING_INDEX].size >= min_pixels * 2:
                  logging.warning(f"[F:{frame_number}] Adding default ring {DEFAULT_RING_INDEX} as fallback."); selected_indices.append(DEFAULT_RING_INDEX)
        if len(selected_indices) < num_to_select:
             for idx in range(n_rings_available):
                 if idx not in selected_indices and rings_coords_np[idx] is not None and rings_coords_np[idx].size >= min_pixels * 2:
                     logging.warning(f"[F:{frame_number}] Adding first available ring {idx} as fallback."); selected_indices.append(idx);
                     if len(selected_indices) == num_to_select: break
    if len(selected_indices) < num_to_select: logging.error(f"[F:{frame_number}] Could not select {num_to_select} valid rings! Selected only: {selected_indices}")
    logging.debug(f"[F:{frame_number}] Ring selection process time: {time.time() - func_start_time:.4f}s")
    return selected_indices[:num_to_select]

# --- ИЗМЕНЕННАЯ load_saved_rings (загружает словарь) ---
def load_saved_rings() -> Dict[int, List[int]]:
    """Загружает словарь {pair_idx: [ring1, ring2]} из JSON файла."""
    rings_data: Dict[int, List[int]] = {}
    if not os.path.exists(SELECTED_RINGS_FILE):
        logging.warning(f"Saved rings file '{SELECTED_RINGS_FILE}' not found.")
        return rings_data
    try:
        with open(SELECTED_RINGS_FILE, 'r') as f:
            loaded_data = json.load(f)
        # Преобразуем ключи обратно в int и проверяем формат значений
        valid_count = 0
        for key_str, value_list in loaded_data.items():
            try:
                pair_idx = int(key_str)
                if isinstance(value_list, list) and all(isinstance(r, int) for r in value_list):
                    rings_data[pair_idx] = value_list
                    valid_count += 1
                else:
                    logging.warning(f"Invalid value format for pair {pair_idx} in {SELECTED_RINGS_FILE}: {value_list}")
            except ValueError:
                logging.warning(f"Invalid key format in {SELECTED_RINGS_FILE}: {key_str}")
        logging.info(f"Loaded {valid_count} valid ring entries from {SELECTED_RINGS_FILE}")
        return rings_data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {SELECTED_RINGS_FILE}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error loading/parsing saved rings: {e}", exc_info=True)
        return {}

def bits_to_bytes(bits: List[Optional[int]]) -> Optional[bytearray]:
    # (Код без изменений)
    valid_bits = [b for b in bits if b is not None]
    if not valid_bits: logging.warning("No valid bits to convert to bytes."); return bytearray()
    remainder = len(valid_bits) % 8
    if remainder != 0:
        # Паддинг ДОЛЖЕН соответствовать паддингу при кодировании.
        # Если ECC добавлялся, packbits в add_ecc мог добавить нули.
        # Если мы извлекли не кратное 8 число бит, нужно решить, как падать.
        # Лучше всего - падать нулями в конце, как делает np.packbits.
        logging.warning(f"Number of valid bits ({len(valid_bits)}) not multiple of 8. Padding with {8-remainder} zeros at the end.")
        valid_bits.extend([0] * (8 - remainder))
    byte_array = bytearray()
    for i in range(0, len(valid_bits), 8):
        byte_bits = valid_bits[i:i+8]
        try: byte_val = int("".join(map(str, byte_bits)), 2); byte_array.append(byte_val)
        except ValueError: logging.error(f"Error converting bits to byte: {byte_bits}"); return None
        except TypeError: logging.error(f"Non-integer found in bits: {byte_bits}"); return None # Добавлено
    return byte_array

def decode_ecc(packet_bytes: bytearray, bch: 'bchlib.BCH', expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    """
    Декодирует пакет с помощью BCH.
    Возвращает: (декодированные байты данных ИЛИ None, количество исправленных ошибок ИЛИ -1).
    """
    if not BCHLIB_AVAILABLE or bch is None:
        logging.warning("bchlib not available or not initialized, skipping ECC decoding.")
        # Возвращаем начало пакета как данные, если длина позволяет
        if len(packet_bytes) >= expected_data_len_bytes:
            return bytes(packet_bytes[:expected_data_len_bytes]), 0 # 0 ошибок исправлено
        else:
            logging.error(f"Packet length ({len(packet_bytes)}) < expected data length ({expected_data_len_bytes}) without ECC.")
            return None, -1 # Ошибка

    # Получаем параметры из объекта bch
    ecc_bits_len = bch.ecc_bits
    ecc_bytes_len = ceil(ecc_bits_len / 8.0)
    k_bits = bch.k
    n_bits = bch.n
    expected_packet_len_bytes = ceil(n_bits / 8.0) # Ожидаемая длина пакета в байтах

    # Проверяем длину пришедшего пакета
    current_packet_len_bytes = len(packet_bytes)
    if current_packet_len_bytes < expected_packet_len_bytes:
         logging.warning(f"decode_ecc: Packet length ({current_packet_len_bytes} bytes) < expected BCH block length ({expected_packet_len_bytes} bytes). Attempting decode.")
         # Декодер может справиться, но не гарантировано. Важно правильно разделить данные/ECC.

    # Определяем длину данных в пришедшем пакете (максимум k бит)
    # data_bytes_len_in_packet = current_packet_len_bytes - ecc_bytes_len # Неверно, если пакет короче n
    data_bits_len_in_packet = max(0, n_bits - ecc_bits_len) # = k_bits
    data_bytes_len_expected = ceil(data_bits_len_in_packet / 8.0)

    # Реальная длина данных в полученном (возможно, урезанном) пакете
    actual_data_bytes_len = max(0, current_packet_len_bytes - ecc_bytes_len)

    if actual_data_bytes_len < data_bytes_len_expected:
         logging.warning(f"  Data part seems truncated ({actual_data_bytes_len} bytes < {data_bytes_len_expected} expected).")

    data_to_decode = bytearray(packet_bytes[:actual_data_bytes_len])
    ecc_to_decode = packet_bytes[actual_data_bytes_len:]

    # Дополнение ECC нулями, если необходимо
    if len(ecc_to_decode) < ecc_bytes_len:
        padding_needed = ecc_bytes_len - len(ecc_to_decode)
        logging.warning(f"decode_ecc: ECC part length {len(ecc_to_decode)} < expected {ecc_bytes_len}. Padding with {padding_needed} zeros.")
        ecc_to_decode.extend([0] * padding_needed)
    elif len(ecc_to_decode) > ecc_bytes_len:
         logging.warning(f"decode_ecc: ECC part length {len(ecc_to_decode)} > expected {ecc_bytes_len}. Truncating.")
         ecc_to_decode = ecc_to_decode[:ecc_bytes_len]

     # Дополнение данных нулями, если они короче ОЖИДАЕМОЙ длины данных k (в байтах)
    if len(data_to_decode) < data_bytes_len_expected:
         padding_needed = data_bytes_len_expected - len(data_to_decode)
         logging.warning(f"decode_ecc: Data part length {len(data_to_decode)} < expected {data_bytes_len_expected}. Padding with {padding_needed} zeros.")
         data_to_decode.extend([0] * padding_needed)
    elif len(data_to_decode) > data_bytes_len_expected:
         logging.error(f"decode_ecc: Data part length {len(data_to_decode)} > expected {data_bytes_len_expected}. Truncating (Logic Error?).")
         data_to_decode = data_to_decode[:data_bytes_len_expected]


    try:
        # decode_inplace модифицирует data_to_decode
        data_copy = data_to_decode[:] # Копируем перед модификацией
        errors_corrected = bch.decode_inplace(data_copy, ecc_to_decode)

        if errors_corrected == -1:
            logging.error("ECC: Uncorrectable errors detected in packet.")
            return None, -1
        else:
            logging.info(f"ECC: Corrected {errors_corrected} bit errors in packet.")
            # Возвращаем ИСПРАВЛЕННЫЕ данные, обрезанные до ОЖИДАЕМОЙ длины ПОЛЕЗНОЙ нагрузки (payload)
            corrected_data = bytes(data_copy)
            if len(corrected_data) >= expected_data_len_bytes:
                return corrected_data[:expected_data_len_bytes], errors_corrected
            else:
                # Эта ситуация не должна возникать, если data_bytes_len_expected правильный
                logging.error(f"Internal error: Corrected data length ({len(corrected_data)}) < expected payload ({expected_data_len_bytes}).")
                return None, -1 # Ошибка

    except Exception as e:
        logging.error(f"Exception during ECC decoding: {e}", exc_info=True)
        return None, -1

# ============================================================
# --- Функции Работы с Видео (Только Чтение) ---
# ============================================================
def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # (Код без изменений)
    func_start_time = time.time(); logging.info(f"Reading video from: {video_path}")
    frames=[]; fps=float(FPS); cap=None; expected_height,expected_width=-1,-1
    try:
        cap=cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed to open video: {video_path}"); return frames,fps
        fps_read=cap.get(cv2.CAP_PROP_FPS);
        if fps_read>0: fps=float(fps_read); logging.info(f"Detected FPS: {fps:.2f}")
        else: logging.warning(f"Failed to get FPS. Using default: {fps}")
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_prop=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); logging.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, ~{frame_count_prop} frames")
        expected_height,expected_width=height,width; frame_index=0; read_count=0; none_frame_count=0; invalid_shape_count=0
        while True:
            ret,frame=cap.read(); frame_number_log=frame_index+1
            if not ret: logging.info(f"End of stream after reading {read_count} frames (index {frame_index})."); break
            if frame is None: logging.warning(f"Received None frame at index {frame_index}. Skipping."); none_frame_count+=1; frame_index+=1; continue
            if frame.ndim==3 and frame.shape[2]==3 and frame.dtype==np.uint8:
                 current_h,current_w=frame.shape[:2]
                 if current_h==expected_height and current_w==expected_width: frames.append(frame); read_count+=1;
                 else: logging.warning(f"Frame {frame_number_log} shape mismatch ({current_w}x{current_h} vs {expected_width}x{expected_height}). Skipping."); invalid_shape_count+=1
            else: logging.warning(f"Frame {frame_number_log} not valid BGR (ndim={frame.ndim}, dtype={frame.dtype}). Skipping."); invalid_shape_count+=1
            frame_index+=1
        logging.info(f"Finished reading. Valid frames: {len(frames)}. Skipped None: {none_frame_count}, Invalid shape: {invalid_shape_count}")
    except Exception as e: logging.error(f"Exception during video reading: {e}", exc_info=True)
    finally:
        if cap is not None and cap.isOpened(): cap.release(); logging.debug("Video capture released.")
    if not frames: logging.error(f"No valid frames were read from {video_path}")
    logging.debug(f"Read video time: {time.time() - func_start_time:.4f}s")
    return frames, fps

# ============================================================
# --- ЛОГИКА ИЗВЛЕЧЕНИЯ (Extract) ---
# ============================================================

# --- ИЗМЕНЕННАЯ extract_frame_pair (извлекает 1 бит из 1 кольца) ---
def extract_frame_pair(
        frame1: np.ndarray, frame2: np.ndarray, ring_index: int,
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    """Извлекает один бит из указанного кольца."""
    func_start_time = time.time(); pair_num_log = frame_number // 2
    logging.debug(f"--- Extract Start: Pair {pair_num_log} (F:{frame_number}), Ring:{ring_index} ---")
    try:
        if frame1 is None or frame2 is None: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Input frame None."); return None
        if frame1.shape != frame2.shape: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Frame shapes mismatch."); return None
        if frame1.dtype != np.uint8: frame1 = np.clip(frame1, 0, 255).astype(np.uint8); # logging.warning(f"[P:{pair_num_log}] F1 type corrected.")
        if frame2.dtype != np.uint8: frame2 = np.clip(frame2, 0, 255).astype(np.uint8); # logging.warning(f"[P:{pair_num_log}] F2 type corrected.")

        try: frame1_ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb); frame2_ycrcb = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)
        except cv2.error as e: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Color conversion failed: {e}"); return None

        comp_name = ['Y', 'Cr', 'Cb'][embed_component]; # logging.debug(f"[P:{pair_num_log}, R:{ring_index}] Using {comp_name}")
        try: comp1 = frame1_ycrcb[:, :, embed_component].astype(np.float32) / 255.0; comp2 = frame2_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
        except IndexError: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Invalid component index."); return None

        # DTCWT
        pyr1 = dtcwt_transform(comp1, frame_number=frame_number); pyr2 = dtcwt_transform(comp2, frame_number=frame_number + 1)
        if pyr1 is None or pyr2 is None or pyr1.lowpass is None or pyr2.lowpass is None:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] DTCWT failed."); return None
        L1 = pyr1.lowpass; L2 = pyr2.lowpass

        # Деление на кольца
        rings1_coords_np = ring_division(L1, n_rings=n_rings, frame_number=frame_number)
        rings2_coords_np = ring_division(L2, n_rings=n_rings, frame_number=frame_number + 1)

        # Проверка индекса кольца
        if not (0 <= ring_index < n_rings and ring_index < len(rings1_coords_np) and ring_index < len(rings2_coords_np)):
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] Invalid ring_index {ring_index}."); return None
        coords_1_np = rings1_coords_np[ring_index]; coords_2_np = rings2_coords_np[ring_index]
        if coords_1_np is None or coords_1_np.size == 0 or coords_2_np is None or coords_2_np.size == 0:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] Ring {ring_index} coordinates invalid/empty."); return None
        if coords_1_np.ndim != 2 or coords_1_np.shape[1] != 2 or coords_2_np.ndim != 2 or coords_2_np.shape[1] != 2:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] Invalid coords shape for ring."); return None

        # Извлечение значений и вычисление SVD
        try:
             rows1, cols1 = coords_1_np[:, 0], coords_1_np[:, 1]; ring_vals_1 = L1[rows1, cols1].astype(np.float32)
             rows2, cols2 = coords_2_np[:, 0], coords_2_np[:, 1]; ring_vals_2 = L2[rows2, cols2].astype(np.float32)
        except IndexError: logging.error(f"[P:{pair_num_log}, R:{ring_index}] IndexError extracting ring values.", exc_info=False); return None
        except Exception as e: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Error extracting ring values: {e}", exc_info=False); return None

        if ring_vals_1.size == 0 or ring_vals_2.size == 0:
            logging.error(f"[P:{pair_num_log}, R:{ring_index}] Extracted empty ring values."); return None

        # DCT -> SVD
        dct1 = dct_1d(ring_vals_1); dct2 = dct_1d(ring_vals_2)
        try: S1_vals = svd(dct1.reshape(-1, 1), compute_uv=False); S2_vals = svd(dct2.reshape(-1, 1), compute_uv=False)
        except np.linalg.LinAlgError as e: logging.error(f"[P:{pair_num_log}, R:{ring_index}] SVD failed: {e}."); return None

        s1 = S1_vals[0] if S1_vals.size > 0 else 0.0; s2 = S2_vals[0] if S2_vals.size > 0 else 0.0

        # Вычисление порога (используем адаптивную альфу, как в эмбеддере)
        alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_index, frame_number) # Альфа зависит от первого кадра
        eps = 1e-12
        # Порог может быть просто 1.0, или использовать альфа
        # Порог = 1.0 более чувствителен к шуму
        # Порог = (alpha + 1/alpha)/2 дает зазор вокруг 1.0
        threshold = (alpha + 1.0 / (alpha + eps)) / 2.0
        # Или используем фиксированный порог чуть больше 1? threshold = 1.001?
        # Остановимся на адаптивном пороге

        # logging.debug(f"[P:{pair_num_log}, R:{ring_index}] Alpha={alpha:.4f}, Threshold={threshold:.4f}")

        # Решение о бите
        ratio = s1 / (s2 + eps)
        bit_extracted = 0 if ratio >= threshold else 1

        logging.info(f"[P:{pair_num_log}, R:{ring_index}] s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f} vs thr={threshold:.4f} -> Bit={bit_extracted}")
        total_pair_time = time.time() - func_start_time; # logging.debug(f"--- Extract Finish: Pair {pair_num_log}, Ring {ring_index}. Time: {total_pair_time:.4f} sec ---")
        return bit_extracted

    except Exception as e:
        pair_num_log_err = frame_number // 2 if frame_number >= 0 else -1
        logging.error(f"!!! UNHANDLED EXCEPTION in extract_frame_pair (Pair {pair_num_log_err}, Ring {ring_index}): {e}", exc_info=True)
        return None

# --- НОВАЯ extract_frame_pair_multi_ring (извлекает N бит из N колец) ---
def extract_frame_pair_multi_ring(
        frame1: np.ndarray, frame2: np.ndarray, ring_indices: List[int],
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> List[Optional[int]]:
    """Извлекает по одному биту из каждого указанного кольца."""
    pair_num_log = frame_number // 2
    if not ring_indices:
        logging.error(f"[P:{pair_num_log}] No ring indices provided for multi-ring extraction"); return [None] * len(ring_indices) # Возвращаем None для каждого ожидаемого бита

    extracted_bits: List[Optional[int]] = []
    logging.debug(f"[P:{pair_num_log}] Multi-ring extraction for rings: {ring_indices}")

    # Параллельное извлечение из колец этой пары? Пока последовательно.
    for i, ring_idx in enumerate(ring_indices):
        bit = extract_frame_pair(frame1, frame2, ring_idx, n_rings, frame_number, embed_component)
        extracted_bits.append(bit)
        if bit is None:
             logging.warning(f"[P:{pair_num_log}] Failed extraction from ring {ring_idx} (bit position {i}).")

    # Не проводим голосование здесь, просто возвращаем извлеченные биты
    logging.info(f"[P:{pair_num_log}] Multi-ring extracted bits: {extracted_bits} from rings {ring_indices}")
    return extracted_bits

# --- ИЗМЕНЕННЫЙ _extract_frame_pair_worker (для N=2) ---
def _extract_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, List[Optional[int]]]:
    """Воркер для извлечения N бит из N колец."""
    pair_idx = args['pair_idx']; frame1 = args['frame1']; frame2 = args['frame2']; n_rings = args['n_rings']
    embed_component = args['embed_component']; effective_use_saved_rings = args['effective_use_saved_rings']
    ring_selection_method = args['ring_selection_method']; ring_selection_metric = args['ring_selection_metric']
    num_rings_to_use: int = args['num_rings_to_use'] # = BITS_PER_PAIR
    saved_rings_dict: Dict[int, List[int]] = args['saved_rings_dict'] # Теперь словарь
    frame_number = 2 * pair_idx
    extracted_bits: List[Optional[int]] = [None] * num_rings_to_use # Инициализируем список None

    try:
        target_ring_indices: List[int] = []
        # 1. Определяем кольца для использования
        if effective_use_saved_rings:
            if pair_idx in saved_rings_dict:
                target_ring_indices = saved_rings_dict[pair_idx]
                if len(target_ring_indices) != num_rings_to_use:
                     logging.error(f"[P:{pair_idx}] Saved rings count mismatch ({len(target_ring_indices)} vs {num_rings_to_use} expected). Using empty list.")
                     target_ring_indices = []
                else:
                     logging.debug(f"[P:{pair_idx}] Worker using saved rings: {target_ring_indices}")
            else:
                 logging.error(f"[P:{pair_idx}] Index not found in saved rings dictionary. Cannot extract.")
                 # Оставляем target_ring_indices пустым
        else:
            # Динамический выбор N колец (требует select_embedding_rings)
            logging.debug(f"[P:{pair_idx}] Worker dynamic selection ('{ring_selection_method}').")
            try:
                # Получаем компонент и lowpass для выбора колец
                comp1_sel = frame1[:, :, embed_component].astype(np.float32) / 255.0
                pyr1_sel = dtcwt_transform(comp1_sel, frame_number=frame_number)
                if pyr1_sel is None or pyr1_sel.lowpass is None: raise RuntimeError("DTCWT failed for dynamic selection")
                L1_sel = pyr1_sel.lowpass; rings_coords_sel_np = ring_division(L1_sel, n_rings=n_rings, frame_number=frame_number)

                if ring_selection_method == 'multi_ring':
                     target_ring_indices = select_embedding_rings(
                          L1_sel, rings_coords_sel_np, num_to_select=num_rings_to_use,
                          metric=ring_selection_metric, frame_number=frame_number
                     )
                # Добавить обработку других методов если нужно, например, детерминистический + дублирование
                # elif ring_selection_method == 'deterministic':
                #    idx = deterministic_ring_selection(...)
                #    target_ring_indices = [idx] * num_rings_to_use
                else:
                     logging.error(f"[P:{pair_idx}] Dynamic selection method '{ring_selection_method}' not fully supported for multi-bit in worker. Using empty list.")
                     target_ring_indices = []

                if len(target_ring_indices) < num_rings_to_use:
                    logging.warning(f"[P:{pair_idx}] Dynamic selection failed to find {num_rings_to_use} rings. Found {len(target_ring_indices)}.")
                    # Оставляем то, что нашли, экстрактор обработает None

            except Exception as sel_err:
                logging.error(f"[P:{pair_idx}] Error during dynamic ring selection: {sel_err}.", exc_info=True)
                target_ring_indices = [] # Ошибка выбора

        # 2. Извлечение бит из выбранных колец
        if target_ring_indices:
             extracted_bits = extract_frame_pair_multi_ring(
                 frame1, frame2, ring_indices=target_ring_indices,
                 n_rings=n_rings, frame_number=frame_number, embed_component=embed_component
             )
        else:
            # Если кольца не выбраны, возвращаем список None
            logging.error(f"[P:{pair_idx}] No target rings determined. Returning None list.")
            extracted_bits = [None] * num_rings_to_use

        # Убедимся, что возвращается список правильной длины
        if len(extracted_bits) != num_rings_to_use:
             logging.warning(f"[P:{pair_idx}] Worker returned list of unexpected length {len(extracted_bits)}. Padding/truncating to {num_rings_to_use}.")
             padded_bits = [None] * num_rings_to_use
             for i in range(min(len(extracted_bits), num_rings_to_use)):
                 padded_bits[i] = extracted_bits[i]
             extracted_bits = padded_bits

        return pair_idx, extracted_bits

    except Exception as e:
        logging.error(f"Exception in worker for pair {pair_idx}: {e}", exc_info=True)
        return pair_idx, [None] * num_rings_to_use # Возвращаем список None при ошибке

# --- ИЗМЕНЕННАЯ extract_watermark_from_video (с голосованием по пакетам) ---
def extract_watermark_from_video(
        frames: List[np.ndarray],
        n_rings: int = N_RINGS,
        ring_selection_method: str = RING_SELECTION_METHOD,
        ring_selection_metric: str = RING_SELECTION_METRIC,
        num_rings_to_use: int = NUM_RINGS_TO_USE, # = BITS_PER_PAIR
        bits_per_pair: int = BITS_PER_PAIR,
        default_ring_index: int = DEFAULT_RING_INDEX,
        embed_component: int = EMBED_COMPONENT,
        use_saved_rings: bool = USE_SAVED_RINGS,
        # coverage_percentage: float = COVERAGE_PERCENTAGE, # Больше не используется напрямую
        max_packet_repeats: int = MAX_PACKET_REPEATS, # Используется для определения кол-ва пар
        use_ecc: bool = USE_ECC,
        bch_m: int = BCH_M,
        bch_t: int = BCH_T,
        payload_len_bytes: int = PAYLOAD_LEN_BYTES
) -> Optional[bytes]:
    """Извлекает ВЗ с учетом multi-ring и проводит голосование по декодированным пакетам."""
    logging.info(f"Starting parallel extraction (Bits/Pair: {bits_per_pair}) with packet voting (Max Repeats: {max_packet_repeats}).")
    logging.info(f"Ring Selection: {'Saved' if use_saved_rings else 'Dynamic (' + ring_selection_method + ')'}, Component: {['Y', 'Cr', 'Cb'][embed_component]}, Expect ECC: {use_ecc}")
    start_time = time.time()
    num_frames = len(frames); total_pairs_available = num_frames // 2
    processed_pairs_count = 0; failed_pair_extractions = 0

    if total_pairs_available == 0:
        logging.error("Not enough frames for any pairs."); return None

    # --- Определение параметров ECC и пакета ---
    bch = None
    packet_len_bits = payload_len_bytes * 8
    effective_use_ecc = False
    calculated_ecc_bits = 0
    bch_k_bits = payload_len_bytes * 8 # Длина данных по умолчанию

    if use_ecc and BCHLIB_AVAILABLE:
        try:
            bch = bchlib.BCH(m=bch_m, t=bch_t)
            calculated_ecc_bits = bch.ecc_bits
            bch_k_bits = bch.k
            bch_n_bits = bch.n
            if calculated_ecc_bits <= 0: raise ValueError("BCH reported non-positive ecc_bits.")
            # Проверяем, нужно ли было паддинг данных в эмбеддере
            if payload_len_bytes * 8 > bch_k_bits:
                 logging.warning(f"Payload size ({payload_len_bytes*8} bits) > BCH capacity k={bch_k_bits} bits! ECC might not work as expected.")
                 # В этом случае, фактический пакет мог быть payload + ecc_bits
                 packet_len_bits = (payload_len_bytes * 8) + calculated_ecc_bits
            else:
                 # Используем полную длину блока n, т.к. данные дополнялись до k
                 packet_len_bits = bch_n_bits

            logging.info(f"BCH initialized (m={bch_m}, t={bch_t}). Payload={payload_len_bytes*8}, ECC={calculated_ecc_bits}, k={bch_k_bits}, n={bch_n_bits}.")
            logging.info(f"===> Ожидаемая длина пакета (n): {packet_len_bits} бит ({ceil(packet_len_bits / 8.0)} байт).")
            effective_use_ecc = True
        except Exception as e:
            logging.error(f"Failed to initialize BCH: {e}. Disabling ECC.", exc_info=False)
            packet_len_bits = payload_len_bytes * 8
            calculated_ecc_bits = 0
            bch = None # Убедимся, что bch=None
    # ... (остальные логи про ECC)
    if packet_len_bits <= 0: logging.error("Packet length is zero."); return None

    # --- Загрузка колец ---
    saved_rings_dict: Dict[int, List[int]] = {}; effective_use_saved_rings = use_saved_rings
    if use_saved_rings:
        saved_rings_dict = load_saved_rings()
        if not saved_rings_dict:
            logging.warning("USE_SAVED_RINGS=True, but failed to load. Switching to dynamic.")
            effective_use_saved_rings = False

    # --- Определение количества пар для извлечения ---
    # Извлекаем достаточно пар, чтобы покрыть MAX_PACKET_REPEATS
    pairs_to_extract = min(total_pairs_available, ceil(max_packet_repeats * packet_len_bits / bits_per_pair))
    logging.info(f"Attempting to extract from {pairs_to_extract} pairs (up to {total_pairs_available} available, covers up to {max_packet_repeats} repeats).")
    if pairs_to_extract == 0: logging.warning("No pairs to extract."); return None
    if effective_use_saved_rings and not all(i in saved_rings_dict for i in range(pairs_to_extract)):
         max_pair_from_rings = max(saved_rings_dict.keys()) if saved_rings_dict else -1
         original_pairs_to_extract = pairs_to_extract
         pairs_to_extract = min(pairs_to_extract, max_pair_from_rings + 1)
         logging.warning(f"Saved rings only available up to pair {max_pair_from_rings}. Reducing pairs to extract from {original_pairs_to_extract} to {pairs_to_extract}.")

    # --- Параллельное извлечение ---
    extracted_bits_per_pair: Dict[int, List[Optional[int]]] = {} # {pair_idx: [bit1, bit2, ...]}
    tasks_args = []
    skipped_pairs_indices = []

    for i in range(pairs_to_extract):
        idx1 = 2 * i; idx2 = idx1 + 1
        if idx2 >= num_frames or frames[idx1] is None or frames[idx2] is None:
            logging.error(f"Frames missing for pair {i}. Skipping."); skipped_pairs_indices.append(i); continue
        if effective_use_saved_rings and i not in saved_rings_dict:
             logging.error(f"Using saved rings, but no data for pair {i}. Skipping.")
             skipped_pairs_indices.append(i); continue

        args = {
            'pair_idx': i, 'frame1': frames[idx1], 'frame2': frames[idx2], 'n_rings': n_rings,
            'embed_component': embed_component, 'effective_use_saved_rings': effective_use_saved_rings,
            'ring_selection_method': ring_selection_method, 'ring_selection_metric': ring_selection_metric,
            'num_rings_to_use': num_rings_to_use, # = bits_per_pair
            'saved_rings_dict': saved_rings_dict, # Передаем словарь
            'default_ring_index': default_ring_index # Для fallback в select_rings
        }
        tasks_args.append(args)

    if not tasks_args: logging.error("No valid tasks for parallel extraction."); return None

    executor_class = concurrent.futures.ThreadPoolExecutor
    logging.info(f"Submitting {len(tasks_args)} extraction tasks to {executor_class.__name__} (max_workers={MAX_WORKERS_EXTRACT})...")
    try:
        with executor_class(max_workers=MAX_WORKERS_EXTRACT) as executor:
            future_to_pair_idx = {executor.submit(_extract_frame_pair_worker, arg): arg['pair_idx'] for arg in tasks_args}
            for future in concurrent.futures.as_completed(future_to_pair_idx):
                pair_idx = future_to_pair_idx[future]
                try:
                    _, bits_result_list = future.result() # Возвращает (pair_idx, List[Optional[int]])
                    extracted_bits_per_pair[pair_idx] = bits_result_list
                    # Считаем ошибки на уровне пар (если хотя бы один бит None)
                    if bits_result_list is None or None in bits_result_list:
                        failed_pair_extractions += 1
                        logging.warning(f"Pair {pair_idx} extraction resulted in None values: {bits_result_list}")
                    processed_pairs_count += 1
                    logging.debug(f"Pair {pair_idx} completed. Result: {bits_result_list} ({processed_pairs_count}/{len(tasks_args)})")
                except Exception as exc:
                    logging.error(f'Pair {pair_idx} generated exception: {exc}', exc_info=True)
                    extracted_bits_per_pair[pair_idx] = [None] * bits_per_pair # Помечаем ошибку
                    failed_pair_extractions += 1
                    # processed_pairs_count не увеличиваем? Или увеличиваем? Увеличим.
                    processed_pairs_count += 1
    except Exception as e:
         logging.critical(f"CRITICAL ERROR during Executor: {e}", exc_info=True)
         # Помечаем все неоконченные как ошибки
         for i in range(pairs_to_extract):
             if i not in skipped_pairs_indices and i not in extracted_bits_per_pair:
                 extracted_bits_per_pair[i] = [None] * bits_per_pair
                 failed_pair_extractions += 1

    logging.info(f"Parallel extraction finished. Pairs submitted: {len(tasks_args)}. Pairs processed: {processed_pairs_count}. Pairs with failed extraction (None bits): {failed_pair_extractions}.")

    # --- Сборка плоского списка бит ---
    extracted_bits_all: List[Optional[int]] = []
    total_extracted_bits_count = 0
    for i in range(pairs_to_extract):
        bits_list = extracted_bits_per_pair.get(i)
        if bits_list: # Если для пары есть результат (даже если там None)
            extracted_bits_all.extend(bits_list)
            total_extracted_bits_count += len(bits_list)
        else:
            # Если пары не было в результатах (пропущена или ошибка executor)
            extracted_bits_all.extend([None] * bits_per_pair)
            total_extracted_bits_count += bits_per_pair
    logging.info(f"Total bits collected: {total_extracted_bits_count} (Expected based on pairs processed: {pairs_to_extract * bits_per_pair})")

    # --- НОВАЯ ЛОГИКА: Декодирование Пакетов и Голосование ---
    num_potential_packets = total_extracted_bits_count // packet_len_bits
    if num_potential_packets == 0 and total_extracted_bits_count >= packet_len_bits:
        num_potential_packets = 1
    logging.info(f"Attempting to decode {num_potential_packets} potential packets (Packet Size: {packet_len_bits} bits)...")

    decoded_payloads: List[bytes] = [] # Список успешно декодированных payload'ов
    decode_success_count = 0
    decode_fail_count = 0
    decode_ecc_corrected_total = 0

    for i in range(num_potential_packets):
        start_idx = i * packet_len_bits
        end_idx = start_idx + packet_len_bits
        if end_idx > total_extracted_bits_count:
            logging.warning(f"Not enough bits for full packet #{i+1} (need {packet_len_bits}, have {total_extracted_bits_count - start_idx}). Skipping remaining.")
            break

        packet_bits_list = extracted_bits_all[start_idx:end_idx]

        if None in packet_bits_list:
            logging.warning(f"Packet #{i+1} contains None bits (extraction error in pair). Skipping decode.")
            decode_fail_count += 1
            continue

        # Конвертация в байты
        packet_bytes = bits_to_bytes(packet_bits_list)
        if packet_bytes is None:
            logging.error(f"Failed to convert bits to bytes for packet #{i+1}.")
            decode_fail_count += 1
            continue

        # Проверка длины байтового пакета
        expected_packet_len_bytes = ceil(packet_len_bits / 8.0)
        if len(packet_bytes) < expected_packet_len_bytes:
             logging.warning(f"Packet #{i+1} byte length ({len(packet_bytes)}) < expected ({expected_packet_len_bytes}). Trying decode.")

        # Попытка декодирования
        payload: Optional[bytes] = None
        errors_corrected: int = -1

        if effective_use_ecc and bch is not None:
            # Обрезаем до нужной длины перед декодированием
            packet_to_decode = packet_bytes[:expected_packet_len_bytes]
            logging.debug(f"Decoding packet #{i+1} ({len(packet_to_decode)} bytes) with ECC...")
            payload, errors_corrected = decode_ecc(packet_to_decode, bch, payload_len_bytes)
        else:
            # Без ECC
            logging.debug(f"Attempting direct payload extraction for packet #{i+1} (ECC disabled/failed).")
            if len(packet_bytes) >= payload_len_bytes:
                payload = bytes(packet_bytes[:payload_len_bytes])
                errors_corrected = 0 # Считаем, что ошибок 0, т.к. не исправляли
            else:
                 payload = None
                 errors_corrected = -1

        # Обработка результата декодирования
        if payload is not None:
            logging.info(f"Successfully decoded payload from packet #{i+1} (ECC corrected: {errors_corrected if errors_corrected >= 0 else 'N/A'}).")
            decoded_payloads.append(payload)
            decode_success_count += 1
            if errors_corrected > 0:
                decode_ecc_corrected_total += errors_corrected
        else:
            logging.warning(f"Failed to decode payload from packet #{i+1}.")
            decode_fail_count += 1

    logging.info(f"Packet decoding summary: Success={decode_success_count}, Failed={decode_fail_count}. Total ECC corrections: {decode_ecc_corrected_total}.")

    # --- Голосование по декодированным Payload ---
    if not decoded_payloads:
        logging.error("Failed to decode any valid payload from any packet.")
        return None

    payload_counts = Counter(decoded_payloads)
    logging.info("Payload voting results:")
    for pld, count in payload_counts.most_common():
        try:
            # Попытка отобразить как ID hex
            id_hex = pld.hex()
            logging.info(f"  Payload (Hex ID: {id_hex}): {count} votes")
        except Exception:
             logging.info(f"  Payload (Bytes: {pld}): {count} votes") # Fallback

    # Выбираем победителя
    most_common_payload, winner_count = payload_counts.most_common(1)[0]
    confidence = winner_count / decode_success_count if decode_success_count > 0 else 0.0
    logging.info(f"Most common payload selected with {winner_count}/{decode_success_count} votes (Confidence: {confidence:.1%}).")

    final_payload_bytes = most_common_payload
    # --- Конец новой логики ---

    end_time = time.time()
    logging.info(f"Extraction and processing finished. Total time: {end_time - start_time:.2f} sec.")
    return final_payload_bytes

# ============================================================
# --- ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ ---
# ============================================================
def main():
    # (Код main почти без изменений, только конвертация в hex для вывода ID)
    main_start_time = time.time()
    input_video_base = "watermarked_video_n2_try"; input_video = input_video_base + INPUT_EXTENSION # <<<< Имя входного файла
    original_id_hex: Optional[str] = None
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r") as f: original_id_hex = f.read().strip()
            if original_id_hex and len(original_id_hex) == PAYLOAD_LEN_BYTES * 2:
                 # Проверка, что это hex строка нужной длины
                 int(original_id_hex, 16) # Попытка конвертации для валидации
                 logging.info(f"Read original ID (Hex) from {ORIGINAL_WATERMARK_FILE}")
            else:
                 logging.error(f"Content of {ORIGINAL_WATERMARK_FILE} is not a valid hex ID of length {PAYLOAD_LEN_BYTES*2}.")
                 original_id_hex = None
        except ValueError:
             logging.error(f"Content of {ORIGINAL_WATERMARK_FILE} is not a valid hex string.")
             original_id_hex = None
        except Exception as e: logging.error(f"Error reading {ORIGINAL_WATERMARK_FILE}: {e}"); original_id_hex = None
    else: logging.warning(f"{ORIGINAL_WATERMARK_FILE} not found.")

    logging.info("--- Starting Extraction Main Process ---")
    if not os.path.exists(input_video): logging.critical(f"Input video not found: '{input_video}'."); print(f"\nERROR: Input video '{input_video}' not found."); return

    frames, input_fps = read_video(input_video)
    if not frames: logging.critical(f"Failed to read frames from {input_video}."); return
    logging.info(f"Read {len(frames)} frames for extraction (Reported FPS: {input_fps:.2f})")

    # Вызов основной функции извлечения
    extracted_payload_bytes: Optional[bytes] = extract_watermark_from_video(
        frames=frames, n_rings=N_RINGS, ring_selection_method=RING_SELECTION_METHOD,
        ring_selection_metric=RING_SELECTION_METRIC, num_rings_to_use=NUM_RINGS_TO_USE,
        bits_per_pair=BITS_PER_PAIR, default_ring_index=DEFAULT_RING_INDEX,
        embed_component=EMBED_COMPONENT, use_saved_rings=USE_SAVED_RINGS,
        # coverage_percentage=COVERAGE_PERCENTAGE, # Больше не нужен
        max_packet_repeats=MAX_PACKET_REPEATS,
        use_ecc=USE_ECC, bch_m=BCH_M, bch_t=BCH_T, payload_len_bytes=PAYLOAD_LEN_BYTES
    )

    # Вывод результатов
    print(f"\n--- Extraction Results ---")
    extracted_id_hex: Optional[str] = None
    if extracted_payload_bytes is not None:
        if len(extracted_payload_bytes) == PAYLOAD_LEN_BYTES:
            extracted_id_hex = extracted_payload_bytes.hex()
            print(f"  Successfully extracted payload bytes.")
            print(f"  Decoded ID (Hex): {extracted_id_hex}")
            logging.info(f"Successfully decoded extracted payload to Hex ID: {extracted_id_hex}")
        else:
            # Эта ветка не должна срабатывать, если decode_ecc работает правильно
            logging.error(f"Extracted payload length mismatch ({len(extracted_payload_bytes)} vs {PAYLOAD_LEN_BYTES} bytes).")
            print(f"  ERROR: Extracted payload length mismatch.")
    else:
        logging.error("Extraction process failed to return payload bytes.")
        print(f"  Extraction FAILED.")

    # Сравнение с оригиналом (если он был загружен)
    if original_id_hex:
        print(f"  Original ID (Hex): {original_id_hex}")
        if extracted_id_hex is not None and extracted_id_hex == original_id_hex:
            print("\n  >>> ID MATCH <<<")
            logging.info("ID verification successful.")
        else:
            print("\n  >>> !!! ID MISMATCH or EXTRACTION FAILED !!! <<<")
            logging.warning("ID verification failed or extraction unsuccessful.")
    else:
        print("\n  Original ID not available for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_script_time = time.time() - main_start_time
    logging.info(f"--- Total Extractor Script Time: {total_script_time:.2f} sec ---")
    print(f"\nExtraction finished. Check log: {LOG_FILENAME}")

# --- Запуск с Профилированием ---
if __name__ == "__main__":
    if USE_ECC and not BCHLIB_AVAILABLE:
        print("\nWARNING: USE_ECC is True, but bchlib library is not installed.")
        print("ECC decoding will be skipped if possible, or extraction might fail.")

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
        profile_file = "profile_stats_extract_n2.txt" # Изменено имя
        try:
            with open(profile_file, "w") as f:
                stats_file = pstats.Stats(profiler, stream=f)
                stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved to {profile_file}")
            print(f"Profiling stats saved to {profile_file}")
        except IOError as e:
             logging.error(f"Could not save profiling stats to {profile_file}: {e}")