# -*- coding: utf-8 -*-
# Файл: extractor.py (Версия с ECC, агрегацией, частичным покрытием и параллелизмом - ИСПРАВЛЕНО v3)

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
import uuid
from math import ceil # <--- Добавлен импорт
try:
    import bchlib
    BCHLIB_AVAILABLE = True
except ImportError:
    BCHLIB_AVAILABLE = False
    # Логирование ошибки произойдет при проверке
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
MAX_WORKERS_EXTRACT: Optional[int] = None

# --- Настройки Адаптивности (ВАЖНО: ДОЛЖНЫ СООТВЕТСТВОВАТЬ EMBEDDER!) ---
RING_SELECTION_METHOD: str = 'deterministic'
RING_SELECTION_METRIC: str = 'entropy'
EMBED_COMPONENT: int = 1 # 0=Y, 1=Cr, 2=Cb
NUM_RINGS_TO_USE: int = 3
USE_SAVED_RINGS: bool = True

# --- Настройки Извлечения и ECC (ВАЖНО: ДОЛЖНЫ СООТВЕТСТВОВАТЬ EMBEDDER!) ---
USE_ECC: bool = True # Ожидается ли ECC?
BCH_M: int = 8
BCH_T: int = 5
COVERAGE_PERCENTAGE: float = 50.0 # Ожидаемый процент покрытия
PAYLOAD_LEN_BYTES: int = 16 # Длина исходных данных (UUID v4 = 16 байт)

# --- Настройка Видео Входа ---
INPUT_EXTENSION: str = '.avi'

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME, filemode='w', level=logging.INFO,
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
# logging.getLogger().setLevel(logging.DEBUG)

effective_use_ecc = USE_ECC and BCHLIB_AVAILABLE
logging.info(f"--- Запуск Скрипта Извлечения (ECC Ожид.: {USE_ECC}, Доступно: {BCHLIB_AVAILABLE}, Покрытие: {COVERAGE_PERCENTAGE}%) ---")
logging.info(f"Ожидаемые настройки эмбеддера: Метод='{RING_SELECTION_METHOD}', Метрика='{RING_SELECTION_METRIC if RING_SELECTION_METHOD in ['adaptive', 'multi_ring'] else 'N/A'}', N_RINGS={N_RINGS}")
logging.info(f"Альфа (ожид.): MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, Исп. сохр. кольца: {USE_SAVED_RINGS}")
if effective_use_ecc:
    logging.info(f"ECC Параметры (ожид.): BCH m={BCH_M}, t={BCH_T}") # Исправлено логирование
elif USE_ECC and not BCHLIB_AVAILABLE:
    logging.warning("ECC ожидается (USE_ECC=True), но bchlib не доступна! Декодирование ECC будет невозможно.")
else:
     logging.info("ECC не ожидается (USE_ECC=False).")

# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Идентичные Embedder) ---
# ============================================================
# (Функции dct_1d, dtcwt_transform, _ring_division_internal,
# get_ring_coords_cached, ring_division, calculate_entropies,
# compute_adaptive_alpha_entropy, deterministic_ring_selection,
# keypoint_based_ring_selection, select_embedding_ring,
# load_saved_rings, bits_to_bytes, decode_ecc
# ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ, как в предыдущих ответах для extractor.py)

def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    return dct(signal_1d, type=2, norm='ortho')

def dtcwt_transform(y_plane: np.ndarray, frame_number: int = -1) -> Optional[Pyramid]:
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
             get_ring_coords_cached.cache_clear(); coords_list_np = _ring_division_internal(shape, n_rings)
        return [arr.copy() if arr is not None else None for arr in coords_list_np] # Возвращаем копии
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception in ring_division or cache lookup: {e}", exc_info=True); return [None] * n_rings

def calculate_entropies(ring_vals: np.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    eps=1e-12;
    if ring_vals.size==0: return 0.0, 0.0
    hist, _ =np.histogram(np.clip(ring_vals,0.0,1.0), bins=256, range=(0.0,1.0), density=False)
    total_count=ring_vals.size;
    if total_count == 0: return 0.0, 0.0
    probabilities=hist/total_count; probabilities=probabilities[probabilities>eps]
    if probabilities.size==0: return 0.0, 0.0
    visual_entropy=-np.sum(probabilities*np.log2(probabilities)); edge_entropy=-np.sum(probabilities*np.exp(1.0-probabilities)); return visual_entropy, edge_entropy

def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    if ring_vals.size == 0: logging.warning(f"[F:{frame_number}, R:{ring_index}] compute_adaptive_alpha empty ring_vals."); return ALPHA_MIN
    visual_entropy, edge_entropy = calculate_entropies(ring_vals, frame_number, ring_index); local_variance = np.var(ring_vals)
    texture_factor = 1.0 / (1.0 + np.clip(local_variance, 0, 1) * 10.0)
    eps = 1e-12
    if abs(visual_entropy) < eps: entropy_ratio = 0.0; logging.debug(f"[F:{frame_number}, R:{ring_index}] Visual entropy near zero.")
    else: entropy_ratio = edge_entropy / visual_entropy
    sigmoid_input = entropy_ratio; sigmoid_ratio = 1.0 / (1.0 + np.exp(-sigmoid_input)) * texture_factor
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid_ratio; final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)
    logging.info(f"[F:{frame_number}, R:{ring_index}] Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, Var={local_variance:.4f}, Factor={texture_factor:.4f}, Ratio={entropy_ratio:.4f} -> final_alpha={final_alpha:.4f}")
    return final_alpha

def deterministic_ring_selection(frame: np.ndarray, n_rings: int, frame_number: int = -1) -> int:
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

def select_embedding_ring(
        lowpass_subband: np.ndarray, rings_coords_np: List[Optional[np.ndarray]],
        metric: str = RING_SELECTION_METRIC, frame_number: int = -1
) -> int:
    func_start_time = time.time(); best_metric_value = -float('inf'); selected_index = DEFAULT_RING_INDEX; metric_values = []
    n_rings_available = len(rings_coords_np); known_metrics = ['entropy', 'energy', 'variance', 'mean_abs_dev']
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2: logging.error(f"[F:{frame_number}] Invalid lowpass_subband input!"); return DEFAULT_RING_INDEX
    metric_to_use = metric if metric in known_metrics else 'entropy'
    if metric not in known_metrics: logging.warning(f"[F:{frame_number}] Unknown metric '{metric}', defaulting to 'entropy'.")
    logging.debug(f"[F:{frame_number}] Selecting ring using metric: '{metric_to_use}'")
    for i, coords_np in enumerate(rings_coords_np):
        current_metric = -float('inf')
        if coords_np is None or coords_np.size == 0: metric_values.append(current_metric); continue
        try:
            if coords_np.ndim != 2 or coords_np.shape[1] != 2: logging.warning(f"[F:{frame_number}, R:{i}] Invalid coords shape."); metric_values.append(current_metric); continue
            rows, cols = coords_np[:, 0], coords_np[:, 1]; ring_vals = lowpass_subband[rows, cols].astype(np.float32)
            if ring_vals.size == 0: metric_values.append(current_metric); continue
            if metric_to_use == 'entropy': v_e, _ = calculate_entropies(ring_vals, frame_number, i); current_metric = v_e
            elif metric_to_use == 'energy': current_metric = np.sum(ring_vals ** 2)
            elif metric_to_use == 'variance': current_metric = np.var(ring_vals)
            elif metric_to_use == 'mean_abs_dev': mean_val = np.mean(ring_vals); current_metric = np.mean(np.abs(ring_vals - mean_val))
            metric_values.append(current_metric)
            if current_metric > best_metric_value: best_metric_value = current_metric; selected_index = i
        except IndexError: logging.error(f"[F:{frame_number}, R:{i}] IndexError calculating metric.", exc_info=False); metric_values.append(-float('inf'))
        except Exception as e: logging.error(f"[F:{frame_number}, R:{i}] Error calculating metric: {e}", exc_info=False); metric_values.append(-float('inf'))
    metric_log_str = ", ".join([f"{i}:{v:.4f}" if v > -float('inf') else f"{i}:Err/Empty" for i, v in enumerate(metric_values)])
    logging.debug(f"[F:{frame_number}] Ring metrics ('{metric_to_use}'): [{metric_log_str}]")
    logging.info(f"[F:{frame_number}] Adaptive ring selection result: Ring={selected_index} (Value: {best_metric_value:.4f})")
    if not (0 <= selected_index < n_rings_available and rings_coords_np[selected_index] is not None and rings_coords_np[selected_index].size > 0):
        logging.error(f"[F:{frame_number}] Selected ring {selected_index} invalid! Checking default {DEFAULT_RING_INDEX}.")
        if 0 <= DEFAULT_RING_INDEX < n_rings_available and rings_coords_np[DEFAULT_RING_INDEX] is not None and rings_coords_np[DEFAULT_RING_INDEX].size > 0:
            selected_index = DEFAULT_RING_INDEX; logging.warning(f"[F:{frame_number}] Using default ring {selected_index}.")
        else:
            logging.warning(f"[F:{frame_number}] Default ring {DEFAULT_RING_INDEX} invalid. Searching..."); found_non_empty = False
            for idx, coords_np_check in enumerate(rings_coords_np):
                if coords_np_check is not None and coords_np_check.size > 0: selected_index = idx; logging.warning(f"[F:{frame_number}] Using first non-empty ring {selected_index}."); found_non_empty = True; break
            if not found_non_empty: logging.critical(f"[F:{frame_number}] All rings are empty!"); return DEFAULT_RING_INDEX
    logging.debug(f"[F:{frame_number}] Ring selection process time: {time.time() - func_start_time:.4f}s")
    return selected_index

def load_saved_rings() -> List[int]:
    if not os.path.exists(SELECTED_RINGS_FILE): logging.warning(f"Saved rings file '{SELECTED_RINGS_FILE}' not found."); return []
    try:
        with open(SELECTED_RINGS_FILE, 'r') as f: rings = json.load(f)
        if isinstance(rings, list) and all(isinstance(r, int) for r in rings): logging.info(f"Loaded {len(rings)} saved rings from {SELECTED_RINGS_FILE}"); return rings
        else: logging.error(f"Invalid format in {SELECTED_RINGS_FILE}."); return []
    except Exception as e: logging.error(f"Error loading/parsing saved rings: {e}", exc_info=True); return []

def bits_to_bytes(bits: List[Optional[int]]) -> Optional[bytearray]:
    valid_bits = [b for b in bits if b is not None]
    if not valid_bits: logging.warning("No valid bits to convert to bytes."); return bytearray()
    remainder = len(valid_bits) % 8
    if remainder != 0:
        logging.warning(f"Number of valid bits ({len(valid_bits)}) not multiple of 8. Padding with {8-remainder} zeros at the end.")
        valid_bits.extend([0] * (8 - remainder))
    byte_array = bytearray()
    for i in range(0, len(valid_bits), 8):
        byte_bits = valid_bits[i:i+8]
        try: byte_val = int("".join(map(str, byte_bits)), 2); byte_array.append(byte_val)
        except ValueError: logging.error(f"Error converting bits to byte: {byte_bits}"); return None
    return byte_array

def decode_ecc(packet_bytes: bytearray, bch: 'bchlib.BCH', expected_data_len_bytes: int) -> Optional[bytes]:
    if not BCHLIB_AVAILABLE or bch is None:
        logging.warning("bchlib not available or not initialized, skipping ECC decoding.")
        if len(packet_bytes) >= expected_data_len_bytes: return bytes(packet_bytes[:expected_data_len_bytes])
        else: logging.error(f"Packet length ({len(packet_bytes)}) < expected data length ({expected_data_len_bytes}) without ECC."); return None

    # Рассчитываем длину ECC в байтах правильно
    ecc_bytes_len = ceil(bch.ecc_bits / 8.0)
    expected_packet_len_bytes = ceil(bch.n / 8.0)

    if len(packet_bytes) < expected_packet_len_bytes:
        logging.warning(f"Packet length ({len(packet_bytes)} bytes) is less than expected BCH block length ({expected_packet_len_bytes} bytes). Trying to decode anyway.")
        # Пытаемся дополнить нулями до нужной длины? Или обрезать? Пока попробуем как есть.
        # bchlib может справиться с неполным пакетом, но это не гарантировано.

    # Определяем длину данных в пакете (округленную до байт)
    data_bytes_len = len(packet_bytes) - ecc_bytes_len
    if data_bytes_len < 0 :
         logging.error(f"Calculated data length is negative ({data_bytes_len}). Packet too short.")
         return None

    data_to_decode = bytearray(packet_bytes[:data_bytes_len])
    ecc_to_decode = packet_bytes[data_bytes_len:] # Берем остаток как ECC

    # Убедимся, что ecc_to_decode имеет правильную длину (дополним, если нужно)
    if len(ecc_to_decode) < ecc_bytes_len:
        logging.warning(f"ECC part length {len(ecc_to_decode)} is less than expected {ecc_bytes_len}. Padding with zeros.")
        ecc_to_decode.extend([0] * (ecc_bytes_len - len(ecc_to_decode)))
    elif len(ecc_to_decode) > ecc_bytes_len:
         logging.warning(f"ECC part length {len(ecc_to_decode)} is more than expected {ecc_bytes_len}. Truncating.")
         ecc_to_decode = ecc_to_decode[:ecc_bytes_len]

    try:
        errors_corrected = bch.decode_inplace(data_to_decode, ecc_to_decode)
        if errors_corrected == -1:
            logging.error("ECC: Too many errors detected to correct.")
            return None
        else:
            logging.info(f"ECC: Corrected {errors_corrected} bit errors.")
            corrected_data = bytes(data_to_decode)
            if len(corrected_data) >= expected_data_len_bytes:
                return corrected_data[:expected_data_len_bytes]
            else:
                logging.error(f"Corrected data length ({len(corrected_data)}) < expected payload ({expected_data_len_bytes}).")
                return None
    except Exception as e:
        logging.error(f"Error during ECC decoding: {e}", exc_info=True)
        return None

# ============================================================
# --- Функции Работы с Видео (Только Чтение) ---
# ============================================================
def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # (Код read_video без изменений)
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
                 else: logging.warning(f"Frame {frame_number_log} shape mismatch. Skipping."); invalid_shape_count+=1
            else: logging.warning(f"Frame {frame_number_log} not valid BGR. Skipping."); invalid_shape_count+=1
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

# --- АДАПТИРОВАННАЯ extract_frame_pair (Идентична embedder.py) ---
def extract_frame_pair(
        frame1: np.ndarray, frame2: np.ndarray, ring_index: int,
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    # (Код extract_frame_pair - идентичен embedder.py)
    func_start_time = time.time(); pair_num_log = frame_number // 2
    logging.debug(f"--- Extract Start: Pair {pair_num_log} (F:{frame_number}), Ring:{ring_index} ---")
    try:
        if frame1 is None or frame2 is None: logging.error(f"[P:{pair_num_log}] Input frame None."); return None
        if frame1.shape != frame2.shape: logging.error(f"[P:{pair_num_log}] Frame shapes mismatch."); return None
        if frame1.dtype != np.uint8: frame1 = np.clip(frame1, 0, 255).astype(np.uint8); logging.warning(f"[P:{pair_num_log}] F1 type corrected.")
        if frame2.dtype != np.uint8: frame2 = np.clip(frame2, 0, 255).astype(np.uint8); logging.warning(f"[P:{pair_num_log}] F2 type corrected.")
        try: frame1_ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb); frame2_ycrcb = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)
        except cv2.error as e: logging.error(f"[P:{pair_num_log}] Color conversion failed: {e}"); return None
        comp_name = ['Y', 'Cr', 'Cb'][embed_component]; logging.debug(f"[P:{pair_num_log}] Using {comp_name}")
        try: comp1 = frame1_ycrcb[:, :, embed_component].astype(np.float32) / 255.0; comp2 = frame2_ycrcb[:, :, embed_component].astype(np.float32) / 255.0
        except IndexError: logging.error(f"[P:{pair_num_log}] Invalid component index."); return None
        pyr1 = dtcwt_transform(comp1, frame_number=frame_number); pyr2 = dtcwt_transform(comp2, frame_number=frame_number + 1)
        if pyr1 is None or pyr2 is None or pyr1.lowpass is None or pyr2.lowpass is None: logging.error(f"[P:{pair_num_log}] DTCWT failed."); return None
        L1 = pyr1.lowpass; L2 = pyr2.lowpass
        rings1_coords_np = ring_division(L1, n_rings=n_rings, frame_number=frame_number)
        rings2_coords_np = ring_division(L2, n_rings=n_rings, frame_number=frame_number + 1)
        if not (0 <= ring_index < n_rings and ring_index < len(rings1_coords_np) and ring_index < len(rings2_coords_np)): logging.error(f"[P:{pair_num_log}] Invalid ring_index {ring_index}."); return None
        coords_1_np = rings1_coords_np[ring_index]; coords_2_np = rings2_coords_np[ring_index]
        if coords_1_np is None or coords_1_np.size == 0 or coords_2_np is None or coords_2_np.size == 0: logging.error(f"[P:{pair_num_log}] Ring {ring_index} empty/invalid."); return None
        if coords_1_np.ndim != 2 or coords_1_np.shape[1] != 2 or coords_2_np.ndim != 2 or coords_2_np.shape[1] != 2: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Invalid coords shape."); return None
        try:
             rows1, cols1 = coords_1_np[:, 0], coords_1_np[:, 1]; ring_vals_1 = L1[rows1, cols1].astype(np.float32)
             rows2, cols2 = coords_2_np[:, 0], coords_2_np[:, 1]; ring_vals_2 = L2[rows2, cols2].astype(np.float32)
        except IndexError: logging.error(f"[P:{pair_num_log}, R:{ring_index}] IndexError extracting vals.", exc_info=False); return None
        except Exception as e: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Error extracting vals: {e}", exc_info=False); return None
        if ring_vals_1.size == 0 or ring_vals_2.size == 0: logging.error(f"[P:{pair_num_log}, R:{ring_index}] Extracted empty vals."); return None
        alpha = compute_adaptive_alpha_entropy(ring_vals_1, ring_index, frame_number)
        eps = 1e-12; threshold = (alpha + 1.0 / (alpha + eps)) / 2.0
        logging.debug(f"[P:{pair_num_log}, R:{ring_index}] Alpha={alpha:.4f}, Threshold={threshold:.4f}")
        dct1 = dct_1d(ring_vals_1); dct2 = dct_1d(ring_vals_2)
        try: S1_vals = svd(dct1.reshape(-1, 1), compute_uv=False); S2_vals = svd(dct2.reshape(-1, 1), compute_uv=False)
        except np.linalg.LinAlgError as e: logging.error(f"[P:{pair_num_log}, R:{ring_index}] SVD failed: {e}."); return None
        s1 = S1_vals[0] if S1_vals.size > 0 else 0.0; s2 = S2_vals[0] if S2_vals.size > 0 else 0.0
        ratio = s1 / (s2 + eps)
        bit_extracted = 0 if ratio >= threshold else 1
        logging.info(f"[P:{pair_num_log}, R:{ring_index}] s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f} vs thr={threshold:.4f} -> Bit={bit_extracted}")
        total_pair_time = time.time() - func_start_time; logging.debug(f"--- Extract Finish: Pair {pair_num_log}. Time: {total_pair_time:.4f} sec ---")
        return bit_extracted
    except Exception as e:
        pair_num_log_err = frame_number // 2 if frame_number >= 0 else -1
        logging.error(f"!!! UNHANDLED EXCEPTION in extract_frame_pair (Pair {pair_num_log_err}, Ring {ring_index}): {e}", exc_info=True)
        return None

# --- Функция для multi-ring extraction (Идентична embedder.py) ---
def extract_frame_pair_multi_ring(
        frame1: np.ndarray, frame2: np.ndarray, ring_indices: List[int],
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT
) -> Optional[int]:
    # (Код extract_frame_pair_multi_ring - идентичен embedder.py)
    pair_num_log = frame_number // 2
    if not ring_indices: logging.error(f"[P:{pair_num_log}] No ring indices for multi-ring"); return None
    logging.info(f"[P:{pair_num_log}] Multi-ring extraction for rings: {ring_indices}")
    bits = []
    for ring_idx in ring_indices:
        bit = extract_frame_pair(frame1, frame2, ring_idx, n_rings, frame_number, embed_component)
        if bit is not None: bits.append(bit); logging.debug(f"[P:{pair_num_log}] Multi-ring: bit={bit} from ring {ring_idx}")
        else: logging.warning(f"[P:{pair_num_log}] Multi-ring: Failed extraction from ring {ring_idx}.")
    if not bits: logging.error(f"[P:{pair_num_log}] Multi-ring: Failed to extract any bits"); return None
    zeros = bits.count(0); ones = bits.count(1); final_bit = 0 if zeros >= ones else 1
    logging.info(f"[P:{pair_num_log}] Multi-ring result: votes 0={zeros}, 1={ones} -> final_bit={final_bit}")
    return final_bit

# --- Воркер для параллельного извлечения (Идентичен embedder.py) ---
def _extract_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, Optional[int]]:
    # (Код _extract_frame_pair_worker - идентичен embedder.py)
    pair_idx = args['pair_idx']; frame1 = args['frame1']; frame2 = args['frame2']; n_rings = args['n_rings']
    embed_component = args['embed_component']; effective_use_saved_rings = args['effective_use_saved_rings']
    ring_selection_method = args['ring_selection_method']; ring_selection_metric = args['ring_selection_metric']
    default_ring_index = args['default_ring_index']; num_rings_to_use = args['num_rings_to_use']
    saved_rings = args['saved_rings']; frame_number = 2 * pair_idx
    bit_extracted = None
    try:
        target_ring_index = -1; target_ring_indices = []
        if effective_use_saved_rings:
            if pair_idx < len(saved_rings):
                saved_ring = saved_rings[pair_idx]
                if 0 <= saved_ring < n_rings: target_ring_index = saved_ring; target_ring_indices = [target_ring_index]; logging.info(f"[P:{pair_idx}] Worker using saved ring: {target_ring_index}")
                else: logging.error(f"[P:{pair_idx}] Invalid saved ring {saved_ring}."); return pair_idx, None
            else: logging.error(f"[P:{pair_idx}] Index out of bounds for saved rings."); return pair_idx, None
        else:
            logging.debug(f"[P:{pair_idx}] Worker dynamic selection ('{ring_selection_method}').")
            try:
                if embed_component == 0: comp1_sel = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
                elif embed_component == 1: comp1_sel = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 1].astype(np.float32) / 255.0
                else: comp1_sel = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 2].astype(np.float32) / 255.0
                pyr1_sel = dtcwt_transform(comp1_sel, frame_number=frame_number)
                if pyr1_sel is None or pyr1_sel.lowpass is None: raise RuntimeError("DTCWT failed for dynamic selection")
                L1_sel = pyr1_sel.lowpass; rings_coords_sel_np = ring_division(L1_sel, n_rings=n_rings, frame_number=frame_number)
                if ring_selection_method == 'deterministic': target_ring_index = deterministic_ring_selection(frame1, n_rings, frame_number=frame_number); target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'keypoint': target_ring_index = keypoint_based_ring_selection(frame1, n_rings, frame_number=frame_number); target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'multi_ring':
                     metric_values = []; temp_metric = ring_selection_metric
                     for ring_i, coords_np in enumerate(rings_coords_sel_np):
                         if coords_np is not None and coords_np.size > 0:
                             try:
                                 rows, cols = coords_np[:, 0], coords_np[:, 1]; ring_vals = L1_sel[rows, cols].astype(np.float32)
                                 if ring_vals.size > 0:
                                     if temp_metric == 'entropy': v_e, _ = calculate_entropies(ring_vals, frame_number, ring_i); metric_values.append((v_e, ring_i))
                                     elif temp_metric == 'energy': metric_values.append((np.sum(ring_vals**2), ring_i))
                                     elif temp_metric == 'variance': metric_values.append((np.var(ring_vals), ring_i))
                                     elif temp_metric == 'mean_abs_dev': mean_val = np.mean(ring_vals); metric_values.append((np.mean(np.abs(ring_vals - mean_val)), ring_i))
                                     else: v_e, _ = calculate_entropies(ring_vals, frame_number, ring_i); metric_values.append((v_e, ring_i))
                                 else: metric_values.append((-float('inf'), ring_i))
                             except IndexError: metric_values.append((-float('inf'), ring_i))
                         else: metric_values.append((-float('inf'), ring_i))
                     metric_values.sort(key=lambda x: x[0], reverse=True); target_ring_indices = [idx for val, idx in metric_values[:num_rings_to_use] if val > -float('inf')]
                     if not target_ring_indices: target_ring_indices = [default_ring_index]
                     logging.info(f"[P:{pair_idx}] Worker multi-ring dynamic (metric '{temp_metric}'): {target_ring_indices}")
                elif ring_selection_method == 'adaptive': target_ring_index = select_embedding_ring(L1_sel, rings_coords_sel_np, metric=ring_selection_metric, frame_number=frame_number); target_ring_indices = [target_ring_index]
                elif ring_selection_method == 'fixed': target_ring_index = default_ring_index; target_ring_indices = [target_ring_index]
                else: logging.error(f"[P:{pair_idx}] Unknown dynamic method. Using default."); target_ring_index = default_ring_index; target_ring_indices = [target_ring_index]
            except Exception as sel_err: logging.error(f"[P:{pair_idx}] Error during dynamic selection: {sel_err}. Using default.", exc_info=True); target_ring_index = default_ring_index; target_ring_indices = [target_ring_index]
        if ring_selection_method == 'multi_ring' and not effective_use_saved_rings: bit_extracted = extract_frame_pair_multi_ring(frame1, frame2, ring_indices=target_ring_indices, n_rings=n_rings, frame_number=frame_number, embed_component=embed_component)
        elif target_ring_index != -1: bit_extracted = extract_frame_pair(frame1, frame2, ring_index=target_ring_index, n_rings=n_rings, frame_number=frame_number, embed_component=embed_component)
        elif target_ring_indices:
             if len(target_ring_indices) == 1: target_ring_index = target_ring_indices[0]; bit_extracted = extract_frame_pair(frame1, frame2, ring_index=target_ring_index, n_rings=n_rings, frame_number=frame_number, embed_component=embed_component)
             else: logging.debug(f"[P:{pair_idx}] Calling multi-ring extraction for indices: {target_ring_indices}"); bit_extracted = extract_frame_pair_multi_ring(frame1, frame2, ring_indices=target_ring_indices, n_rings=n_rings, frame_number=frame_number, embed_component=embed_component)
        else: logging.error(f"[P:{pair_idx}] Worker: No valid target ring determined."); bit_extracted = None
        return pair_idx, bit_extracted
    except Exception as e: logging.error(f"Exception in worker for pair {pair_idx}: {e}", exc_info=True); return pair_idx, None

# --- ПАРАЛЛЕЛИЗОВАННАЯ extract_watermark_from_video (с исправленной логикой агрегации и ECC) ---
def extract_watermark_from_video(
        frames: List[np.ndarray],
        n_rings: int = N_RINGS,
        ring_selection_method: str = RING_SELECTION_METHOD,
        ring_selection_metric: str = RING_SELECTION_METRIC,
        default_ring_index: int = DEFAULT_RING_INDEX,
        embed_component: int = EMBED_COMPONENT,
        num_rings_to_use: int = NUM_RINGS_TO_USE,
        use_saved_rings: bool = USE_SAVED_RINGS,
        coverage_percentage: float = COVERAGE_PERCENTAGE,
        use_ecc: bool = USE_ECC,
        bch_m: int = BCH_M,
        bch_t: int = BCH_T,
        payload_len_bytes: int = PAYLOAD_LEN_BYTES
) -> Optional[bytes]:
    logging.info(f"Starting parallel extraction with aggregation & ECC (expected coverage: {coverage_percentage}%).")
    logging.info(f"Ring Selection (dynamic): '{ring_selection_method}', Metric: '{ring_selection_metric}'")
    logging.info(f"Component: {['Y', 'Cr', 'Cb'][embed_component]}, Use saved rings: {use_saved_rings}, Expect ECC: {use_ecc}")
    start_time = time.time()
    num_frames = len(frames); total_pairs = num_frames // 2
    processed_pairs = 0; error_pairs = 0

    # Инициализация BCH и определение длины пакета
    bch = None; packet_len_bits = payload_len_bytes * 8; effective_use_ecc = False
    if use_ecc and BCHLIB_AVAILABLE:
        try:
            bch = bchlib.BCH(m=bch_m, t=bch_t)
            # Рассчитываем ОЖИДАЕМУЮ длину пакета в битах (n из объекта BCH)
            # Убедимся, что используем bch.n, если он доступен, иначе рассчитываем теоретически
            if hasattr(bch, 'n'):
                 packet_len_bits = bch.n
                 logging.info(f"BCH initialized: n={bch.n}, t={bch.t}. Expected packet len: {packet_len_bits} bits.")
            else: # Старая версия bchlib? Рассчитаем теоретически (может быть неточно)
                 packet_len_bits = (1 << bch_m) - 1
                 logging.warning(f"BCH attribute 'n' not found. Assuming block size n={packet_len_bits} bits based on m={bch_m}.")
                 logging.info(f"BCH initialized (m={bch_m}, t={bch_t}). Assumed packet len: {packet_len_bits} bits.")

            # Проверка вместимости полезной нагрузки (если атрибут k доступен)
            if hasattr(bch, 'k') and payload_len_bytes * 8 > bch.k:
                logging.warning(f"Expected payload ({payload_len_bytes*8} bits) > BCH data capacity ({bch.k} bits)!")
            elif hasattr(bch, 'ecc_bits') and payload_len_bytes * 8 > packet_len_bits - bch.ecc_bits:
                 logging.warning(f"Expected payload ({payload_len_bytes*8} bits) > calculated data capacity ({packet_len_bits - bch.ecc_bits} bits)!")

            effective_use_ecc = True # ECC будет использоваться
        except Exception as e:
            logging.error(f"Failed to initialize BCH(m={bch_m}, t={bch_t}): {e}. Disabling ECC for this run.", exc_info=False)
            packet_len_bits = payload_len_bytes * 8 # Сбрасываем на длину payload
    elif use_ecc and not BCHLIB_AVAILABLE:
        logging.warning("ECC expected but bchlib not available. ECC will not be used.")
        packet_len_bits = payload_len_bytes * 8
    else:
        logging.info("ECC is not expected to be used.")
        packet_len_bits = payload_len_bytes * 8

    if packet_len_bits <= 0: logging.error("Packet length is zero or negative."); return None

    # Загрузка сохраненных колец
    saved_rings: List[int] = []; effective_use_saved_rings = use_saved_rings
    if use_saved_rings:
        saved_rings = load_saved_rings()
        if not saved_rings: logging.warning("USE_SAVED_RINGS=True, but failed to load. Switching to dynamic."); effective_use_saved_rings = False

    # Расчет пар для извлечения
    target_coverage = max(0.0, min(100.0, coverage_percentage))
    num_pairs_to_extract = int(total_pairs * (target_coverage / 100.0))
    if num_pairs_to_extract == 0 and target_coverage > 0: num_pairs_to_extract = 1
    if num_pairs_to_extract > total_pairs: num_pairs_to_extract = total_pairs
    logging.info(f"Attempting to extract from {num_pairs_to_extract} pairs (up to {total_pairs} available).")
    if num_pairs_to_extract == 0: logging.warning("No pairs to extract."); return None
    if effective_use_saved_rings and len(saved_rings) < num_pairs_to_extract:
        logging.warning(f"Saved rings count ({len(saved_rings)}) < pairs to extract ({num_pairs_to_extract}).")

    # Параллельное извлечение
    extracted_bits_all: List[Optional[int]] = [None] * num_pairs_to_extract
    tasks_args = []; skipped_pairs_indices = []
    for i in range(num_pairs_to_extract):
        idx1 = 2 * i; idx2 = idx1 + 1
        if idx2 >= num_frames or frames[idx1] is None or frames[idx2] is None:
            logging.error(f"Frames missing for pair {i}. Skipping."); skipped_pairs_indices.append(i); error_pairs += 1; continue
        args = {
            'pair_idx': i, 'frame1': frames[idx1], 'frame2': frames[idx2], 'n_rings': n_rings,
            'embed_component': embed_component, 'effective_use_saved_rings': effective_use_saved_rings,
            'ring_selection_method': ring_selection_method, 'ring_selection_metric': ring_selection_metric,
            'default_ring_index': default_ring_index, 'num_rings_to_use': num_rings_to_use,
            'saved_rings': saved_rings
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
                    _, bit_result = future.result()
                    if 0 <= pair_idx < len(extracted_bits_all):
                        extracted_bits_all[pair_idx] = bit_result
                        if bit_result is None: error_pairs += 1
                        processed_pairs += 1
                        logging.debug(f"Pair {pair_idx} completed. Result: {bit_result} ({processed_pairs}/{len(tasks_args)})")
                    else: logging.error(f"Result for out-of-bounds pair index {pair_idx}."); error_pairs += 1
                except Exception as exc:
                    logging.error(f'Pair {pair_idx} generated exception: {exc}', exc_info=True)
                    if 0 <= pair_idx < len(extracted_bits_all): extracted_bits_all[pair_idx] = None
                    error_pairs += 1; processed_pairs += 1
    except Exception as e:
         logging.critical(f"CRITICAL ERROR during Executor: {e}", exc_info=True)
         for i in range(num_pairs_to_extract):
             if i not in skipped_pairs_indices and extracted_bits_all[i] is None: extracted_bits_all[i] = None

    total_processed = processed_pairs + len(skipped_pairs_indices)
    logging.info(f"Parallel extraction finished. Pairs submitted: {len(tasks_args)}. Processed: {total_processed}. Errors/Nones: {error_pairs}.")

    # --- Агрегация (Исправленная логика) ---
    # Агрегируем только те биты, которые могли быть извлечены
    num_bits_to_reconstruct = min(packet_len_bits, num_pairs_to_extract)
    logging.info(f"Starting aggregation for {num_bits_to_reconstruct} bits...")
    reconstructed_packet_bits: List[Optional[int]] = [None] * num_bits_to_reconstruct
    votes_per_bit = [0] * num_bits_to_reconstruct

    for bit_idx in range(num_bits_to_reconstruct):
        # Собираем голоса только из доступных пар
        votes = [extracted_bits_all[pair_idx]
                 for pair_idx in range(bit_idx, num_pairs_to_extract, packet_len_bits)
                 if extracted_bits_all[pair_idx] is not None]
        votes_per_bit[bit_idx] = len(votes)
        if not votes:
            reconstructed_packet_bits[bit_idx] = None # Или random.choice([0,1])?
        else:
            zeros = votes.count(0); ones = votes.count(1)
            reconstructed_packet_bits[bit_idx] = 0 if zeros >= ones else 1

    valid_reconstructed_bits = sum(1 for b in reconstructed_packet_bits if b is not None)
    logging.info(f"Aggregation finished. Reconstructed bits: {valid_reconstructed_bits}/{num_bits_to_reconstruct}.")
    logging.debug(f"Votes per bit position (first {num_bits_to_reconstruct}): {votes_per_bit}")

    if valid_reconstructed_bits < num_bits_to_reconstruct:
        logging.error("Aggregation resulted in an incomplete packet (contains None).")
        return None
    # --- Конец Агрегации ---

    # --- ECC Декодирование и Извлечение Payload ---
    logging.info("Converting aggregated bits to bytes...")
    reconstructed_packet_bytes = bits_to_bytes(reconstructed_packet_bits) # Используем только реконструированные биты

    if reconstructed_packet_bytes is None:
        logging.error("Failed to convert aggregated bits to bytes.")
        return None

    # Определяем, нужно ли пытаться декодировать ECC
    can_try_ecc = effective_use_ecc and len(reconstructed_packet_bytes) * 8 >= packet_len_bits

    logging.info(f"Attempting to decode packet ({len(reconstructed_packet_bytes)} bytes). Will use ECC: {can_try_ecc}")
    final_payload_bytes: Optional[bytes] = None

    if can_try_ecc:
        # Обрезаем до ожидаемой длины пакета BCH перед декодированием
        packet_to_decode = reconstructed_packet_bytes[:ceil(packet_len_bits / 8.0)]
        final_payload_bytes = decode_ecc(packet_to_decode, bch, payload_len_bytes)
        if final_payload_bytes is None:
            logging.error("ECC decoding failed.")
            # Можно добавить fallback - попытку взять начало без ECC
            if len(reconstructed_packet_bytes) >= payload_len_bytes:
                 logging.warning("Attempting to extract payload without ECC correction as fallback.")
                 final_payload_bytes = bytes(reconstructed_packet_bytes[:payload_len_bytes])
            else:
                 return None # Ошибка окончательная
        else:
             logging.info(f"ECC decoding successful. Payload length: {len(final_payload_bytes)} bytes.")
    else:
        # ECC не используется или пакет слишком короткий
        logging.info("ECC was disabled or packet too short. Taking payload directly.")
        if len(reconstructed_packet_bytes) >= payload_len_bytes:
            final_payload_bytes = bytes(reconstructed_packet_bytes[:payload_len_bytes])
            logging.info(f"Extracted payload length: {len(final_payload_bytes)} bytes.")
        else:
             logging.error(f"Aggregated packet length ({len(reconstructed_packet_bytes)}) < expected payload ({payload_len_bytes}).")
             return None

    end_time = time.time()
    logging.info(f"Extraction and processing finished. Total time: {end_time - start_time:.2f} sec.")
    return final_payload_bytes

# ============================================================
# --- ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ ---
# ============================================================
def main():
    # (Код main без изменений)
    main_start_time = time.time()
    input_video_base = "watermarked_video"; input_video = input_video_base + INPUT_EXTENSION
    original_uuid_str: Optional[str] = None
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r") as f: original_uuid_str = f.read().strip()
            if original_uuid_str:
                try: uuid.UUID(original_uuid_str); logging.info(f"Read original UUID from {ORIGINAL_WATERMARK_FILE}")
                except ValueError: logging.error(f"Content of {ORIGINAL_WATERMARK_FILE} is not valid UUID."); original_uuid_str = None
            else: logging.warning(f"{ORIGINAL_WATERMARK_FILE} is empty."); original_uuid_str = None
        except Exception as e: logging.error(f"Error reading {ORIGINAL_WATERMARK_FILE}: {e}"); original_uuid_str = None
    else: logging.warning(f"{ORIGINAL_WATERMARK_FILE} not found.")

    logging.info("--- Starting Extraction Main Process ---")
    if not os.path.exists(input_video): logging.critical(f"Input video not found: '{input_video}'."); print(f"\nERROR: Input video '{input_video}' not found."); return

    frames, input_fps = read_video(input_video)
    if not frames: logging.critical(f"Failed to read frames from {input_video}."); return
    logging.info(f"Read {len(frames)} frames for extraction (Reported FPS: {input_fps:.2f})")

    extracted_payload_bytes: Optional[bytes] = extract_watermark_from_video(
        frames=frames, n_rings=N_RINGS, ring_selection_method=RING_SELECTION_METHOD,
        ring_selection_metric=RING_SELECTION_METRIC, default_ring_index=DEFAULT_RING_INDEX,
        embed_component=EMBED_COMPONENT, num_rings_to_use=NUM_RINGS_TO_USE,
        use_saved_rings=USE_SAVED_RINGS, coverage_percentage=COVERAGE_PERCENTAGE,
        use_ecc=USE_ECC, bch_m=BCH_M, bch_t=BCH_T, payload_len_bytes=PAYLOAD_LEN_BYTES
    )

    print(f"\n--- Extraction Results ---")
    extracted_uuid: Optional[uuid.UUID] = None
    if extracted_payload_bytes is not None:
        if len(extracted_payload_bytes) == PAYLOAD_LEN_BYTES:
            try:
                extracted_uuid = uuid.UUID(bytes=extracted_payload_bytes)
                print(f"  Successfully extracted payload bytes.")
                print(f"  Decoded UUID: {str(extracted_uuid)}")
                logging.info(f"Successfully decoded extracted payload to UUID: {str(extracted_uuid)}")
            except ValueError: logging.error(f"Extracted bytes are not a valid UUID."); print(f"  ERROR: Extracted bytes are not a valid UUID.")
        else: logging.error(f"Extracted payload length mismatch."); print(f"  ERROR: Extracted payload length mismatch.")
    else: logging.error("Extraction process failed."); print(f"  Extraction FAILED.")

    if original_uuid_str:
        print(f"  Original UUID: {original_uuid_str}")
        if extracted_uuid is not None and str(extracted_uuid) == original_uuid_str: print("\n  >>> UUID MATCH <<<"); logging.info("UUID verification successful.")
        else: print("\n  >>> !!! UUID MISMATCH or EXTRACTION FAILED !!! <<<"); logging.warning("UUID verification failed or extraction unsuccessful.")
    else: print("\n  Original UUID not available for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_script_time = time.time() - main_start_time
    logging.info(f"--- Total Extractor Script Time: {total_script_time:.2f} sec ---")
    print(f"\nExtraction finished. Check log: {LOG_FILENAME}")

# --- Запуск с Профилированием ---
if __name__ == "__main__":
    if USE_ECC and not BCHLIB_AVAILABLE:
        print("\nWARNING: USE_ECC is True, but bchlib library is not installed.")
        print("ECC decoding will be skipped.")
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
        profile_file = "profile_stats_extract.txt"
        try:
            with open(profile_file, "w") as f:
                stats_file = pstats.Stats(profiler, stream=f)
                stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved to {profile_file}")
            print(f"Profiling stats saved to {profile_file}")
        except IOError as e:
             logging.error(f"Could not save profiling stats to {profile_file}: {e}")