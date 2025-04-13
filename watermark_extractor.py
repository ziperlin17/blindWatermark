# Файл: extractor.py (Версия: Вариант 5 + GPU + MP4/H264 + Fixes)
import cv2
import numpy as np # Остается для CPU операций и типов
import random
import logging
import time
import json
import os
import imagehash
import hashlib
from PIL import Image
# SciPy убираем
# from scipy.fftpack import dct
# from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid # Предполагаем CPU-версию
from typing import List, Tuple, Optional, Dict, Any
import functools
import concurrent.futures
import uuid
from math import ceil

from watermark_embedder import OUTPUT_CODEC

# --- GPU / CuPy ---
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).use()
        CUPY_AVAILABLE = True
        logging.info("CuPy доступен и GPU инициализирован.")
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode()
        total_mem_gb = props['totalGlobalMem'] / (1024**3)
        logging.info(f"Используется GPU: {gpu_name}, Память: {total_mem_gb:.2f} GB")
    except cp.cuda.runtime.CUDARuntimeError as e:
        logging.warning(f"CuPy установлен, но не удалось инициализировать GPU: {e}. GPU НЕ будет использоваться.")
        CUPY_AVAILABLE = False
    except Exception as e:
        logging.warning(f"Неизвестная ошибка при инициализации CuPy/GPU: {e}. GPU НЕ будет использоваться.")
        CUPY_AVAILABLE = False
except ImportError:
    logging.warning("CuPy не найден. GPU ускорение будет НЕДОСТУПНО.")
    CUPY_AVAILABLE = False
    cp = np # Используем NumPy как замену

# --- BCH ---
try:
    import bchlib
    BCHLIB_AVAILABLE = True
except ImportError:
    BCHLIB_AVAILABLE = False

import cProfile
import pstats
from collections import Counter

# --- Параметры Алгоритма (должны совпадать с embedder.py) ---
ALPHA_MIN: float = 1.005
ALPHA_MAX: float = 1.1
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
CANDIDATE_POOL_SIZE: int = 4
EMBED_COMPONENT: int = 1 # 0=Y, 1=Cr, 2=Cb

# --- Параметры Извлечения и ECC ---
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 5
MAX_PACKET_REPEATS: int = 5

# --- Параметры Ввода/Вывода и Логирования ---
FPS: int = 30
LOG_FILENAME: str = 'watermarking_extract_gpu.log' # Новое имя лога
MAX_WORKERS_EXTRACT: Optional[int] = None
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
# INPUT_EXTENSION: str = '.avi' # Старое
INPUT_EXTENSION: str = '.mp4' # Новое по умолчанию

# --- Настройка Логирования ---
# (Код без изменений)
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME, filemode='w', level=logging.INFO,
    format='[%(asctime)s] %(levelname).1s %(threadName)s %(module)s:%(lineno)d - %(message)s'
)
# logging.getLogger().setLevel(logging.DEBUG)

# --- Проверки и Логирование Параметров ---
xp = cp if CUPY_AVAILABLE else np
effective_use_ecc = USE_ECC and BCHLIB_AVAILABLE
logging.info(f"--- Запуск Скрипта Извлечения (Вариант 5, GPU: {CUPY_AVAILABLE}) ---")
logging.info(f"Используемый модуль для массивов: {'CuPy' if CUPY_AVAILABLE else 'NumPy'}")
logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit, Ожид. ECC: {USE_ECC}, Доступно: {BCHLIB_AVAILABLE}, Max Repeats: {MAX_PACKET_REPEATS}")
logging.info(f"N_RINGS_Total={N_RINGS}, Pool={CANDIDATE_POOL_SIZE}, Select={NUM_RINGS_TO_USE}, Bits/Pair: {BITS_PER_PAIR}")
logging.info(f"Ожид. Альфа диапазон: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}")
logging.info(f"Компонент для извлечения: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")

# Инициализация BCH и определение длины пакета
bch = None
payload_bits_len = PAYLOAD_LEN_BYTES * 8
packet_len_bits = payload_bits_len
ecc_bits_len = 0
if effective_use_ecc:
    try:
        bch = bchlib.BCH(m=BCH_M, t=BCH_T)
        if hasattr(bch, 'ecc_bits') and bch.ecc_bits > 0:
            ecc_bits_len = bch.ecc_bits
            packet_len_bits = payload_bits_len + ecc_bits_len
            logging.info(f"BCH инициализирован (m={BCH_M}, t={BCH_T}). Ожид. Payload={payload_bits_len}, ECC={ecc_bits_len} bits.")
            logging.info(f"===> Ожидаемая длина пакета (payload+ecc): {packet_len_bits} бит ({ceil(packet_len_bits / 8.0)} байт).")
        else:
            logging.warning("BCH создан, но ecc_bits недоступны. ECC декодирование невозможно.")
            effective_use_ecc = False; bch = None; packet_len_bits = payload_bits_len
    except Exception as e:
        logging.error(f"Ошибка инициализации BCH: {e}. ECC декодирование невозможно.", exc_info=False)
        effective_use_ecc = False; bch = None; packet_len_bits = payload_bits_len

if not effective_use_ecc:
    logging.warning("ECC не используется (отключен, недоступен или ошибка инициализации).")
    logging.info(f"Извлечение без ECC. Ожидаемая длина пакета = Длина payload = {packet_len_bits} бит.")

# Проверки размеров колец (без изменений)
if NUM_RINGS_TO_USE != BITS_PER_PAIR: NUM_RINGS_TO_USE = BITS_PER_PAIR
if CANDIDATE_POOL_SIZE < NUM_RINGS_TO_USE: CANDIDATE_POOL_SIZE = NUM_RINGS_TO_USE
if CANDIDATE_POOL_SIZE > N_RINGS: CANDIDATE_POOL_SIZE = N_RINGS

# --- Основные Функции (Адаптированные для GPU/CuPy где возможно) ---

def dct_1d(signal_1d: cp.ndarray) -> cp.ndarray:
    """1D DCT с использованием CuPy."""
    # (Код идентичен embedder.py)
    if not isinstance(signal_1d, cp.ndarray): signal_1d = cp.asarray(signal_1d)
    return cp.fft.dct(signal_1d, type=2, norm='ortho')

# idct_1d не нужен в экстракторе

def dtcwt_transform(y_plane_cpu: np.ndarray, frame_number: int = -1) -> Optional[Tuple[Pyramid, np.ndarray]]:
    """Прямое DTCWT (CPU). Возвращает Пирамиду (CPU) и LL-подполосу (CPU)."""
    # (Код идентичен embedder.py)
    func_start_time = time.time()
    if np.any(np.isnan(y_plane_cpu)): y_plane_cpu = np.nan_to_num(y_plane_cpu)
    if y_plane_cpu.ndim != 2 or y_plane_cpu.size == 0 or y_plane_cpu.shape[0]<4 or y_plane_cpu.shape[1]<4:
         logging.error(f"[F:{frame_number}] Invalid input for DTCWT transform."); return None
    try:
        t = Transform2d()
        rows, cols = y_plane_cpu.shape
        pad_rows = rows % 2 != 0; pad_cols = cols % 2 != 0
        if pad_rows or pad_cols: y_plane_padded_cpu = np.pad(y_plane_cpu, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        else: y_plane_padded_cpu = y_plane_cpu
        pyramid_cpu = t.forward(y_plane_padded_cpu.astype(np.float32), nlevels=1)
        if not hasattr(pyramid_cpu, 'lowpass') or pyramid_cpu.lowpass is None:
            logging.error(f"[F:{frame_number}] DTCWT failed (CPU)."); return None
        pyramid_cpu.padding_info = (pad_rows, pad_cols)
        ll_subband_cpu = pyramid_cpu.lowpass.copy()
        # logging.debug(f"[F:{frame_number}] DTCWT transform time (CPU): {time.time() - func_start_time:.4f}s")
        return pyramid_cpu, ll_subband_cpu
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception during DTCWT transform (CPU): {e}", exc_info=True)
        return None

# dtcwt_inverse не нужен

# --- Кэширование для Ring Division (CPU) ---
# (Код идентичен embedder.py)
@functools.lru_cache(maxsize=8)
def _ring_division_internal_cpu(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    func_start_time = time.time(); H, W = subband_shape
    if H < 2 or W < 2: return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0; rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2); min_dist, max_dist = 0.0, np.max(distances)
    if max_dist < 1e-6: ring_bins = np.array([0.0, 1.0]); n_rings_eff = 1
    else: ring_bins = np.linspace(min_dist, max_dist + 1e-6, n_rings + 1); n_rings_eff = n_rings
    if len(ring_bins) < 2: return [None] * n_rings
    ring_indices = np.digitize(distances, ring_bins) - 1; ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    rings_coords_np: List[Optional[np.ndarray]] = [None] * n_rings
    for ring_idx in range(n_rings_eff):
        coords_for_ring_np = np.argwhere(ring_indices == ring_idx)
        if coords_for_ring_np.shape[0] > 0: rings_coords_np[ring_idx] = coords_for_ring_np
    # logging.debug(f"_ring_division_internal_cpu calc time for shape {subband_shape}: {time.time() - func_start_time:.6f}s")
    return rings_coords_np

@functools.lru_cache(maxsize=8)
def get_ring_coords_cached_cpu(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    logging.debug(f"Cache check for ring_division_cpu shape={subband_shape}, n_rings={n_rings}.")
    return _ring_division_internal_cpu(subband_shape, n_rings)

def ring_division_coords_cpu(subband_shape: Tuple[int, int], n_rings: int = N_RINGS, frame_number: int = -1) -> List[Optional[np.ndarray]]:
    try:
        coords_list_np = get_ring_coords_cached_cpu(subband_shape, n_rings)
        if not isinstance(coords_list_np, list) or not all(isinstance(item, (np.ndarray, type(None))) for item in coords_list_np):
            logging.error(f"[F:{frame_number}] Cached ring division (CPU) result invalid. Recalculating.")
            get_ring_coords_cached_cpu.cache_clear()
            coords_list_np = _ring_division_internal_cpu(subband_shape, n_rings)
        return [arr.copy() if arr is not None else None for arr in coords_list_np]
    except Exception as e:
        logging.error(f"[F:{frame_number}] Exception in ring_division_coords_cpu: {e}", exc_info=True)
        return [None] * n_rings

def get_ring_values(ll_subband_gpu: cp.ndarray, ring_coords_cpu: np.ndarray) -> cp.ndarray:
    """Извлекает значения пикселей из GPU подполосы по CPU координатам."""
    # (Код идентичен embedder.py)
    if ring_coords_cpu is None or ring_coords_cpu.size == 0: return cp.array([], dtype=cp.float32)
    try:
        rows_cpu, cols_cpu = ring_coords_cpu[:, 0], ring_coords_cpu[:, 1]
        return ll_subband_gpu[rows_cpu, cols_cpu].astype(cp.float32)
    except IndexError:
         logging.error(f"IndexError extracting ring values from GPU subband (shape {ll_subband_gpu.shape}) using CPU coords (shape {ring_coords_cpu.shape}).")
         return cp.array([], dtype=cp.float32)
    except Exception as e:
        logging.error(f"Error in get_ring_values: {e}", exc_info=True)
        return cp.array([], dtype=cp.float32)


def calculate_entropies_gpu(ring_vals_gpu: cp.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    """Вычисляет энтропии на GPU."""
    # (Код идентичен embedder.py)
    eps = 1e-12; visual_entropy = 0.0; edge_entropy = 0.0
    if ring_vals_gpu.size == 0: return visual_entropy, edge_entropy
    try:
        min_v_gpu, max_v_gpu = cp.min(ring_vals_gpu), cp.max(ring_vals_gpu)
        if float(min_v_gpu.get()) < 0.0 or float(max_v_gpu.get()) > 1.0:
            ring_vals_clipped_gpu = cp.clip(ring_vals_gpu, 0.0, 1.0)
        else: ring_vals_clipped_gpu = ring_vals_gpu
        hist_gpu, _ = cp.histogram(ring_vals_clipped_gpu, bins=256, range=(0.0, 1.0))
        total_count = ring_vals_clipped_gpu.size;
        if total_count == 0: return 0.0, 0.0
        probabilities_gpu = hist_gpu / total_count
        probabilities_gpu = probabilities_gpu[probabilities_gpu > eps]
        if probabilities_gpu.size == 0: return 0.0, 0.0
        visual_entropy_gpu = -cp.sum(probabilities_gpu * cp.log2(probabilities_gpu))
        edge_entropy_gpu = -cp.sum(probabilities_gpu * cp.exp(1.0 - probabilities_gpu))
        visual_entropy = float(cp.clip(visual_entropy_gpu, 0.0, MAX_THEORETICAL_ENTROPY).get())
        edge_entropy = float(edge_entropy_gpu.get())
    except Exception as e:
        logging.error(f"[F:{frame_number}, R:{ring_index}] Error calculating GPU entropy: {e}")
    return visual_entropy, edge_entropy

def compute_adaptive_alpha_entropy_gpu(ring_vals_gpu: cp.ndarray, ring_index: int, frame_number: int) -> float:
    """Вычисляет адаптивную альфа на GPU (для порога)."""
    # (Код идентичен embedder.py)
    min_pixels_for_alpha = 10; final_alpha = ALPHA_MIN
    if ring_vals_gpu.size < min_pixels_for_alpha: return final_alpha
    try:
        visual_entropy, _ = calculate_entropies_gpu(ring_vals_gpu, frame_number, ring_index)
        local_variance_gpu = cp.var(ring_vals_gpu)
        entropy_norm = np.clip(visual_entropy / MAX_THEORETICAL_ENTROPY, 0.0, 1.0)
        variance_midpoint = 0.005; variance_scale = 500
        local_variance_float = float(local_variance_gpu.get())
        texture_norm = 1.0 / (1.0 + np.exp(-variance_scale * (local_variance_float - variance_midpoint)))
        w_entropy = 0.6; w_texture = 0.4
        masking_factor = np.clip((w_entropy * entropy_norm + w_texture * texture_norm), 0.0, 1.0)
        final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * masking_factor
        final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)
    except Exception as e:
        logging.error(f"[F:{frame_number}, R:{ring_index}] Error computing GPU adaptive alpha: {e}")
        final_alpha = ALPHA_MIN # Fallback
    # logging.debug(f"[F:{frame_number}, R:{ring_index}] GPU Adaptive Alpha -> {final_alpha:.4f}")
    return final_alpha


def get_fixed_pseudo_random_rings(pair_idx: int, n_rings: int, num_to_select: int) -> List[int]:
    """Генерирует детерминированный псевдослучайный список УНИКАЛЬНЫХ индексов колец."""
    # (Код идентичен embedder.py)
    if num_to_select > n_rings: num_to_select = n_rings
    if num_to_select <= 0: return []
    pair_idx_bytes = str(pair_idx).encode('utf-8'); hash_object = hashlib.sha256(pair_idx_bytes)
    hash_digest = hash_object.digest(); seed = int.from_bytes(hash_digest, 'big')
    prng = random.Random(seed); available_indices = list(range(n_rings))
    selected_indices = prng.sample(available_indices, num_to_select)
    # logging.debug(f"[P:{pair_idx}] Fixed pseudo-random rings: {selected_indices}")
    return sorted(selected_indices)

# calculate_perceptual_mask не нужен
# Функции load_saved_rings, select_embedding_rings и др. удалены

def bits_to_bytes(bits: List[Optional[int]]) -> Optional[bytearray]:
    """Конвертирует список бит (0, 1, None) в bytearray."""
    # (Код идентичен embedder.py)
    valid_bits = [b for b in bits if b is not None]
    if not valid_bits: logging.warning("bits_to_bytes: No valid bits."); return bytearray()
    num_valid_bits = len(valid_bits); remainder = num_valid_bits % 8
    if remainder != 0:
        padding_needed = 8 - remainder
        # logging.warning(f"bits_to_bytes: Padding {padding_needed} zeros.")
        valid_bits.extend([0] * padding_needed)
    byte_array = bytearray()
    try:
        for i in range(0, len(valid_bits), 8):
            byte_str = "".join(map(str, valid_bits[i:i+8])); byte_val = int(byte_str, 2)
            byte_array.append(byte_val)
    except (ValueError, TypeError) as e: logging.error(f"Bits to byte conversion error: {e}"); return None
    return byte_array


def decode_ecc(packet_bytes: bytearray, bch: 'bchlib.BCH', expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    """Декодирует пакет с помощью BCH, используя метод decode()."""
    # (Код идентичен предыдущей версии extractor.py с исправлениями)
    if not BCHLIB_AVAILABLE or bch is None:
        logging.warning("decode_ecc: bchlib unavailable/uninitialized.")
        if len(packet_bytes) >= expected_data_len_bytes: return bytes(packet_bytes[:expected_data_len_bytes]), 0
        else: logging.error("Packet too short without ECC."); return None, -1
    try:
        current_packet_len_bytes = len(packet_bytes)
        logging.debug(f"decode_ecc: Decoding {current_packet_len_bytes} bytes using bch.decode(). Expecting {expected_data_len_bytes} data bytes.")
        try:
            result = bch.decode(packet_bytes) # Основной вызов
            if isinstance(result, tuple) and len(result) == 3:
                 corrected_data, _, errors_corrected = result # Игнорируем исправленный ECC
                 if errors_corrected == -1: logging.error("ECC: Uncorrectable errors (decode)."); return None, -1
                 else:
                     logging.info(f"ECC: Corrected {errors_corrected} bits (decode).")
                     final_data = bytes(corrected_data[:expected_data_len_bytes])
                     if len(final_data) != expected_data_len_bytes: logging.warning(f"Corrected data len {len(final_data)} != expected {expected_data_len_bytes}.")
                     return final_data, errors_corrected
            elif isinstance(result, int): # Попытка обработки inplace-подобного ответа
                 errors_corrected = result
                 if errors_corrected == -1: logging.error("ECC: Uncorrectable errors (decode, inplace assumption)."); return None, -1
                 else:
                     logging.info(f"ECC: Corrected {errors_corrected} bits (decode, inplace assumption).")
                     final_data = bytes(packet_bytes[:expected_data_len_bytes]) # Берем из измененного оригинала
                     if len(final_data) != expected_data_len_bytes: logging.warning(f"Corrected data len {len(final_data)} != expected {expected_data_len_bytes} (inplace assumption).")
                     return final_data, errors_corrected
            else: logging.error(f"Unexpected result format from bch.decode(): {type(result)}"); return None, -1
        except AttributeError: logging.error("bch.decode() method not found."); return None, -1
        except TypeError as te: logging.error(f"TypeError calling bch.decode(): {te}"); return None, -1
    except Exception as e: logging.error(f"Exception in decode_ecc: {e}", exc_info=True); return None, -1


def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Читает кадры из видеофайла (на CPU)."""
    # (Код идентичен embedder.py)
    func_start_time = time.time(); logging.info(f"Reading video from: {video_path}")
    frames: List[np.ndarray] = []; fps = float(FPS); cap = None; expected_height, expected_width = -1, -1
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed to open video: {video_path}"); return frames, fps
        fps_read = cap.get(cv2.CAP_PROP_FPS); fps = float(fps_read) if fps_read > 0 else fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, ~{frame_count_prop} frames")
        if width <= 0 or height <= 0: logging.error("Invalid dimensions"); cap.release(); return [], fps
        expected_height, expected_width = height, width; frame_index = 0; read_count = 0; none_frame_count = 0; invalid_shape_count = 0
        while True:
            ret, frame = cap.read(); frame_number_log = frame_index + 1
            if not ret: logging.info(f"EOF after {read_count} frames (idx {frame_index})."); break
            if frame is None: none_frame_count += 1; frame_index += 1; continue
            if frame.ndim == 3 and frame.shape[:2] == (expected_height, expected_width) and frame.dtype == np.uint8:
                frames.append(frame); read_count += 1
            else: invalid_shape_count += 1
            frame_index += 1
        logging.info(f"Read: Valid={len(frames)}, None={none_frame_count}, Invalid={invalid_shape_count}")
    except Exception as e: logging.error(f"Video reading exception: {e}", exc_info=True)
    finally:
        if cap is not None: cap.release()
    if not frames: logging.error(f"No valid frames read from {video_path}")
    logging.debug(f"Read video time: {time.time() - func_start_time:.4f}s")
    return frames, fps

# --- Основной Извлекающий Процесс (GPU Адаптированный) ---

def extract_frame_pair_gpu(
        frame1_cpu: np.ndarray, frame2_cpu: np.ndarray, ring_index: int,
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT,
        gpu_id: int = 0
) -> Optional[int]:
    """Извлекает один бит из указанного кольца пары кадров, используя GPU."""
    if not CUPY_AVAILABLE: return None # CPU fallback не реализован

    try:
        cp.cuda.Device(gpu_id).use()
        xp = cp
    except Exception as e_dev:
        logging.error(f"Failed to set GPU device {gpu_id} in extract thread: {e_dev}")
        return None

    func_start_time = time.time()
    pair_num_log = frame_number // 2
    extracted_bit: Optional[int] = None
    L1_gpu, L2_gpu = None, None # Инициализация для finally

    try:
        # --- CPU -> GPU ---
        # YCrCb (CPU)
        frame1_ycrcb_cpu = cv2.cvtColor(frame1_cpu, cv2.COLOR_BGR2YCrCb)
        frame2_ycrcb_cpu = cv2.cvtColor(frame2_cpu, cv2.COLOR_BGR2YCrCb)
        # Компонент (CPU -> GPU)
        comp1_cpu = (frame1_ycrcb_cpu[:, :, embed_component].astype(np.float32) / 255.0)
        comp2_cpu = (frame2_ycrcb_cpu[:, :, embed_component].astype(np.float32) / 255.0)
        comp1_gpu = xp.asarray(comp1_cpu)
        comp2_gpu = xp.asarray(comp2_cpu)

        # DTCWT (CPU) -> LL (GPU)
        dtcwt_res1 = dtcwt_transform(comp1_cpu, frame_number=frame_number)
        dtcwt_res2 = dtcwt_transform(comp2_cpu, frame_number=frame_number + 1)
        if dtcwt_res1 is None or dtcwt_res2 is None: raise RuntimeError("DTCWT failed")
        _, L1_cpu = dtcwt_res1; _, L2_cpu = dtcwt_res2
        L1_gpu = xp.asarray(L1_cpu); L2_gpu = xp.asarray(L2_cpu)
        ll_shape = L1_gpu.shape

        # Координаты колец (CPU)
        rings_coords_cpu_list = ring_division_coords_cpu(ll_shape, n_rings, frame_number)
        coords_cpu = rings_coords_cpu_list[ring_index]
        if coords_cpu is None or coords_cpu.size == 0: raise ValueError("Invalid ring coords")

        # --- GPU Вычисления ---
        # Значения колец (GPU)
        ring_vals_1_gpu = get_ring_values(L1_gpu, coords_cpu)
        ring_vals_2_gpu = get_ring_values(L2_gpu, coords_cpu)
        if ring_vals_1_gpu.size == 0 or ring_vals_2_gpu.size == 0: raise ValueError("Empty ring values")

        # DCT (GPU)
        dct1_gpu = dct_1d(ring_vals_1_gpu)
        dct2_gpu = dct_1d(ring_vals_2_gpu)

        # SVD (GPU)
        S1_vals_gpu = xp.linalg.svd(dct1_gpu.reshape(-1, 1), compute_uv=False)
        S2_vals_gpu = xp.linalg.svd(dct2_gpu.reshape(-1, 1), compute_uv=False)
        s1_gpu = S1_vals_gpu[0] if S1_vals_gpu.size > 0 else xp.array(0.0, dtype=xp.float32)
        s2_gpu = S2_vals_gpu[0] if S2_vals_gpu.size > 0 else xp.array(0.0, dtype=xp.float32)

        # Альфа (GPU -> CPU)
        alpha = compute_adaptive_alpha_entropy_gpu(ring_vals_1_gpu, ring_index, frame_number)

        # --- CPU Логика ---
        s1 = float(s1_gpu.get()); s2 = float(s2_gpu.get())
        eps = 1e-12
        threshold = (alpha + 1.0 / (alpha + eps)) / 2.0
        ratio = s1 / (s2 + eps)
        extracted_bit = 0 if ratio >= threshold else 1

        logging.info(f"[GPU-{gpu_id} P:{pair_num_log}, R:{ring_index}] Extracted: s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f} vs thr={threshold:.4f} (alpha={alpha:.4f}) -> Bit={extracted_bit}")

    except Exception as e:
        logging.error(f"!!! EXCEPTION in extract_frame_pair_gpu (Pair {pair_num_log}, Ring {ring_index}): {e}", exc_info=True)
        extracted_bit = None
    # finally: # Очистка памяти не нужна здесь, т.к. массивы локальные
    #     del L1_gpu, L2_gpu, comp1_gpu, comp2_gpu # Пример

    # total_pair_time = time.time() - func_start_time
    # logging.debug(f"[GPU-{gpu_id} P:{pair_num_log}, R:{ring_index}] Extract time: {total_pair_time:.4f}s")
    return extracted_bit


def extract_frame_pair_multi_ring_gpu(
        frame1_cpu: np.ndarray, frame2_cpu: np.ndarray, ring_indices: List[int], # Кольца для извлечения
        n_rings: int = N_RINGS, frame_number: int = 0,
        embed_component: int = EMBED_COMPONENT,
        gpu_id: int = 0
) -> List[Optional[int]]:
    """Извлекает по биту из каждого кольца, используя GPU версию extract_frame_pair_gpu."""
    pair_num_log = frame_number // 2
    if not ring_indices: return []

    extracted_bits: List[Optional[int]] = []
    logging.debug(f"[GPU-{gpu_id} P:{pair_num_log}] Multi-ring GPU extraction for rings: {ring_indices}")

    # Здесь можно оптимизировать: выполнить DTCWT один раз, затем параллельно SVD для колец.
    # Но для простоты пока последовательный вызов GPU функции.
    for i, ring_idx in enumerate(ring_indices):
        bit = extract_frame_pair_gpu(frame1_cpu, frame2_cpu, ring_idx, n_rings, frame_number, embed_component, gpu_id)
        extracted_bits.append(bit)
        # if bit is None: logging.warning(...) # Лог уже внутри extract_frame_pair_gpu

    logging.info(f"[GPU-{gpu_id} P:{pair_num_log}] Multi-ring GPU extracted bits: {extracted_bits} from target rings {ring_indices}")
    return extracted_bits


def _extract_frame_pair_worker_gpu(args: Dict[str, Any]) -> Tuple[int, List[Optional[int]]]:
    """Воркер для извлечения (Вариант 5) с использованием GPU."""
    pair_idx = args['pair_idx']
    frame1_cpu = args['frame1']
    frame2_cpu = args['frame2']
    gpu_id = args['gpu_id']

    n_rings: int = args['n_rings']
    num_rings_to_use: int = args['num_rings_to_use']
    candidate_pool_size: int = args['candidate_pool_size']
    embed_component: int = args['embed_component']
    frame_number = 2 * pair_idx
    extracted_bits: List[Optional[int]] = [None] * num_rings_to_use

    if not CUPY_AVAILABLE: return pair_idx, extracted_bits

    try:
        cp.cuda.Device(gpu_id).use(); xp = cp
    except Exception as e_dev:
        logging.error(f"Failed set GPU {gpu_id} in worker {pair_idx}: {e_dev}"); return pair_idx, extracted_bits

    target_ring_indices: List[int] = []

    try:
        # logging.debug(f"[Worker GPU-{gpu_id} P:{pair_idx}] Starting extraction.")
        # --- Этап 1: Пул кандидатов (CPU) ---
        candidate_ring_indices = get_fixed_pseudo_random_rings(pair_idx, n_rings, candidate_pool_size)
        # logging.info(f"[Worker GPU-{gpu_id} P:{pair_idx}] Candidates: {candidate_ring_indices}")
        if not candidate_ring_indices or len(candidate_ring_indices) < num_rings_to_use:
            raise ValueError(f"Not enough candidates: {len(candidate_ring_indices)}")

        # --- Этап 2: Выбор лучших по энтропии (GPU) ---
        try:
            # Получаем L1' (CPU -> GPU)
            comp1_cpu = (frame1_cpu[:, :, embed_component].astype(np.float32) / 255.0)
            dtcwt_res = dtcwt_transform(comp1_cpu, frame_number=frame_number)
            if dtcwt_res is None: raise RuntimeError("DTCWT failed for L1' calc")
            _, L1_cpu = dtcwt_res
            L1_gpu = xp.asarray(L1_cpu); ll_shape = L1_gpu.shape
            # Координаты колец (CPU)
            rings_coords_cpu_list = ring_division_coords_cpu(ll_shape, n_rings, frame_number)
            # Энтропия кандидатов (GPU)
            entropy_values: List[Tuple[float, int]] = []
            min_pixels = 10
            for r_idx in candidate_ring_indices:
                current_entropy = -1.0
                if 0 <= r_idx < len(rings_coords_cpu_list): # Проверка индекса
                    coords_cpu = rings_coords_cpu_list[r_idx]
                    if coords_cpu is not None and coords_cpu.size >= min_pixels * 2:
                        ring_vals_gpu = get_ring_values(L1_gpu, coords_cpu)
                        if ring_vals_gpu.size >= min_pixels:
                            v_entropy, _ = calculate_entropies_gpu(ring_vals_gpu, frame_number, r_idx)
                            if np.isfinite(v_entropy): current_entropy = v_entropy
                entropy_values.append((current_entropy, r_idx))
            # Выбор лучших (CPU)
            entropy_values.sort(key=lambda x: x[0], reverse=True)
            valid_candidates = [(e, i) for e, i in entropy_values if e >= 0.0]
            if len(valid_candidates) < num_rings_to_use:
                 raise ValueError(f"Only {len(valid_candidates)}/{num_rings_to_use} valid rings in pool on L1'")
            target_ring_indices = [idx for _, idx in valid_candidates[:num_rings_to_use]]
            logging.info(f"[Worker GPU-{gpu_id} P:{pair_idx}] Target rings: {target_ring_indices} from {candidate_ring_indices}")
            del L1_gpu # Освобождаем память L1'

        except Exception as e_select:
            logging.error(f"[Worker GPU-{gpu_id} P:{pair_idx}] Error during GPU entropy selection: {e_select}", exc_info=True)
            raise # Передаем ошибку

        # --- Этап 3: Извлечение (GPU) ---
        if not target_ring_indices or len(target_ring_indices) != num_rings_to_use:
             raise ValueError("Invalid target rings selected")

        extracted_bits = extract_frame_pair_multi_ring_gpu(
            frame1_cpu, frame2_cpu, ring_indices=target_ring_indices,
            n_rings=n_rings, frame_number=frame_number,
            embed_component=embed_component, gpu_id=gpu_id
        )

        # Проверка результата
        if len(extracted_bits) != num_rings_to_use:
             logging.warning(f"[Worker GPU-{gpu_id} P:{pair_idx}] Extracted bits len mismatch.")
             # Дополняем None, если нужно
             extracted_bits.extend([None] * (num_rings_to_use - len(extracted_bits)))
             extracted_bits = extracted_bits[:num_rings_to_use]

        return pair_idx, extracted_bits

    except Exception as e:
        logging.error(f"!!! EXCEPTION in worker GPU-{gpu_id} P:{pair_idx} (F:{frame_number}): {e}", exc_info=True)
        return pair_idx, [None] * num_rings_to_use # Возвращаем None list
    # finally: # Очистка памяти не нужна здесь
    #      pass


def extract_watermark_from_video(
        frames_cpu: List[np.ndarray],
        n_rings: int = N_RINGS,
        num_rings_to_use: int = NUM_RINGS_TO_USE,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        bits_per_pair: int = BITS_PER_PAIR,
        embed_component: int = EMBED_COMPONENT,
        max_packet_repeats: int = MAX_PACKET_REPEATS,
        use_ecc: bool = USE_ECC,
        bch_m: int = BCH_M, bch_t: int = BCH_T,
        payload_len_bytes: int = PAYLOAD_LEN_BYTES
) -> Optional[bytes]:
    """Извлекает ВЗ (Вариант 5, GPU) и проводит голосование."""
    global packet_len_bits # Используем глобальную переменную, рассчитанную при старте
    logging.info(f"Starting parallel {'GPU' if CUPY_AVAILABLE else 'CPU (ERROR: Fallback not implemented)'} extraction (Variant 5 - Pool={candidate_pool_size}, Select={num_rings_to_use})")
    logging.info(f"Bits/Pair: {bits_per_pair}, Packet Voting (Max Repeats: {max_packet_repeats}).")
    logging.info(f"Ring Selection: Fixed Pool + Entropy on L1', Component: {['Y', 'Cr', 'Cb'][embed_component]}, Expect ECC: {use_ecc}")

    start_time = time.time()
    num_frames = len(frames_cpu); total_pairs_available = num_frames // 2
    processed_pairs_count = 0; failed_pair_extractions = 0
    if total_pairs_available == 0: logging.error("Not enough frames."); return None
    if packet_len_bits <= 0: logging.error("Packet length is zero."); return None

    # --- Определение количества пар ---
    pairs_to_extract_needed = ceil(max_packet_repeats * packet_len_bits / bits_per_pair)
    pairs_to_extract = min(total_pairs_available, pairs_to_extract_needed)
    logging.info(f"Attempting to extract from {pairs_to_extract} pairs (up to {total_pairs_available} available).")
    if pairs_to_extract == 0: logging.warning("No pairs to extract."); return None

    # --- Подготовка задач ---
    extracted_bits_per_pair: Dict[int, List[Optional[int]]] = {}
    tasks_args = []
    num_gpus = cp.cuda.runtime.getDeviceCount() if CUPY_AVAILABLE else 0
    gpu_counter = 0

    for i in range(pairs_to_extract):
        idx1 = 2 * i; idx2 = idx1 + 1
        if idx2 >= num_frames or frames_cpu[idx1] is None or frames_cpu[idx2] is None:
             logging.warning(f"Skipping pair {i} due to missing frames.")
             continue # Пропускаем эту пару
        current_gpu_id = gpu_counter % num_gpus if num_gpus > 0 else 0; gpu_counter += 1
        args = {
            'pair_idx': i,
            'frame1': frames_cpu[idx1], 'frame2': frames_cpu[idx2],
            'n_rings': n_rings,
            'embed_component': embed_component,
            'num_rings_to_use': num_rings_to_use,
            'candidate_pool_size': candidate_pool_size,
            'gpu_id': current_gpu_id
        }
        tasks_args.append(args)
    if not tasks_args: logging.error("No valid tasks created."); return None

    # --- Параллельное извлечение ---
    # === ИЗМЕНЕНИЕ НАЧАЛО ===
    # Используем ТОЛЬКО GPU воркер.
    worker_func = _extract_frame_pair_worker_gpu

    if not CUPY_AVAILABLE:
        logging.error("CuPy/GPU не доступен, но код пытается использовать GPU воркер. Остановка извлечения.")
        print("\nERROR: CuPy/GPU is required for this script version but not available.")
        return None # Возвращаем None, т.к. извлечение невозможно
    # === ИЗМЕНЕНИЕ КОНЕЦ ===

    try:
        logging.info(f"Submitting {len(tasks_args)} tasks to ThreadPoolExecutor (max_workers={MAX_WORKERS_EXTRACT}) using GPU workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_EXTRACT) as executor:
            future_to_pair_idx = {executor.submit(worker_func, arg): arg['pair_idx'] for arg in tasks_args}
            for future in concurrent.futures.as_completed(future_to_pair_idx):
                pair_idx = future_to_pair_idx[future]
                try:
                    # Получаем результат: индекс пары и список извлеченных бит
                    p_idx_check, bits_result_list = future.result()
                    if p_idx_check != pair_idx:
                         logging.error(f"Worker for pair {pair_idx} returned wrong index {p_idx_check}!")
                    extracted_bits_per_pair[pair_idx] = bits_result_list
                    # Проверяем, есть ли None в результате
                    if bits_result_list is None or None in bits_result_list:
                         failed_pair_extractions += 1
                         logging.warning(f"Pair {pair_idx} extraction resulted in None values: {bits_result_list}")
                    processed_pairs_count += 1
                    logging.debug(f"Pair {pair_idx} completed. Result: {bits_result_list} ({processed_pairs_count}/{len(tasks_args)})")
                except Exception as exc:
                     logging.error(f'Pair {pair_idx} generated exception in executor: {exc}', exc_info=True)
                     extracted_bits_per_pair[pair_idx] = [None] * bits_per_pair # Заполняем None при ошибке
                     failed_pair_extractions += 1
                     processed_pairs_count += 1 # Считаем обработанной
    except Exception as e:
        logging.critical(f"CRITICAL ERROR during Executor: {e}", exc_info=True)
        # Заполняем все неоконченные None
        for i in range(pairs_to_extract):
            if i not in extracted_bits_per_pair: # Используем range(pairs_to_extract), т.к. не все могут быть в tasks_args
                 extracted_bits_per_pair[i] = [None] * bits_per_pair

    logging.info(f"Parallel extraction finished. Pairs submitted: {len(tasks_args)}. Processed: {processed_pairs_count}. Pairs with failed extraction (at least one None bit): {failed_pair_extractions}.")

    # --- Сборка бит и декодирование ---
    extracted_bits_all: List[Optional[int]] = []
    # Собираем биты в правильном порядке от 0 до pairs_to_extract-1
    for i in range(pairs_to_extract):
        bits_list = extracted_bits_per_pair.get(i, [None] * bits_per_pair) # Получаем результат или None list
        extracted_bits_all.extend(bits_list)

    total_extracted_bits_count = len(extracted_bits_all)
    logging.info(f"Total bits collected (incl. None): {total_extracted_bits_count} (Expected: {pairs_to_extract * bits_per_pair})")

    # --- ГОЛОСОВАНИЕ ---
    num_potential_packets = total_extracted_bits_count // packet_len_bits
    if num_potential_packets == 0 and total_extracted_bits_count >= packet_len_bits: num_potential_packets = 1
    logging.info(f"Attempting to decode up to {num_potential_packets} potential packets (Packet Size: {packet_len_bits} bits)...")

    decoded_payloads: List[bytes] = []; decode_success = 0; decode_fail = 0; ecc_corrected = 0
    for i in range(num_potential_packets):
        start_idx = i * packet_len_bits; end_idx = start_idx + packet_len_bits
        if end_idx > total_extracted_bits_count: logging.warning(f"Not enough bits for full packet #{i+1}. Stop."); break
        packet_bits_list = extracted_bits_all[start_idx:end_idx]
        if None in packet_bits_list: logging.warning(f"Packet #{i+1} has None bits. Skip decode."); decode_fail += 1; continue
        packet_bytes = bits_to_bytes(packet_bits_list)
        if packet_bytes is None: logging.error(f"Bits to bytes failed for packet #{i+1}. Skip."); decode_fail += 1; continue

        payload: Optional[bytes] = None; errors: int = -1
        if effective_use_ecc and bch is not None:
            payload, errors = decode_ecc(packet_bytes, bch, payload_len_bytes)
        else:
            if len(packet_bytes) >= payload_len_bytes: payload = bytes(packet_bytes[:payload_len_bytes]); errors = 0
            else: payload = None; errors = -1

        if payload is not None and len(payload) == payload_len_bytes:
             decoded_payloads.append(payload); decode_success += 1
             if errors > 0: ecc_corrected += errors
        else:
            logging.warning(f"Failed decode or wrong payload len for packet #{i+1}.")
            decode_fail += 1

    logging.info(f"Decoding summary: Success={decode_success}, Failed={decode_fail}. ECC corrections: {ecc_corrected}.")
    if not decoded_payloads: logging.error("No valid payload decoded."); return None

    # --- Финальное Голосование ---
    payload_counts = Counter(decoded_payloads)
    logging.info("Voting results:")
    for pld, count in payload_counts.most_common():
        try: logging.info(f"  ID: {pld.hex()} - Votes: {count}")
        except: logging.info(f"  Payload (bytes): {pld} - Votes: {count}")
    most_common_payload, winner_count = payload_counts.most_common(1)[0]
    confidence = winner_count / decode_success if decode_success > 0 else 0.0
    logging.info(f"Winner selected with {winner_count}/{decode_success} votes (Confidence: {confidence:.1%}).")

    end_time = time.time()
    logging.info(f"Extraction finished. Total time: {end_time - start_time:.2f} sec.")
    return most_common_payload

def main():
    main_start_time = time.time()
    input_video_base = f"watermarked_video_v5_gpu_{OUTPUT_CODEC}" # Имя файла от embedder
    input_video = input_video_base + INPUT_EXTENSION
    original_id_hex: Optional[str] = None
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try: # Чтение ID для сравнения
            with open(ORIGINAL_WATERMARK_FILE, "r") as f: original_id_hex = f.read().strip()
            if original_id_hex and len(original_id_hex) == PAYLOAD_LEN_BYTES * 2: int(original_id_hex, 16); logging.info("Read original ID.")
            else: logging.error("Invalid original ID file."); original_id_hex = None
        except Exception as e: logging.error(f"Error reading original ID: {e}"); original_id_hex = None
    else: logging.warning("Original ID file not found.")

    logging.info(f"--- Starting Extraction Main Process (Variant 5, GPU: {CUPY_AVAILABLE}) ---")
    if not os.path.exists(input_video): logging.critical(f"Input not found: '{input_video}'."); print(f"\nERROR: Input '{input_video}' not found."); return

    # 1. Чтение видео (CPU)
    frames_cpu, input_fps = read_video(input_video)
    if not frames_cpu: logging.critical("Read failed."); return
    logging.info(f"Read {len(frames_cpu)} frames (FPS: {input_fps:.2f})")

    # 2. Извлечение ВЗ (GPU или CPU)
    extracted_payload_bytes: Optional[bytes] = extract_watermark_from_video(
        frames_cpu=frames_cpu, n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE,
        candidate_pool_size=CANDIDATE_POOL_SIZE, bits_per_pair=BITS_PER_PAIR,
        embed_component=EMBED_COMPONENT, max_packet_repeats=MAX_PACKET_REPEATS,
        use_ecc=USE_ECC, bch_m=BCH_M, bch_t=BCH_T, payload_len_bytes=PAYLOAD_LEN_BYTES
    )

    # 3. Вывод результата
    print(f"\n--- Extraction Results (Variant 5, GPU: {CUPY_AVAILABLE}) ---")
    extracted_id_hex: Optional[str] = None
    if extracted_payload_bytes is not None and len(extracted_payload_bytes) == PAYLOAD_LEN_BYTES:
        extracted_id_hex = extracted_payload_bytes.hex()
        print(f"  Successfully extracted payload.")
        print(f"  Decoded ID (Hex): {extracted_id_hex}")
        logging.info(f"Decoded ID: {extracted_id_hex}")
    else:
        logging.error("Extraction failed or returned invalid payload.")
        print(f"  Extraction FAILED.")

    # 4. Сравнение
    if original_id_hex:
        print(f"  Original ID (Hex): {original_id_hex}")
        if extracted_id_hex is not None and extracted_id_hex == original_id_hex: print("\n  >>> ID MATCH <<<"); logging.info("ID match.")
        else: print("\n  >>> !!! ID MISMATCH or FAILED !!! <<<"); logging.warning("ID mismatch or failed.")
    else: print("\n  (Original ID not available for comparison)")

    logging.info("--- Extraction Main Process Finished ---")
    total_script_time = time.time() - main_start_time
    logging.info(f"--- Total Extractor Script Time: {total_script_time:.2f} sec ---")
    print(f"\nExtraction finished. Log: {LOG_FILENAME}")


# --- Запуск с Профилированием ---
if __name__ == "__main__":
    if not CUPY_AVAILABLE:
         print("\nWARNING: CuPy not found or GPU not usable. Running on CPU.")
         print("         CPU worker '_extract_frame_pair_worker_cpu' is NOT implemented.")
         # exit()

    if USE_ECC and not BCHLIB_AVAILABLE:
        print("\nWARNING: USE_ECC=True, but 'bchlib' not installed. ECC decoding skipped.")

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception in main (Extractor): {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}. Check logs: {LOG_FILENAME}")
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        profile_file = "profile_stats_extract_gpu.txt"
        try:
            with open(profile_file, "w") as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved to {profile_file}")
            print(f"Profiling stats saved to {profile_file}")
        except IOError as e: logging.error(f"Could not save profile stats: {e}")