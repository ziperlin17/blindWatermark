# Файл: extractor.py (Версия: OpenCL Attempt, ThreadPool + Batches, Galois BCH, Syntax Fix V4 - МТ Версия Активна)
import cv2
import numpy as np
import random
import logging
import time
import json
import os
import hashlib
from PIL import Image # Оставим на всякий случай
from line_profiler import profile
# from line_profiler.explicit_profiler import profile # Закомментировано, т.к. импорт не стандартный
from scipy.fftpack import dct
from scipy.linalg import svd
import dtcwt
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import uuid
from math import ceil
import cProfile
import pstats
from collections import Counter
import sys

# --- Переменная для хранения информации об успехе переключения бэкенда ---
DTCWT_OPENCL_ENABLED = True # Флаг для OpenCL

# --- Попытка импорта и инициализации Galois ---
try:
    import galois
    logging.info("galois: импортирован.")
    _test_bch_ok = False; _test_decode_ok = False; BCH_CODE_OBJECT = None
    try:
        _test_m = 8
        _test_t = 9 # Желаемое t
        _test_n = (1 << _test_m) - 1 # n = 255
        _test_d = 2 * _test_t + 1 # d = 11 (Вычисляем d из t)

        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)

        if _test_t == 5:
            expected_k = 215
        elif _test_t == 7:
            expected_k = 201
        elif _test_t == 9:
            expected_k = 187  # <--- Установить правильное значение для вашего BCH_T
        elif _test_t == 11:
            expected_k = 173
        elif _test_t == 15:
            expected_k = 131
        else:
            logging.error(f"Неизвестное ожидаемое k для t={_test_t}")
            expected_k = -1
        if _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
             logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
             _test_bch_ok = True; BCH_CODE_OBJECT = _test_bch_galois
        else:
             logging.error(f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t={_test_bch_galois.t}, k={_test_bch_galois.k}.")
             _test_bch_ok = False; BCH_CODE_OBJECT = None

        if _test_bch_ok:
            _n_bits = _test_bch_galois.n; _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)
            GF2 = galois.GF(2); _dummy_cw_vec = GF2(_dummy_cw_bits)
            _msg, _flips = _test_bch_galois.decode(_dummy_cw_vec, errors=True)
            if _flips is not None: logging.info(f"galois: decode() test OK (flips={_flips})."); _test_decode_ok = True
            else: logging.warning("galois: decode() test failed?"); _test_decode_ok = False
    except ValueError as ve:
         logging.error(f"galois: ОШИБКА ValueError при инициализации BCH(d={_test_d}): {ve}")
         BCH_CODE_OBJECT = None; _test_bch_ok = False
    except Exception as test_err:
         logging.error(f"galois: ОШИБКА теста инициализации/декодирования: {test_err}", exc_info=True)
         BCH_CODE_OBJECT = None; _test_bch_ok = False

    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None
    if GALOIS_AVAILABLE: logging.info("galois: Тесты пройдены.")
    else: logging.warning("galois: Тесты НЕ ПРОЙДЕНЫ. ECC будет отключен или работать некорректно.")




except ImportError: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; logging.info("galois library not found.")
except Exception as import_err: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; logging.error(f"galois: Ошибка импорта: {import_err}", exc_info=True)


LAMBDA_PARAM: float = 0.01
ALPHA_MIN: float = 1.13
ALPHA_MAX: float = 1.27
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2 # Cb -
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection'
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 9
# MAX_PACKET_REPEATS: int = 5
FPS: int = 30





LOG_FILENAME: str = 'watermarking_extract_opencl_batched.log'
INPUT_EXTENSION: str = '.mp4'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS_EXTRACT: Optional[int] = None

for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование Конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Извлечения (ThreadPool + Batches + OpenCL Attempt) ---")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
# logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit, Ожид. ECC: {USE_ECC} (Galois BCH m={BCH_M}, t={BCH_T}), Доступно/Работает: {GALOIS_AVAILABLE}")
logging.info(f"Ожид. Альфа для логирования: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS_EXTRACT or 'default'}) с батчингом.")
logging.info(f"DTCWT Бэкенд: Попытка использовать OpenCL (иначе NumPy).")
if USE_ECC and not GALOIS_AVAILABLE: logging.warning("ECC ожидается, но galois недоступна/не работает! Декодирование ECC невозможно.")
elif not USE_ECC: logging.info("ECC не ожидается.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error(f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) > CANDIDATE_POOL_SIZE ({CANDIDATE_POOL_SIZE})!")
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) != BITS_PER_PAIR ({BITS_PER_PAIR}).")

# --- Базовые Функции ---
def dct_1d(s: np.ndarray) -> np.ndarray:
    return dct(s, type=2, norm='ortho')

def dtcwt_transform(yp: np.ndarray, fn: int = -1) -> Optional[Pyramid]:
    """Применяет прямое DTCWT, используя текущий активный бэкенд dtcwt."""
    if not isinstance(yp, np.ndarray) or yp.ndim != 2:
        logging.error(f"[Frame:{fn}] Invalid input type/dims for dtcwt_transform: {type(yp)}, {yp.ndim if hasattr(yp, 'ndim') else 'N/A'}")
        return None
    if np.any(np.isnan(yp)):
        logging.warning(f"[Frame:{fn}] Input DTCWT contains NaN!")
        # Optional: Handle NaNs if necessary, e.g., by replacing or returning None earlier
        # yp = np.nan_to_num(yp)
    try:
        t = dtcwt.Transform2d()
        r, c = yp.shape
        pr = r % 2 != 0
        pc = c % 2 != 0
        ypp = np.pad(yp, ((0, pr), (0, pc)), mode='reflect') if pr or pc else yp
        py = t.forward(ypp.astype(np.float32), nlevels=1)
        if hasattr(py, 'lowpass') and py.lowpass is not None:
            py.padding_info = (pr, pc)
            return py
        else:
            logging.error(f"[Frame:{fn}] DTCWT forward did not return lowpass.")
            return None
    except Exception as e:
        logging.error(f"[Frame:{fn}] DTCWT fwd err ({dtcwt.backend_name}): {e}", exc_info=True)
        return None

@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    H, W = subband_shape
    if H < 2 or W < 2: return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2)
    min_dist, max_dist = 0.0, np.max(distances)
    # Handle cases where max_dist is very small or zero
    if max_dist < 1e-6:
         ring_bins = np.array([0.0, max_dist + 1e-6] * (n_rings + 1))[:n_rings + 1] # Create degenerate bins
    else:
         ring_bins = np.linspace(min_dist, max_dist + 1e-6, n_rings + 1)

    n_rings_eff = len(ring_bins)-1
    if n_rings_eff <= 0: return [None] * n_rings

    # Digitize and clip indices
    ring_indices = np.digitize(distances, ring_bins) - 1
    # Ensure the center point belongs to the first ring
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)

    rc: List[Optional[np.ndarray]] = [None] * n_rings
    for rdx in range(n_rings_eff):
        coords = np.argwhere(ring_indices == rdx)
        if coords.shape[0] > 0:
            rc[rdx] = coords
        # else: rc[rdx] remains None
    return rc

@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(ss: Tuple[int, int], nr: int) -> List[Optional[np.ndarray]]:
    return _ring_division_internal(ss, nr)

def ring_division(lp: np.ndarray, nr: int = N_RINGS, fn: int = -1) -> List[Optional[np.ndarray]]:
    """Разбивает 2D массив (lowpass подполосу) на N_RINGS концентрических колец."""
    if not isinstance(lp, np.ndarray) or lp.ndim != 2:
        logging.error(f"[Frame:{fn}] Invalid input for ring_division.")
        return [None] * nr

    sh: Tuple[int, int] = lp.shape
    try:
        cached_list = get_ring_coords_cached(sh, nr)
        if not isinstance(cached_list, list) or not all(isinstance(i, (np.ndarray, type(None))) for i in cached_list):
            logging.warning(f"[Frame:{fn}] Ring division cache returned invalid type, clearing and recalculating.")
            get_ring_coords_cached.cache_clear()
            cached_list = _ring_division_internal(sh, nr)

        # Return copies to prevent modification of cached arrays
        return [a.copy() if a is not None else None for a in cached_list]
    except Exception as e:
        logging.error(f"Ring division error Frame {fn}: {e}", exc_info=True)
        return [None] * nr

def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    eps=1e-12; shannon_entropy=0.; collision_entropy=0. # Use more descriptive names
    if rv.size > 0:
        # Ensure working with a copy and clip values to [0, 1] range
        rvc = np.clip(rv.copy(), 0.0, 1.0)
        # Check for constant array which leads to log2(0) -> NaN
        if np.all(rvc == rvc[0]):
             return 0.0, 0.0 # Entropy is 0 for constant data

        hist, _ = np.histogram(rvc, bins=256, range=(0., 1.), density=False)
        total_count = rvc.size # Use size for total count
        if total_count > 0:
            probabilities = hist / total_count
            # Filter out zero probabilities to avoid log2(0)
            p = probabilities[probabilities > eps]
            if p.size > 0:
                shannon_entropy = -np.sum(p * np.log2(p))
                # Collision entropy definition varies, using Renyi entropy of order 2 common interpretation
                collision_entropy = -np.log2(np.sum(p**2)) if np.sum(p**2) > eps else 0.0
                # Previous calculation seemed non-standard: ee=-np.sum(p*np.exp(1.-p))
    return shannon_entropy, collision_entropy

def compute_adaptive_alpha_entropy(rv: np.ndarray, ri: int, fn: int) -> float:
    """Computes adaptive alpha based on Shannon entropy and variance."""
    # Use the same name as in embedder for consistency
    if rv.size < 10: return ALPHA_MIN # Return min alpha for small rings
    shannon_entropy, _ = calculate_entropies(rv, fn, ri)
    local_variance = np.var(rv)
    # Normalize entropy (assuming MAX_THEORETICAL_ENTROPY=8.0 for 8-bit data)
    normalized_entropy = np.clip(shannon_entropy / MAX_THEORETICAL_ENTROPY, 0.0, 1.0)
    # Variance mapping (using sigmoid-like function)
    variance_threshold = 0.005
    variance_scale = 500
    variance_map = 1.0 / (1.0 + np.exp(-variance_scale * (local_variance - variance_threshold)))
    # Combine entropy and variance (example weights)
    weight_entropy = 0.6
    weight_variance = 0.4
    modulation_factor = np.clip((weight_entropy * normalized_entropy + weight_variance * variance_map), 0.0, 1.0)
    # Calculate adaptive alpha
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * modulation_factor
    logging.debug(f"[F:{fn}, R:{ri}] Extractor Alpha Calc (for consistency)={final_alpha:.4f} (E={shannon_entropy:.3f}, V={local_variance:.6f})")
    return np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX) # Clip to defined bounds

def get_fixed_pseudo_random_rings(pi: int, nr: int, ps: int) -> List[int]:
    """Generates a deterministic list of candidate ring indices based on pair index."""
    if ps <= 0: return []
    if ps > nr: ps = nr # Cannot select more rings than available

    seed_str = str(pi).encode('utf-8')
    hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big')
    prng = random.Random(seed_int) # Use a local PRNG instance
    try:
        # Sample without replacement
        candidate_indices = prng.sample(range(nr), ps)
    except ValueError: # If nr < ps (should not happen due to check above)
        candidate_indices = list(range(nr))
        prng.shuffle(candidate_indices) # Still shuffle if we take all
        candidate_indices = candidate_indices[:ps] # Ensure correct size

    logging.debug(f"[P:{pi}] Candidates: {candidate_indices}");
    return candidate_indices

def bits_to_bytes(bit_list: List[Optional[int]]) -> Optional[bytes]:
    """Converts a list of bits (0/1 or None) into bytes, ignoring None."""
    valid_bits = [b for b in bit_list if b is not None]
    num_bits = len(valid_bits)
    if num_bits == 0: return b'' # Return empty bytes if no valid bits
    # Pad with zeros at the end if length is not a multiple of 8
    remainder = num_bits % 8
    if remainder != 0:
        padding_len = 8 - remainder
        valid_bits.extend([0] * padding_len)
        logging.warning(f"Bit list length ({num_bits}) not multiple of 8. Padded with {padding_len} zeros.")
        num_bits += padding_len

    byte_array = bytearray()
    for i in range(0, num_bits, 8):
        byte_chunk = valid_bits[i:i+8]
        try:
            byte_val = int("".join(map(str, byte_chunk)), 2)
            byte_array.append(byte_val)
        except ValueError:
            logging.error(f"Invalid symbols found during bit-to-byte conversion: {byte_chunk}")
            return None # Indicate error
    return bytes(byte_array)

def decode_ecc(packet_bits_list: List[int], bch_code: galois.BCH, expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    """Декодирует пакет бит с использованием Galois BCH."""
    if not isinstance(packet_bits_list, list) or not all(isinstance(b, int) for b in packet_bits_list):
         logging.error(f"Decode ECC: Input must be a list of integers.")
         return None, -1

    packet_len_expected_n = bch_code.n
    packet_len_received = len(packet_bits_list)

    if packet_len_received != packet_len_expected_n:
        logging.error(f"Decode ECC: Неверная длина пакета ({packet_len_received} бит), ожидалось n={packet_len_expected_n}.")
        return None, -1 # Ошибка длины

    k = bch_code.k
    expected_payload_len_bits = expected_data_len_bytes * 8
    if expected_payload_len_bits > k:
        logging.error(f"Decode ECC: Ожидаемая длина payload ({expected_payload_len_bits} бит) > k ({k}) кода.")
        return None, -1 # Несоответствие параметров

    n_corrected_symbols = -1 # Default error value
    try:
        packet_bits_np = np.array(packet_bits_list, dtype=np.uint8)
        GF = bch_code.field
        received_vector = GF(packet_bits_np)

        try:
            # Попытка декодирования
            corrected_message_vector, n_corrected_symbols = bch_code.decode(received_vector, errors=True)
            # ИСПРАВЛЕНО: Используем n_corrected_symbols в логе
            logging.info(f"Galois ECC: Декодировано, исправлено {n_corrected_symbols} ошибок.")

        except galois.errors.UncorrectableError:
            logging.warning("Galois ECC: Слишком много ошибок, не удалось декодировать пакет.")
            return None, -1 # Пакет неисправим (возвращаем -1 для ошибок)

        # Извлечение payload
        corrected_k_bits_np = corrected_message_vector.view(np.ndarray).astype(np.uint8)
        if corrected_k_bits_np.size < expected_payload_len_bits:
            logging.error(f"Decode ECC: Длина декодированного сообщения ({corrected_k_bits_np.size}) < ожидаемой ({expected_payload_len_bits}).")
            return None, n_corrected_symbols # Возвращаем кол-во исправленных, но payload None

        corrected_payload_bits_np = corrected_k_bits_np[:expected_payload_len_bits]
        logging.debug(f"Decode ECC: Extracted {len(corrected_payload_bits_np)} payload bits.")

        corrected_payload_bytes = bits_to_bytes(corrected_payload_bits_np.tolist()) # bits_to_bytes handles padding

        if corrected_payload_bytes is None:
            logging.error("Decode ECC: Ошибка конвертации бит payload в байты.")
            return None, n_corrected_symbols # Payload None, но ошибки исправлены (если были)
        # Note: bits_to_bytes pads, so length check might not be needed here if padding is acceptable
        # if len(corrected_payload_bytes) != expected_data_len_bytes:
        #      logging.error(f"Decode ECC: Неверная финальная длина payload ({len(corrected_payload_bytes)} байт), ожидалось {expected_data_len_bytes}.")
        #      return None, n_corrected_symbols

        return corrected_payload_bytes, n_corrected_symbols # Успех

    except Exception as e:
        logging.error(f"Decode ECC: Неожиданная ошибка: {e}", exc_info=True)
        return None, -1 # Общая ошибка


def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # Используем исправленную версию из embedder
    logging.info(f"Reading: {video_path}"); frames: List[np.ndarray] = []; fps = float(FPS); cap = None; h, w = -1, -1;
    try:
        assert os.path.exists(video_path), f"Not found: {video_path}";
        cap = cv2.VideoCapture(video_path); assert cap.isOpened(), f"Cannot open {video_path}";
        fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); logging.info(f"Props: {w}x{h}@{fps:.2f}~{fc}f"); rc,nc,ic=0,0,0;
        while True:
            ret,f=cap.read();
            if not ret: break
            if f is None: nc+=1; continue
            if f.ndim==3 and f.shape[:2]==(h,w) and f.dtype==np.uint8: frames.append(f); rc+=1;
            else: ic+=1; logging.warning(f"Skipped invalid frame #{rc+nc+ic}");
        logging.info(f"Read loop finished. V:{rc},N:{nc},I:{ic}"); assert rc>0, "No valid frames read."
    except Exception as e: logging.error(f"Read error: {e}", exc_info=True); frames=[]
    finally:
        if cap and cap.isOpened(): logging.debug("Releasing capture"); cap.release()
    return frames, fps

# --- Функция извлечения ОДНОГО бита из ОДНОГО кольца ---
# АКТИВИРУЕМ ВЕРСИЮ "МТ" (медленная, но точная по вашим наблюдениям)
def extract_single_bit(f1:np.ndarray, f2:np.ndarray, ring_idx:int, n_rings:int, fn:int) -> Optional[int]:
    """Извлекает один бит из заданного кольца пары кадров. (ВЕРСИЯ С ВНУТРЕННИМ DTCWT)"""
    pair_index = fn // 2
    prefix = f"[МТ P:{pair_index}, R:{ring_idx}]"

    try:
        # --- Шаг 1: Проверка входных данных ---
        if f1 is None or f2 is None or f1.shape != f2.shape:
            logging.warning(f"{prefix} Invalid input frames (None or shape mismatch).")
            return None

        # --- Шаг 2: Преобразование цвета и DTCWT (ВНУТРИ ФУНКЦИИ) ---
        try:
            # Используем ГЛОБАЛЬНУЮ константу EMBED_COMPONENT
            y1 = cv2.cvtColor(f1, cv2.COLOR_BGR2YCrCb)
            y2 = cv2.cvtColor(f2, cv2.COLOR_BGR2YCrCb)
            c1 = y1[:, :, EMBED_COMPONENT].astype(np.float32) / 255.0
            c2 = y2[:, :, EMBED_COMPONENT].astype(np.float32) / 255.0
        except cv2.error as e_conv:
            logging.warning(f"{prefix} YCrCb/Comp extract error: {e_conv}")
            return None

        p1 = dtcwt_transform(c1, fn)
        p2 = dtcwt_transform(c2, fn + 1)
        if p1 is None or p2 is None or p1.lowpass is None or p2.lowpass is None:
            logging.warning(f"{prefix} DTCWT failed for one or both components.")
            return None
        # Конвертируем в numpy массив, если это результат OpenCL
        L1 = np.array(p1.lowpass)
        L2 = np.array(p2.lowpass)
        # logging.debug(f"{prefix} DTCWT L1/L2 Calculated.")
        # ИСПРАВЛЕНО: Форматирование для комплексных чисел (показываем магнитуду)
        logging.debug(f"{prefix} L1(in) s={L1.shape} m={np.mean(np.abs(L1)):.8e} v={np.var(np.abs(L1)):.8e} L1[0,0]={np.abs(L1[0,0]):.8e}")
        logging.debug(f"{prefix} L2(in) s={L2.shape} m={np.mean(np.abs(L2)):.8e} v={np.var(np.abs(L2)):.8e} L2[0,0]={np.abs(L2[0,0]):.8e}")


        # --- Шаг 3: Кольцевое деление ---
        # Применяем к L1 и L2, полученным на предыдущем шаге
        r1c = ring_division(np.abs(L1), n_rings, fn)  # Используем магнитуду для деления
        r2c = ring_division(np.abs(L2), n_rings, fn + 1) # Используем магнитуду для деления
        if not(0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
             logging.warning(f"{prefix} Invalid ring index {ring_idx} for n_rings={n_rings}.")
             return None
        cd1 = r1c[ring_idx]
        cd2 = r2c[ring_idx]
        if cd1 is None or cd2 is None:
             logging.debug(f"{prefix} Ring coords are None for index {ring_idx}.")
             return None # Кольцо пустое

        # --- Шаг 4: Извлечение значений, DCT, SVD ---
        try:
            # Получаем КОМПЛЕКСНЫЕ значения из L1, L2 по координатам
            rs1, cs1 = cd1[:, 0], cd1[:, 1]
            rv1_complex = L1[rs1, cs1].astype(np.complex64) # Работаем с комплексными
            rs2, cs2 = cd2[:, 0], cd2[:, 1]
            rv2_complex = L2[rs2, cs2].astype(np.complex64)
        except IndexError:
             logging.warning(f"{prefix} Index error getting ring values.")
             return None

        if rv1_complex.size == 0 or rv2_complex.size == 0:
             logging.debug(f"{prefix} Ring values empty.")
             return None

        # Синхронизация размера
        min_s = min(rv1_complex.size, rv2_complex.size)
        if rv1_complex.size != rv2_complex.size:
            rv1_complex = rv1_complex[:min_s]
            rv2_complex = rv2_complex[:min_s]
        if min_s == 0: return None

        # DCT от МАГНИТУДЫ (как скорее всего было в эмбеддере при вычислении alpha?)
        # ИЛИ DCT от комплексных чисел? Если от комплексных, SVD будет другим.
        # Пробуем от магнитуды, как наиболее вероятный вариант, согласующийся с SVD на действительных числах далее
        rv1 = np.abs(rv1_complex).astype(np.float32)
        rv2 = np.abs(rv2_complex).astype(np.float32)

        # logging.debug(f"{prefix} rv1 s={rv1.shape} m={np.mean(rv1):.8f} data[:3]={np.array2string(rv1[:3], precision=8)}")
        # logging.debug(f"{prefix} rv2 s={rv2.shape} m={np.mean(rv2):.8f} data[:3]={np.array2string(rv2[:3], precision=8)}")

        d1 = dct_1d(rv1); d2 = dct_1d(rv2)
        # logging.debug(f"{prefix} d1[:3]={np.array2string(d1[:3], precision=8)}")
        # logging.debug(f"{prefix} d2[:3]={np.array2string(d2[:3], precision=8)}")

        try:
            # SVD ожидает действительные числа
            S1 = svd(d1.reshape(-1, 1), compute_uv=False)
            S2 = svd(d2.reshape(-1, 1), compute_uv=False)
        except np.linalg.LinAlgError:
            logging.warning(f"{prefix} SVD failed.")
            return None

        s1 = S1[0] if S1.size > 0 else 0.0
        s2 = S2[0] if S2.size > 0 else 0.0
        logging.debug(f"{prefix} s1={s1:.12e}, s2={s2:.12e}")

        # --- Шаг 5: Принятие решения ---
        eps = 1e-12
        threshold = 1.0
        ratio = s1 / (s2 + eps)
        extracted_bit = 0 if ratio >= threshold else 1
        logging.debug(f"{prefix} ratio={ratio:.12e} -> Bit={extracted_bit}")

        return extracted_bit
    except Exception as e:
        logging.error(f"Extract single bit MT failed (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
        return None

# --- ЗАКОММЕНТИРОВАНА ВЕРСИЯ "БН" (Быстрая Неточная) ---
# def extract_single_bit(L1: np.ndarray, L2: np.ndarray, ring_idx:int, n_rings:int, fn:int=0) -> Optional[int]:
#     """
#     Извлекает один бит из заданного кольца, используя ПРЕДВАРИТЕЛЬНО ВЫЧИСЛЕННЫЕ
#     lowpass компоненты L1 и L2. (БН - Быстрая Неточная)
#     """
#     pair_index = fn // 2
#     prefix = f"[БН P:{pair_index}, R:{ring_idx}]"
#     try:
#         # ИСПРАВЛЕНО: Форматирование комплексных чисел
#         logging.debug(f"{prefix} L1(in) s={L1.shape} m={np.mean(np.abs(L1)):.8e} v={np.var(np.abs(L1)):.8e} L1[0,0]={np.abs(L1[0,0]):.8e}")
#         logging.debug(f"{prefix} L2(in) s={L2.shape} m={np.mean(np.abs(L2)):.8e} v={np.var(np.abs(L2)):.8e} L2[0,0]={np.abs(L2[0,0]):.8e}")
#
#         if not isinstance(L1, np.ndarray) or not isinstance(L2, np.ndarray) or L1.shape != L2.shape:
#             logging.warning(f"{prefix} Invalid L1/L2 provided.")
#             return None
#
#         # Используем магнитуду для деления на кольца
#         r1c = ring_division(np.abs(L1), n_rings, fn)
#         r2c = ring_division(np.abs(L2), n_rings, fn + 1)
#
#         if not(0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
#              logging.warning(f"{prefix} Invalid ring index.")
#              return None
#         cd1 = r1c[ring_idx]; cd2 = r2c[ring_idx]
#         if cd1 is None or cd2 is None:
#              logging.debug(f"{prefix} Ring coords are None.")
#              return None
#
#         try:
#             rs1, cs1 = cd1[:, 0], cd1[:, 1]; rv1_complex = L1[rs1, cs1].astype(np.complex64)
#             rs2, cs2 = cd2[:, 0], cd2[:, 1]; rv2_complex = L2[rs2, cs2].astype(np.complex64)
#             logging.debug(f"{prefix} rv1_complex[:5]: {np.array2string(rv1_complex[:5], precision=6)}")
#         except IndexError:
#             logging.warning(f"{prefix} Index error getting ring values.")
#             return None
#
#         if rv1_complex.size == 0 or rv2_complex.size == 0:
#              logging.debug(f"{prefix} Ring values empty.")
#              return None
#
#         min_s = min(rv1_complex.size, rv2_complex.size)
#         if rv1_complex.size != rv2_complex.size:
#             rv1_complex = rv1_complex[:min_s]; rv2_complex = rv2_complex[:min_s]
#             if min_s == 0: return None
#
#         # DCT от магнитуды
#         rv1 = np.abs(rv1_complex).astype(np.float32)
#         rv2 = np.abs(rv2_complex).astype(np.float32)
#
#         d1 = dct_1d(rv1); d2 = dct_1d(rv2)
#         try:
#             S1 = svd(d1.reshape(-1, 1), compute_uv=False)
#             S2 = svd(d2.reshape(-1, 1), compute_uv=False)
#         except np.linalg.LinAlgError:
#             logging.warning(f"{prefix} SVD failed.")
#             return None
#
#         s1 = S1[0] if S1.size > 0 else 0.0
#         s2 = S2[0] if S2.size > 0 else 0.0
#
#         eps = 1e-12; threshold = 1.0
#         ratio = s1 / (s2 + eps)
#         extracted_bit = 0 if ratio >= threshold else 1
#         logging.debug(f"{prefix} s1={s1:.6f}, s2={s2:.6f}, ratio={ratio:.6f} -> Bit={extracted_bit}")
#
#         return extracted_bit
#     except Exception as e:
#         logging.error(f"Extract single bit BN failed (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
#         return None


# --- Воркер для ОДНОЙ ПАРЫ (ВЕРСИЯ С ВЫБОРОМ ПО ЭНТРОПИИ) ---
# @profile # Профилирование можно включить, если line_profiler установлен
def _extract_single_pair_task(args: Dict[str, Any]) -> Tuple[int, List[Optional[int]]]:
    """
    Обрабатывает одну пару: выбирает кольца на основе энтропии из пула кандидатов
    и вызывает extract_single_bit для каждого из них.
    """
    pair_idx = args['pair_idx']
    f1 = args['frame1'] # Используем первый кадр для выбора колец
    f2 = args['frame2']
    nr = args['n_rings']
    nrtu = args['num_rings_to_use']
    cps = args['candidate_pool_size']
    ec = args['embed_component']
    fn = 2 * pair_idx
    selected_rings = [] # Инициализация
    extracted_bits: List[Optional[int]] = [None] * nrtu # Инициализация результата

    try:
        # --- ШАГ 1: Получение детерминированного пула кандидатов ---
        candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
        if len(candidate_rings) < nrtu:
             raise ValueError(f"Not enough candidates {len(candidate_rings)}<{nrtu} for pair {pair_idx}")

        # --- ШАГ 2: Адаптивный выбор из пула по энтропии ---
        comp1_sel = f1[:, :, ec].astype(np.float32) / 255.0
        pyr1 = dtcwt_transform(comp1_sel, fn)
        if pyr1 is None or pyr1.lowpass is None:
            raise RuntimeError(f"DTCWT L1 failed for selection in pair {pair_idx}")

        L1s = pyr1.lowpass
        if not isinstance(L1s, np.ndarray): L1s = np.array(L1s)

        # Используем магнитуду для деления на кольца и расчета энтропии/вар
        L1s_abs = np.abs(L1s)
        coords = ring_division(L1s_abs, nr, fn)
        if coords is None or len(coords) != nr:
             raise RuntimeError(f"Ring division failed for L1s_abs in pair {pair_idx}")

        entropies = []
        min_pixels_for_entropy = 10
        for r_idx in candidate_rings:
            entropy_val = -float('inf')
            if 0 <= r_idx < len(coords) and coords[r_idx] is not None:
                 c = coords[r_idx]
                 if c.shape[0] >= min_pixels_for_entropy:
                      try:
                           rs, cs = c[:, 0], c[:, 1]
                           rv = L1s_abs[rs, cs] # Используем магнитуду
                           shannon_entropy, _ = calculate_entropies(rv, fn, r_idx)
                           if np.isfinite(shannon_entropy):
                                entropy_val = shannon_entropy
                      except IndexError: logging.warning(f"IndexError during entropy calculation P:{pair_idx} R:{r_idx}")
                      except Exception as entropy_e: logging.warning(f"Entropy calc error P:{pair_idx} R:{r_idx}: {entropy_e}")
            entropies.append((entropy_val, r_idx))

        entropies.sort(key=lambda x: x[0], reverse=True)
        selected_rings = [idx for e, idx in entropies if e > -float('inf')][:nrtu]

        if len(selected_rings) < nrtu:
            logging.warning(f"[P:{pair_idx}] Not enough rings with valid entropy ({len(selected_rings)}<{nrtu}). Falling back for extraction.")
            deterministic_fallback = candidate_rings[:nrtu]
            for ring in deterministic_fallback:
                if ring not in selected_rings: selected_rings.append(ring)
                if len(selected_rings) == nrtu: break
            if len(selected_rings) < nrtu: raise RuntimeError(f"Fallback failed for extraction, still not enough rings for pair {pair_idx}")
            logging.warning(f"[P:{pair_idx}] Selected rings for extraction after fallback: {selected_rings}")
        else:
             logging.info(f"[P:{pair_idx}] Selected rings for extraction (Entropy based): {selected_rings}")


        # --- ШАГ 3: Извлечение бит из выбранных колец ---
        # ИСПРАВЛЕНО: Вызываем extract_single_bit БЕЗ аргумента 'ec'
        for i, ring_idx_to_extract in enumerate(selected_rings):
             if 0 <= i < nrtu:
                  # ВЫЗОВ ИСПРАВЛЕН - 5 аргументов
                  extracted_bits[i] = extract_single_bit(f1, f2, ring_idx_to_extract, nr, fn)
             else:
                 logging.warning(f"[P:{pair_idx}] Index 'i' ({i}) out of range for extracted_bits (size {nrtu}) during extraction loop")

        return pair_idx, extracted_bits

    except Exception as e:
        logging.error(f"Error in single pair task (Extract - Entropy Sel.) P:{pair_idx}: {e}", exc_info=True)
        return pair_idx, [None] * nrtu


# --- Воркер для обработки БАТЧА задач (ThreadPoolExecutor) ---
def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
    """Обрабатывает батч задач извлечения последовательно внутри одного ПОТОКА."""
    batch_results: Dict[int, List[Optional[int]]] = {}
    for args in batch_args_list:
        pair_idx, bits = _extract_single_pair_task(args)
        batch_results[pair_idx] = bits
    return batch_results


# --- Основная функция извлечения (ThreadPool + Batches) ---
@profile
def extract_watermark_from_video(
        frames:List[np.ndarray],
        nr:int=N_RINGS,
        nrtu:int=NUM_RINGS_TO_USE,
        bp:int=BITS_PER_PAIR,
        cps:int=CANDIDATE_POOL_SIZE,
        ec:int=EMBED_COMPONENT,
        # --- Параметры для гибридного извлечения ---
        expect_hybrid_ecc: bool = True,      # Ожидать ли гибридный формат?
        max_expected_packets: int = 15,      # Макс. пакетов для попытки извлечения
        # --- Параметры ECC (для первого пакета и проверки) ---
        ue:bool=USE_ECC, # Ожидается ли ECC в принципе (для первого пакета)
        bch_code:Optional[galois.BCH]=BCH_CODE_OBJECT, # Глобальный объект BCH
        # --- Остальные параметры ---
        plb:int=PAYLOAD_LEN_BYTES,
        mw:Optional[int]=MAX_WORKERS_EXTRACT
    ) -> Optional[bytes]:
    """
    Основная функция, управляющая процессом извлечения с использованием ThreadPoolExecutor,
    батчинга и ПОБИТОВОГО мажоритарного голосования, с поддержкой гибридного ECC/Raw формата.
    """
    logging.info(f"--- Starting Extraction ---")
    logging.info(f"Mode: Hybrid Expected={expect_hybrid_ecc}, Max Packets={max_expected_packets}")
    logging.info(f"ECC Config: Used if Possible={ue}, BCH Object Present={bch_code is not None}")
    start_time = time.time()
    nf = len(frames)
    total_pairs_available = nf // 2

    if total_pairs_available == 0:
        logging.error("No frame pairs to process.")
        return None

    # Определяем длины пакетов и возможность ECC
    payload_len_bits = plb * 8
    packet_len_if_ecc = payload_len_bits # Длина по умолчанию
    packet_len_if_raw = payload_len_bits # Всегда длина raw payload
    ecc_possible_for_first = False
    bch_n = 0 # Длина ECC пакета (для расчетов)

    if ue and GALOIS_AVAILABLE and bch_code is not None:
        try:
            n = bch_code.n; k = bch_code.k; t_bch = bch_code.t
            if payload_len_bits <= k:
                packet_len_if_ecc = n; bch_n = n # Сохраняем n
                ecc_possible_for_first = True
                logging.info(f"ECC check: Possible for first packet (n={n}, k={k}, t={t_bch}).")
            else:
                logging.warning(f"ECC check: Payload size ({payload_len_bits}) > Galois k ({k}). ECC decoding impossible.")
        except Exception as e:
            logging.error(f"ECC check: Error getting Galois params: {e}.")
            ecc_possible_for_first = False
    else:
        logging.info("ECC check: Disabled or unavailable.")

    # --- Вычисляем, сколько пар кадров нужно обработать ---
    # Оценка сверху: самый длинный сценарий - 1 ECC + (max-1) Raw
    max_possible_bits = 0
    if expect_hybrid_ecc and ecc_possible_for_first:
        max_possible_bits = packet_len_if_ecc + max(0, max_expected_packets - 1) * packet_len_if_raw
    elif ecc_possible_for_first and not expect_hybrid_ecc: # Ожидаем только ECC
        max_possible_bits = max_expected_packets * packet_len_if_ecc
    else: # Ожидаем только Raw
        max_possible_bits = max_expected_packets * packet_len_if_raw

    pairs_needed = ceil(max_possible_bits / bp) if bp > 0 else 0
    pairs_to_process = min(total_pairs_available, pairs_needed)
    logging.info(f"Target extraction: Up to {max_expected_packets} packets.")
    logging.info(f"Frame pairs: Available={total_pairs_available}, Needed(max)={pairs_needed}, To Process={pairs_to_process}")

    if pairs_to_process == 0:
        logging.warning("Zero pairs to process based on calculations.")
        return None

    # --- Подготовка и запуск батчей для извлечения сырых бит ---
    # Этот блок НЕ МЕНЯЕТСЯ. Он просто собирает биты.
    all_pairs_args = []
    skipped_pairs = 0
    for pair_idx in range(pairs_to_process):
        i1 = 2 * pair_idx; i2 = i1 + 1
        if i2 >= nf or frames[i1] is None or frames[i2] is None: skipped_pairs += 1; continue
        args = {'pair_idx': pair_idx, 'frame1': frames[i1], 'frame2': frames[i2],
                'n_rings': nr, 'num_rings_to_use': nrtu, 'candidate_pool_size': cps,
                'embed_component': ec}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0: logging.warning(f"Skipped {skipped_pairs} pairs during task preparation.")
    if num_valid_tasks == 0: logging.error("No valid extraction tasks generated."); return None

    num_workers = mw if mw is not None and mw > 0 else (os.cpu_count() or 1)
    ideal_batch_size = ceil(num_valid_tasks / (num_workers * 2))
    batch_size = max(1, min(ideal_batch_size, 100))
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i : i + batch_size] for i in range(0, num_valid_tasks, batch_size)]
    batched_args_list = [batch for batch in batched_args_list if batch]

    logging.info(f"Launching {len(batched_args_list)} batches ({num_valid_tasks} pairs) using ThreadPool (mw={num_workers}, batch_size≈{batch_size})...")

    extracted_bits_map: Dict[int, List[Optional[int]]] = {}
    ppc = 0; fpe = 0 # Счетчики обработанных пар и пар с ошибками
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch_idx = {executor.submit(_extract_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                try:
                    batch_results_map = future.result()
                    extracted_bits_map.update(batch_results_map)
                    ppc += len(batch_results_map)
                    fpe += sum(1 for bits in batch_results_map.values() if bits is None or None in bits)
                except Exception as e:
                     batch_size_failed = len(batched_args_list[batch_idx])
                     logging.error(f"Batch {batch_idx} (size {batch_size_failed}) execution failed: {e}", exc_info=True)
                     fpe += batch_size_failed
    except Exception as e:
        logging.critical(f"ThreadPoolExecutor critical error during extraction: {e}", exc_info=True)
        return None

    logging.info(f"Extraction task processing finished. Processed pairs confirmed: {ppc}. Pairs with errors/Nones: {fpe}.")
    if ppc == 0: logging.error("No pairs processed successfully by workers."); return None

    # --- Сборка общего потока извлеченных бит ---
    # Этот блок НЕ МЕНЯЕТСЯ.
    extracted_bits_all: List[Optional[int]] = []
    for pair_idx in range(pairs_to_process):
        bits = extracted_bits_map.get(pair_idx)
        if bits is not None:
            if len(bits) == bp: extracted_bits_all.extend(bits)
            else:
                 logging.warning(f"Pair {pair_idx} returned incorrect bits ({len(bits)} != {bp}). Adding Nones.")
                 extracted_bits_all.extend([None] * bp); #fpe += 1 # Ошибку уже посчитали выше
        else:
            # Если для пары нет результата (была пропущена или батч упал)
            extracted_bits_all.extend([None] * bp)

    total_bits_collected = len(extracted_bits_all)
    valid_bits = [b for b in extracted_bits_all if b is not None]
    num_valid_bits = len(valid_bits)
    num_none_bits = total_bits_collected - num_valid_bits
    success_rate = (1 - num_none_bits / total_bits_collected) * 100 if total_bits_collected > 0 else 0
    logging.info(f"Bit collection: Total collected={total_bits_collected}, Valid (non-None)={num_valid_bits} ({success_rate:.1f}%), None/Error={num_none_bits}.")
    if not valid_bits: logging.error("No valid (non-None) bits extracted."); return None

    # --- ИЗМЕНЕНИЕ: Гибридное Декодирование Пакетов ---
    all_payload_attempts_bits: List[Optional[List[int]]] = [] # Список для хранения результатов (64 бита или None)
    decoded_success_count = 0; decode_failed_count = 0; total_corrected_symbols = 0
    num_processed_bits = 0 # Счетчик обработанных бит из valid_bits

    print("\n--- Попытки Декодирования Пакетов ---")
    print(f"{'Pkt #':<6} | {'Type':<7} | {'ECC Status':<18} | {'Corrected':<10} | {'Payload (Hex)':<20}")
    print("-" * 68) # Увеличил ширину

    for i in range(max_expected_packets): # Итерируем до макс. числа пакетов
        # --- Определяем тип и длину ОЖИДАЕМОГО пакета ---
        is_first_packet = (i == 0)
        # Используем ECC для первого пакета ТОЛЬКО если включен гибридный режим И ECC возможен
        use_ecc_for_this = is_first_packet and expect_hybrid_ecc and ecc_possible_for_first

        current_packet_len = packet_len_if_ecc if use_ecc_for_this else packet_len_if_raw
        packet_type_str = "ECC" if use_ecc_for_this else "Raw"

        # --- Проверяем, хватает ли бит ---
        start_idx = num_processed_bits
        end_idx = start_idx + current_packet_len

        if end_idx > num_valid_bits:
            logging.warning(f"Not enough valid bits remaining for potential packet {i + 1} (type {packet_type_str}, needed {current_packet_len}, have {num_valid_bits - start_idx}). Stopping decode.")
            break # Заканчиваем, если бит не хватает

        packet_candidate_bits = valid_bits[start_idx:end_idx]
        num_processed_bits += current_packet_len # Сдвигаем указатель ВНЕ зависимости от успеха декодирования

        # --- Пытаемся получить payload ---
        payload_bytes: Optional[bytes] = None
        payload_bits: Optional[List[int]] = None
        errors: int = -1 # По умолчанию - ошибка
        status_str = f"Failed ({packet_type_str})" # Статус по умолчанию

        if use_ecc_for_this:
            # Пытаемся декодировать первый пакет с ECC
            if bch_code is not None: # Доп. проверка
                payload_bytes, errors = decode_ecc(packet_candidate_bits, bch_code, plb)
                if payload_bytes is not None:
                    # Используем n_corrected_symbols (errors) для статуса
                    corrected_count = errors if errors != -1 else 0 # Считаем 0 если decode вернул payload но errors=-1
                    status_str = f"OK (ECC: {corrected_count} fixed)"
                    if errors > 0: total_corrected_symbols += errors
                else:
                    # errors тут будет -1 если неисправимо
                    status_str = f"Uncorrectable (ECC)" if errors == -1 else "ECC Decode Error"
                    decode_failed_count += 1
            else:
                 # Сюда не должны попадать из-за ecc_possible_for_first, но на всякий случай
                 status_str = "ECC Code Missing"
                 decode_failed_count += 1
        else:
            # Обрабатываем как Raw пакет
            if len(packet_candidate_bits) >= payload_len_bits:
                 # Берем только нужное кол-во бит для payload
                 payload_candidate_bits_raw = packet_candidate_bits[:payload_len_bits]
                 packet_bytes_raw = bits_to_bytes(payload_candidate_bits_raw)
                 # bits_to_bytes вернет None при ошибке конвертации
                 if packet_bytes_raw is not None and len(packet_bytes_raw) == plb:
                     payload_bytes = packet_bytes_raw
                     errors = 0 # Нет ECC -> 0 ошибок для статуса
                     status_str = "OK (Raw)"
                 else:
                      status_str = "Failed (Raw Convert)"
                      decode_failed_count += 1
            else: # Не хватило бит даже для raw payload - этого не должно быть из-за проверки выше
                  status_str = "Failed (Raw Short)"
                  decode_failed_count += 1

        # --- Получаем биты из байт (если удалось) ---
        payload_hex_str = "N/A"
        if payload_bytes is not None: # Проверяем результат предыдущего шага
            payload_hex_str = payload_bytes.hex() # Можем показать hex, даже если unpack не удастся
            try:
                payload_np_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
                if len(payload_np_bits) == payload_len_bits:
                    payload_bits = payload_np_bits.tolist() # Успех!
                    decoded_success_count += 1
                else:
                     # Ошибка длины после unpackbits
                     logging.warning(f"Packet {i+1} unpacked to len {len(payload_np_bits)} != {payload_len_bits}.")
                     status_str += "[Len Fail]"
                     if status_str.startswith("OK"): decode_failed_count += 1 # Считаем ошибкой, если статус был OK
                     payload_bits = None # Сбрасываем результат
            except Exception as e_unpack:
                logging.error(f"Error unpacking bits for packet {i+1}: {e_unpack}")
                status_str += "[Unpack Fail]"
                if status_str.startswith("OK"): decode_failed_count += 1
                payload_bits = None
        # else: # Если payload_bytes был None, decode_failed_count уже увеличен

        all_payload_attempts_bits.append(payload_bits) # Добавляем результат (или None)
        corrected_str = str(errors) if errors != -1 else "-"
        print(f"{i+1:<6} | {packet_type_str:<7} | {status_str:<18} | {corrected_str:<10} | {payload_hex_str:<20}")

    # --- Конец цикла декодирования ---

    print("-" * 68) # Увеличил ширину
    logging.info(f"Decode attempts summary: Total packets processed = {len(all_payload_attempts_bits)}, Success (yielded {payload_len_bits} bits) = {decoded_success_count}, Failed/Skipped = {decode_failed_count}.")
    if ecc_possible_for_first and expect_hybrid_ecc:
         logging.info(f"Total ECC corrections reported for first packet: {total_corrected_symbols}.")


    # --- ИЗМЕНЕНИЕ: Побитовое Голосование с Приоритетом Первого Пакета ---
    num_attempted_packets = len(all_payload_attempts_bits)
    if num_attempted_packets == 0:
        logging.error("No packets were processed for voting.")
        return None

    # Получаем результат первого пакета для разрешения ничьих
    first_packet_payload = all_payload_attempts_bits[0] if num_attempted_packets > 0 else None

    # Фильтруем валидные пакеты *только для самого голосования*
    valid_decoded_payloads = [p for p in all_payload_attempts_bits if p is not None]
    num_valid_packets_for_vote = len(valid_decoded_payloads)

    if num_valid_packets_for_vote == 0:
        logging.error(f"No valid {payload_len_bits}-bit payloads available for bit-wise voting.")
        return None

    final_payload_bits = []
    logging.info(f"Performing bit-wise majority vote across {num_valid_packets_for_vote} validly decoded packets (tie-break to first packet if valid)...")

    print("\n--- Bit-wise Voting Details ---")
    print(f"{'Bit Pos':<8} | {'Votes 0':<8} | {'Votes 1':<8} | {'Winner':<8} | {'Tiebreak?':<10}")
    print("-" * 50)

    for j in range(payload_len_bits): # Итерация по позициям бит
        votes_for_0 = 0; votes_for_1 = 0

        for i in range(num_valid_packets_for_vote): # Итерация по ВАЛИДНЫМ пакетам
            # Не нужна проверка на None здесь, так как valid_decoded_payloads их не содержит
            if j < len(valid_decoded_payloads[i]):
                 if valid_decoded_payloads[i][j] == 1: votes_for_1 += 1
                 else: votes_for_0 += 1
            else: # На всякий случай
                 logging.warning(f"Bit index {j} out of range for valid packet {i} during voting.")

        # Определение победителя
        winner_bit: Optional[int] = None # Используем Optional
        tiebreak_used = "No"

        if votes_for_1 > votes_for_0: winner_bit = 1
        elif votes_for_0 > votes_for_1: winner_bit = 0
        else: # Ничья
            tiebreak_used = "Yes"
            logging.warning(f"Bit position {j}: Tie in voting ({votes_for_0} vs {votes_for_1}). Trying bit from the first packet.")
            # Берем бит из первого пакета, только ЕСЛИ ОН БЫЛ ВАЛИДНЫМ
            if first_packet_payload is not None and j < len(first_packet_payload):
                 winner_bit = first_packet_payload[j]
            else:
                 logging.error(f"Cannot resolve tie for bit {j}: First packet decoding failed or index invalid! Voting failed.")
                 print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {'FAIL':<8} | {tiebreak_used:<10}")
                 # Если не можем разрешить ничью, голосование не удалось
                 final_payload_bits = None # Сигнализируем об общей ошибке
                 break # Прерываем цикл голосования

        # Если не удалось определить победителя (хотя этого не должно быть при текущей логике, кроме FAIL выше)
        if winner_bit is None:
             logging.error(f"Winner bit is None for position {j}, this shouldn't happen.")
             final_payload_bits = None
             break

        final_payload_bits.append(winner_bit)
        print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {winner_bit:<8} | {tiebreak_used:<10}")

    # --- Конец цикла голосования ---

    print("-" * 50)

    # Проверяем, не прервалось ли голосование
    if final_payload_bits is None:
         logging.error("Bit-wise voting failed due to unresolvable tie or other error.")
         return None
    else:
         logging.info(f"Bit-wise voting complete.")

    # --- Конвертация и возврат результата (остается без изменений) ---
    final_payload_bytes = bits_to_bytes(final_payload_bits)
    if final_payload_bytes is None:
         logging.error("Failed to convert final voted bits to bytes.")
         return None
    if len(final_payload_bytes) != plb:
         logging.error(f"Final payload length after voting ({len(final_payload_bytes)}B) != expected ({plb}B).")
         # ... (обработка ошибки длины, как раньше) ...
         return None

    logging.info(f"Final ID after bit-wise voting: {final_payload_bytes.hex()}")
    end_time = time.time()
    logging.info(f"Extraction done. Total time: {end_time - start_time:.2f} sec.")
    return final_payload_bytes

# --- Основная Функция (main) ---
def main():
    global BCH_CODE_OBJECT, DTCWT_OPENCL_ENABLED

    start_time_main = time.time()
    backend_name_str = 'opencl' if DTCWT_OPENCL_ENABLED else 'numpy'
    logging.info(f"Selected DTCWT backend: {backend_name_str.upper()}") # Log selected backend

    # Имя входного файла должно соответствовать выходному имени эмбеддера
    # Ensure consistency or make it an argument
    input_base = f"compressed_h264_crf28"
    input_video = input_base + INPUT_EXTENSION
    original_id = None

    # Load original ID for comparison
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r") as f:
                original_id = f.read().strip()
            # Validate loaded ID
            assert original_id and len(original_id) == PAYLOAD_LEN_BYTES * 2, "Invalid original ID format"
            int(original_id, 16) # Check if it's valid hex
            logging.info(f"Original ID loaded: {original_id}")
        except (IOError, AssertionError, ValueError) as e:
            logging.error(f"Read/Validate original ID failed: {e}")
            original_id = None
    else:
        logging.warning(f"'{ORIGINAL_WATERMARK_FILE}' not found. Cannot compare results.")

    logging.info(f"--- Starting Extraction Main Process (ThreadPool + Batches + {backend_name_str.upper()} DTCWT) ---")
    if not os.path.exists(input_video):
        logging.critical(f"Input video missing: '{input_video}'.")
        print(f"ERROR: Input video '{input_video}' not found.")
        return # Exit if input is missing

    frames, fps_read = read_video(input_video)
    if not frames:
        logging.critical("Failed to read video frames.")
        return # Exit if video reading failed
    logging.info(f"Read {len(frames)} frames from video.")

    # Вызов основной функции извлечения
    extracted_bytes = extract_watermark_from_video(
        frames=frames,
        nr=N_RINGS, nrtu=NUM_RINGS_TO_USE, bp=BITS_PER_PAIR,
        cps=CANDIDATE_POOL_SIZE, ec=EMBED_COMPONENT,
        # --- Новые параметры ---
        expect_hybrid_ecc=True,  # <--- Указываем, что ожидаем гибрид (или False)
        max_expected_packets=15,  # <--- Макс. пакетов (должно совпадать с embedder)
        # --- ECC параметры ---
        ue=USE_ECC,  # Передаем глобальный флаг USE_ECC
        bch_code=BCH_CODE_OBJECT,  # Передаем глобальный объект BCH
        # --- Остальные ---
        plb=PAYLOAD_LEN_BYTES, mw=MAX_WORKERS_EXTRACT
    )

    print(f"\n--- Extraction Results ---")
    extracted_hex = None
    if extracted_bytes:
        if len(extracted_bytes) == PAYLOAD_LEN_BYTES:
            extracted_hex = extracted_bytes.hex()
            print(f"  Payload Length OK.")
            print(f"  Decoded ID (Hex): {extracted_hex}")
            logging.info(f"Decoded ID: {extracted_hex}")
        else:
            # This case should ideally be handled by bits_to_bytes or decode_ecc returning None
            print(f"  ERROR: Decoded payload length mismatch! Got {len(extracted_bytes)}B, expected {PAYLOAD_LEN_BYTES}B.")
            logging.error(f"Decoded payload length mismatch!")
    else:
        print(f"  Extraction FAILED (No payload returned).")
        logging.error("Extraction failed or returned no payload.")

    # Сравнение с оригиналом
    if original_id:
        print(f"  Original ID (Hex): {original_id}")
        if extracted_hex and extracted_hex == original_id:
            print("\n  >>> ID MATCH <<<")
            logging.info("ID MATCH.")
        else:
            print("\n  >>> !!! ID MISMATCH or FAILED !!! <<<")
            logging.warning("ID MISMATCH or Extraction Failed.")
    else:
        print("\n  Original ID unavailable for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_time_main = time.time() - start_time_main
    logging.info(f"--- Total Extractor Time: {total_time_main:.2f} sec ---")
    print(f"\nExtraction finished. Log: {LOG_FILENAME}")


# --- Точка Входа ---
if __name__ == "__main__":
    original_dtcwt_backend = 'numpy'
    DTCWT_OPENCL_ENABLED = False # Default to False

    # --- Попытка переключить бэкенд DTCWT на OpenCL ---
    try:
        original_dtcwt_backend = dtcwt.backend_name
        logging.info(f"Original dtcwt backend: {original_dtcwt_backend}")
        logging.info("Attempting to switch dtcwt backend to OpenCL...")
        dtcwt.push_backend('opencl')
        current_backend = dtcwt.backend_name
        logging.info(f"DTCWT backend switched to: {current_backend}")
        if current_backend == 'opencl':
            try:
                logging.info("Initializing OpenCL backend via Transform2d()...")
                _test_transform = dtcwt.Transform2d()
                _test_data = np.random.rand(16, 16).astype(np.float32)
                _test_pyramid = _test_transform.forward(_test_data, nlevels=1)
                assert _test_pyramid is not None and _test_pyramid.lowpass is not None, "Test transform failed"
                logging.info("OpenCL backend initialized and test transform successful.")
                DTCWT_OPENCL_ENABLED = True
            except Exception as e_init:
                 logging.warning(f"OpenCL backend test/init failed: {e_init}. Falling back to NumPy.", exc_info=False)
                 if dtcwt.backend_name == 'opencl': dtcwt.pop_backend()
                 DTCWT_OPENCL_ENABLED = False
        else:
            logging.warning("push_backend('opencl') did not change backend! Using NumPy.")
            DTCWT_OPENCL_ENABLED = False
            if dtcwt.backend_name != 'numpy': # Ensure numpy is set if push failed
                try:
                    dtcwt.push_backend('numpy') # May error if stack is weird
                except Exception: pass # Ignore errors here
    except ImportError: logging.warning("dtcwt library not found."); DTCWT_OPENCL_ENABLED = False
    except ValueError as e_push: logging.warning(f"Failed switch to OpenCL: {e_push}. Using NumPy."); DTCWT_OPENCL_ENABLED = False
    except Exception as e_ocl: logging.warning(f"Error setting/testing OpenCL: {e_ocl}. Using NumPy.", exc_info=True); DTCWT_OPENCL_ENABLED = False

    # --- Логируем финальный активный бэкенд ПЕРЕД вызовом main ---
    try:
        final_backend_before_main = dtcwt.backend_name
        logging.info(f"Active DTCWT backend before calling main: {final_backend_before_main}")
        # Final consistency check
        if final_backend_before_main == 'opencl' and not DTCWT_OPENCL_ENABLED:
            logging.error("Inconsistency detected before main: OpenCL active but flag is False! Forcing NumPy.")
            try: dtcwt.push_backend('numpy')
            except Exception: pass
            DTCWT_OPENCL_ENABLED = False
        elif final_backend_before_main != 'opencl' and DTCWT_OPENCL_ENABLED:
            logging.error("Inconsistency detected before main: NumPy active but OpenCL flag is True! Resetting flag.")
            DTCWT_OPENCL_ENABLED = False
    except Exception as e_check:
         logging.error(f"Error checking backend before main: {e_check}")

    # --- Основной блок запуска ---
    if USE_ECC and not GALOIS_AVAILABLE:
        print("\nWARNING: USE_ECC=True, but galois unavailable/failed. ECC decoding disabled.")

    profiler = cProfile.Profile(); profiler.enable()
    final_exit_code = 0
    try:
        main()
        print(f"\n--- DTCWT Backend Used During Main: {'OpenCL (Forward Only)' if DTCWT_OPENCL_ENABLED else 'NumPy'} ---")
    except FileNotFoundError as e:
         print(f"\nERROR: Input file not found: {e}")
         logging.error(f"Input file not found: {e}", exc_info=True)
         final_exit_code = 1
    except Exception as e:
         logging.critical(f"Unhandled exception in main (Extractor): {e}", exc_info=True)
         print(f"\nCRITICAL ERROR: {e}. See log: {LOG_FILENAME}")
         final_exit_code = 1
    finally:
        profiler.disable(); stats = pstats.Stats(profiler)
        print("\n--- Profiling Stats (Top 30 Cumulative Time) ---")
        try:
             stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        except Exception as e_stats: print(f"Error printing stats: {e_stats}")
        print("-------------------------------------------------")
        backend_str = 'opencl' if DTCWT_OPENCL_ENABLED else 'numpy'
        profile_file = f"profile_extract_{backend_str}_batched_galois_t{BCH_T}.txt"
        try:
            with open(profile_file, "w") as f:
                stats_file = pstats.Stats(profiler, stream=f)
                stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved: {profile_file}"); print(f"Profiling stats saved: {profile_file}")
        except IOError as e: logging.error(f"Could not save profiling stats: {e}")
        except Exception as e_prof_save: logging.error(f"Error saving profile: {e_prof_save}", exc_info=True)


        # --- Блок восстановления исходного бэкенда ПОСЛЕ ВСЕГО ---
        try:
            if 'dtcwt' in sys.modules and 'original_dtcwt_backend' in locals() and dtcwt.backend_name != original_dtcwt_backend:
                logging.info(f"Attempting to restore original dtcwt backend: {original_dtcwt_backend}")
                max_pops = 10
                pop_count = 0
                # Pop until original is restored or stack is empty (or limit reached)
                while dtcwt.backend_name != original_dtcwt_backend and len(getattr(dtcwt, '_backend_stack', [])) > 0 and pop_count < max_pops:
                     dtcwt.pop_backend()
                     pop_count += 1
                # If still not matching, force push (this might mess up the stack if it was already correct)
                if dtcwt.backend_name != original_dtcwt_backend:
                     logging.warning(f"Stack popping didn't restore to {original_dtcwt_backend}, forcing push.")
                     dtcwt.push_backend(original_dtcwt_backend)
                     # Clean up potential extra push if stack existed
                     if hasattr(dtcwt,'_backend_stack') and len(dtcwt._backend_stack)>1 and dtcwt._backend_stack[0]==original_dtcwt_backend: dtcwt._backend_stack.pop(0);
                logging.info(f"DTCWT backend after restore attempt: {dtcwt.backend_name}")
        except Exception as e_restore:
            logging.warning(f"Could not restore dtcwt backend: {e_restore}", exc_info=True)

    sys.exit(final_exit_code)