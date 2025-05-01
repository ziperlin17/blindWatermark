# Файл: extractor_pytorch_wavelets.py (ЧАСТЬ 1 - Полная)
import cv2
import numpy as np
import random
import logging
import time
import json
import os
import hashlib
from PIL import Image # Оставлен на всякий случай
# line_profiler убран из примера, добавьте если нужно
# from line_profiler import profile
from scipy.fftpack import dct as scipy_dct # Переименовал для ясности
from scipy.linalg import svd as scipy_svd # Переименовал для ясности
# --- НОВЫЕ ИМПОРТЫ ---
import torch
import torch.nn.functional as F # Понадобится для interpolate
try:
    # Импортируем только нужные классы
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    # Определим классы-пустышки, чтобы код не падал при импорте, но выдавал ошибку при использовании
    class DTCWTForward: pass
    class DTCWTInverse: pass
    logging.error("Библиотека pytorch_wavelets не найдена!")
# ---------------------
from typing import List, Tuple, Optional, Dict, Any
# functools убран
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import uuid
from math import ceil
import cProfile
import pstats
from collections import Counter
import sys

try:
    import galois
    BCH_TYPE = galois.BCH # Используем реальный тип
    GALOIS_IMPORTED = True
    logging.info("galois library imported.")
except ImportError:
    class BCH: pass # Определяем пустышку
    BCH_TYPE = BCH
    GALOIS_IMPORTED = False
    logging.info("galois library not found.")
except Exception as import_err:
    class BCH: pass
    BCH_TYPE = BCH
    GALOIS_IMPORTED = False
    logging.error(f"Galois import error: {import_err}", exc_info=True)

# --- Глобальные Параметры ---
LAMBDA_PARAM: float = 0.05
ALPHA_MIN: float = 1.13
ALPHA_MAX: float = 1.28
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2 # Cb
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection' # Метод выбора колец
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 9 # Используем t=9
FPS: int = 30 # Не используется в extractor, но оставим
LOG_FILENAME: str = 'watermarking_extract_pytorch.log' # Новое имя
INPUT_EXTENSION: str = '.mp4' # Согласовано с embedder
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS_EXTRACT: Optional[int] = None # Оставляем для ThreadPool

 # Используем тип BCH (или пустышку)
GALOIS_AVAILABLE = False
BCH_CODE_OBJECT: Optional[BCH_TYPE] = None

if GALOIS_IMPORTED: # Выполняем только если импорт удался
    # logging.info("galois: импортирован (повторно).") # Это сообщение можно убрать
    _test_bch_ok = False
    _test_decode_ok = False
    try:
        # Получаем параметры из глобальных переменных
        _test_m = BCH_M
        _test_t = BCH_T
        _test_n = (1 << _test_m) - 1 # n = 255 для m=8
        _test_d = 2 * _test_t + 1    # d = 19 для t=9

        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        # Инициализируем объект BCH
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)

        # Определяем ожидаемое k на основе t
        if _test_t == 5: expected_k = 215
        elif _test_t == 7: expected_k = 201
        elif _test_t == 9: expected_k = 187 # <--- Ваше значение
        elif _test_t == 11: expected_k = 173
        elif _test_t == 15: expected_k = 131
        else:
             logging.error(f"Неизвестное ожидаемое k для t={_test_t}")
             expected_k = -1 # Вызовет ошибку проверки ниже

        # Проверяем параметры инициализированного объекта
        if expected_k != -1 and hasattr(_test_bch_galois, 't') and hasattr(_test_bch_galois, 'k') \
           and _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
             logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) инициализирован OK.")
             _test_bch_ok = True
             BCH_CODE_OBJECT = _test_bch_galois # <--- Присваиваем глобальный объект
        else:
             # Логируем детальную информацию об ошибке
             actual_t = getattr(_test_bch_galois, 't', 'N/A')
             actual_k = getattr(_test_bch_galois, 'k', 'N/A')
             logging.error(f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t={actual_t}, k={actual_k}.")
             _test_bch_ok = False
             BCH_CODE_OBJECT = None # Не присваиваем

        # Проводим тест декодирования, только если инициализация прошла успешно
        if _test_bch_ok and BCH_CODE_OBJECT is not None:
            try:
                _n_bits = BCH_CODE_OBJECT.n
                _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)
                GF2 = galois.GF(2); _dummy_cw_vec = GF2(_dummy_cw_bits)
                # Пытаемся декодировать нулевой вектор
                _msg, _flips = BCH_CODE_OBJECT.decode(_dummy_cw_vec, errors=True)
                # Проверяем, что декодирование не вызвало исключение и вернуло _flips
                # (значение _flips может быть 0 или None, это нормально)
                _test_decode_ok = (_flips is not None or _flips == 0) # Считаем успехом, если не было ошибки
                logging.info(f"galois: decode() test {'OK' if _test_decode_ok else 'failed/unexpected'} (flips={_flips}).")
            except Exception as decode_err:
                 logging.error(f"galois: decode() test failed with error: {decode_err}", exc_info=True)
                 _test_decode_ok = False

    except ValueError as ve: # Ошибка инициализации galois.BCH
         logging.error(f"galois: ОШИБКА ValueError при инициализации BCH: {ve}")
         BCH_CODE_OBJECT = None; _test_bch_ok = False; _test_decode_ok = False
    except AttributeError as ae: # Может возникнуть, если объект BCH не создался или не имеет нужных атрибутов
        logging.error(f"galois: ОШИБКА AttributeError при доступе к свойствам BCH: {ae}")
        BCH_CODE_OBJECT = None; _test_bch_ok = False; _test_decode_ok = False
    except Exception as test_err: # Другие неожиданные ошибки
         logging.error(f"galois: ОШИБКА теста инициализации/декодирования: {test_err}", exc_info=True)
         BCH_CODE_OBJECT = None; _test_bch_ok = False; _test_decode_ok = False

    # Финально устанавливаем флаг GALOIS_AVAILABLE
    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    # Если тесты не прошли, еще раз убедимся, что объект None
    if not GALOIS_AVAILABLE:
        BCH_CODE_OBJECT = None

# Финальное логирование статуса Galois
if GALOIS_AVAILABLE:
    logging.info("galois: Инициализация и тесты пройдены, объект BCH_CODE_OBJECT готов к использованию.")
else:
    logging.warning("galois: НЕ ДОСТУПЕН (ошибка импорта, инициализации или теста decode). ECC не будет работать.")


# --- Настройка логирования ---
# Очистка старых обработчиков, если есть
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG) # Раскомментировать для детального лога

# --- Логирование конфигурации ---
# Определяем потенциальную возможность использования ECC до вызова main
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Извлечения (PyTorch Wavelets) ---")
logging.info(f"PyTorch Wavelets Доступно: {PYTORCH_WAVELETS_AVAILABLE}")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit")
# Логируем параметры ECC, которые *будут* использоваться, если USE_ECC=True и GALOIS_AVAILABLE
logging.info(f"ECC Ожидается (для 1-го пак.): {USE_ECC}, Доступен/Работает: {GALOIS_AVAILABLE} (BCH m={BCH_M}, t={BCH_T})")
# logging.info(f"Ожид. Альфа для логирования: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}") # Не релевантно для экстрактора
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, N_RINGS_Total={N_RINGS}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS_EXTRACT or 'default'}) с батчингом.")
# Предупреждения об ECC выводятся позже, в main или extract_watermark_from_video
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error(f"NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE! Проверьте настройки.")
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE != BITS_PER_PAIR.")

# --- Базовые Функции ---

def dct_1d(s: np.ndarray) -> np.ndarray:
    """1D DCT используя SciPy (для NumPy массивов)."""
    return scipy_dct(s, type=2, norm='ortho')

# --- Функции-обертки для PyTorch DTCWT ---
def dtcwt_pytorch_forward(yp_tensor: torch.Tensor, xfm: DTCWTForward, device: torch.device, fn: int = -1) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Применяет прямое DTCWT PyTorch к одному каналу (2D тензору)."""
    if not PYTORCH_WAVELETS_AVAILABLE:
         logging.error("PyTorch Wavelets не доступна для dtcwt_pytorch_forward.")
         return None, None
    if not isinstance(yp_tensor, torch.Tensor):
        logging.error(f"[Frame:{fn}] Input is not a torch Tensor.")
        return None, None
    if yp_tensor.ndim != 2:
        logging.error(f"[Frame:{fn}] Input tensor must be 2D (H, W), got {yp_tensor.shape}")
        return None, None
    try:
        yp_tensor = yp_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        xfm = xfm.to(device)
        with torch.no_grad():
            Yl, Yh = xfm(yp_tensor)
        if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh:
             logging.error(f"[Frame:{fn}] DTCWTForward вернула некорректный результат (None или пустой Yh).")
             return None, None
        # logging.debug(f"[Frame:{fn}] PyTorch DTCWT FWD done. Yl shape: {Yl.shape}, Yh[0] shape: {Yh[0].shape}")
        return Yl, Yh
    except RuntimeError as e:
        if "CUDA out of memory" in str(e): logging.error(f"[Frame:{fn}] CUDA out of memory during PyTorch DTCWT forward!", exc_info=True)
        else: logging.error(f"[Frame:{fn}] PyTorch DTCWT forward runtime error: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logging.error(f"[Frame:{fn}] PyTorch DTCWT forward unexpected error: {e}", exc_info=True)
        return None, None

# Обратное преобразование не нужно для экстрактора, поэтому dtcwt_pytorch_inverse здесь не требуется

# --- Переписанная ring_division для PyTorch ---
def ring_division(lp_tensor: torch.Tensor, nr: int = N_RINGS, fn: int = -1) -> List[Optional[torch.Tensor]]:
    """Разбивает 2D PyTorch тензор на N концентрических колец. Возвращает список тензоров координат."""
    if not isinstance(lp_tensor, torch.Tensor) or lp_tensor.ndim != 2:
        logging.error(f"[Frame:{fn}] Invalid input for ring_division (expected 2D torch.Tensor). Got {type(lp_tensor)} with ndim {lp_tensor.ndim if hasattr(lp_tensor, 'ndim') else 'N/A'}")
        return [None] * nr
    H, W = lp_tensor.shape
    if H < 2 or W < 2:
         logging.warning(f"[Frame:{fn}] Tensor too small for ring division ({H}x{W})")
         return [None] * nr
    device = lp_tensor.device
    try:
        rr, cc = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32),
                                indexing='ij')
        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
        distances = torch.sqrt((rr - center_r)**2 + (cc - center_c)**2)
        min_dist, max_dist = torch.tensor(0.0, device=device), torch.max(distances)
        if max_dist < 1e-9:
             ring_bins = torch.tensor([0.0] * (nr + 1), device=device); ring_bins[1:] = max_dist + 1e-6
        else:
             ring_bins = torch.linspace(min_dist.item(), (max_dist + 1e-6).item(), nr + 1, device=device)
        ring_indices = torch.zeros_like(distances, dtype=torch.long) - 1
        for i in range(nr):
             lower_bound = ring_bins[i]; upper_bound = ring_bins[i+1]
             if i < nr - 1: mask = (distances >= lower_bound) & (distances < upper_bound)
             else: mask = (distances >= lower_bound) & (distances <= upper_bound)
             ring_indices[mask] = i
        ring_indices[distances < ring_bins[1]] = 0
        rings: List[Optional[torch.Tensor]] = [None] * nr
        for rdx in range(nr):
            coords_tensor = torch.nonzero(ring_indices == rdx, as_tuple=False)
            if coords_tensor.shape[0] > 0: rings[rdx] = coords_tensor.long()
        return rings
    except Exception as e:
         logging.error(f"Ring division PyTorch error Frame {fn}: {e}", exc_info=True)
         return [None] * nr


# --- calculate_entropies - остается на NumPy ---
def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    eps=1e-12; shannon_entropy=0.; collision_entropy=0.
    if rv.size > 0:
        rvc = np.clip(rv.copy(), 0.0, 1.0)
        if np.all(rvc == rvc[0]): return 0.0, 0.0
        hist, _ = np.histogram(rvc, bins=256, range=(0., 1.), density=False)
        total_count = rvc.size
        if total_count > 0:
            probabilities = hist / total_count
            p = probabilities[probabilities > eps]
            if p.size > 0:
                shannon_entropy = -np.sum(p * np.log2(p))
                ee = -np.sum(p*np.exp(1.-p)) # Старая формула "collision"
                collision_entropy = ee
    return shannon_entropy, collision_entropy

# compute_adaptive_alpha_entropy не нужна в экстракторе

# --- get_fixed_pseudo_random_rings - остается без изменений ---
def get_fixed_pseudo_random_rings(pi: int, nr: int, ps: int) -> List[int]:
    if ps <= 0: return []
    if ps > nr: ps = nr
    seed_str = str(pi).encode('utf-8'); hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big'); prng = random.Random(seed_int)
    try: candidate_indices = prng.sample(range(nr), ps)
    except ValueError: candidate_indices = list(range(nr)); prng.shuffle(candidate_indices); candidate_indices = candidate_indices[:ps]
    logging.debug(f"[P:{pi}] Candidates: {candidate_indices}");
    return candidate_indices

# --- bits_to_bytes - остается без изменений ---
def bits_to_bytes(bit_list: List[Optional[int]]) -> Optional[bytes]:
    """Converts a list of bits (0/1 or None) into bytes, ignoring None."""
    valid_bits = [b for b in bit_list if b is not None and b in (0, 1)] # Добавил проверку b in (0, 1)
    num_bits = len(valid_bits)

    if num_bits == 0:
        return b'' # Возвращаем пустые байты, если нет валидных бит

    # --- ИСПРАВЛЕНИЕ: Вычисляем remainder ПЕРЕД использованием ---
    remainder = num_bits % 8
    # -------------------------------------------------------------

    # Pad with zeros at the end if length is not a multiple of 8
    if remainder != 0:
        padding_len = 8 - remainder
        valid_bits.extend([0] * padding_len)
        # logging.warning(f"Bit list length ({num_bits}) not multiple of 8. Padded with {padding_len} zeros.") # Можно вернуть лог
        num_bits += padding_len # Обновляем num_bits после паддинга

    byte_array = bytearray()
    for i in range(0, num_bits, 8):
        byte_chunk = valid_bits[i:i+8]
        try:
            # Убедимся, что в чанке точно 8 бит (особенно важно после паддинга)
            if len(byte_chunk) != 8:
                 logging.error(f"Ошибка формирования байтового чанка: {byte_chunk}")
                 return None
            byte_val = int("".join(map(str, byte_chunk)), 2)
            byte_array.append(byte_val)
        except ValueError:
            # Эта ошибка может возникнуть, если в valid_bits попало что-то кроме 0 или 1
            logging.error(f"Неверные символы в битовом чанке при конвертации в байт: {byte_chunk}")
            return None # Ошибка конвертации
    return bytes(byte_array)

# --- decode_ecc - остается без изменений (работает с NumPy) ---
def decode_ecc(packet_bits_list: List[int], bch_code: Optional[galois.BCH], expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    if not GALOIS_AVAILABLE or bch_code is None: logging.error("ECC decode called but unavailable."); return None, -1
    n_corrected = -1
    try:
        n=bch_code.n; k=bch_code.k; expected_payload_bits=expected_data_len_bytes*8
        if len(packet_bits_list) != n: logging.error(f"Decode ECC: Bad packet len {len(packet_bits_list)} vs {n}"); return None, -1
        if expected_payload_bits > k: logging.error(f"Decode ECC: Payload len {expected_payload_bits} > k {k}"); return None, -1
        packet_bits_np=np.array(packet_bits_list, dtype=np.uint8); GF=bch_code.field; rx_vec=GF(packet_bits_np)
        try: corr_msg_vec, n_corrected = bch_code.decode(rx_vec, errors=True)
        except galois.errors.UncorrectableError: logging.warning("Galois ECC: Uncorrectable error."); return None, -1
        corr_k_bits=corr_msg_vec.view(np.ndarray).astype(np.uint8)
        if corr_k_bits.size < expected_payload_bits: logging.error(f"Decode ECC: Decoded msg len {corr_k_bits.size} < payload {expected_payload_bits}"); return None, n_corrected
        payload_bits=corr_k_bits[:expected_payload_bits]; payload_bytes = bits_to_bytes(payload_bits.tolist())
        if payload_bytes is None: logging.error("Decode ECC: bits_to_bytes failed."); return None, n_corrected
        # Логирование успешного декодирования (уменьшено)
        logging.info(f"Galois ECC: Decoded, corrected {n_corrected} errors.")
        return payload_bytes, n_corrected
    except Exception as e: logging.error(f"Decode ECC unexpected error: {e}", exc_info=True); return None, -1

def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Читает видеофайл и возвращает список BGR NumPy кадров и FPS."""
    logging.info(f"Reading: {video_path}")
    frames: List[np.ndarray] = []
    fps = float(FPS) # Значение по умолчанию
    cap = None
    h, w = -1, -1
    try:
        if not os.path.exists(video_path):
             raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0:
             # Попытка прочитать первый кадр, чтобы узнать размер
             ret, f_check = cap.read()
             if ret and f_check is not None:
                  h, w, _ = f_check.shape
                  logging.info(f"Размеры кадра определены по первому кадру: {w}x{h}")
                  # Важно вернуть указатель или прочитать заново с 0
                  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             else:
                  raise ValueError("Не удалось определить размеры кадра видео.")
        logging.info(f"Props: {w}x{h}@{fps:.2f} FPS ~{fc if fc > 0 else '?'} frames")

        frame_count = 0; read_count = 0; none_count = 0; invalid_count = 0
        while True:
            ret, f = cap.read(); frame_count += 1
            if not ret: logging.debug(f"End of video reached or read error at frame approx {frame_count}."); break
            if f is None: none_count += 1; logging.warning(f"Пустой кадр #{frame_count-1}"); continue
            if f.ndim == 3 and f.shape[0] == h and f.shape[1] == w and f.dtype == np.uint8: frames.append(f); read_count += 1
            else: invalid_count += 1; logging.warning(f"Пропущен невалидный кадр #{frame_count-1}. Shape: {f.shape}, Dtype: {f.dtype}")
        logging.info(f"Read loop finished. Valid: {read_count}, None: {none_count}, Invalid: {invalid_count}")
        if read_count == 0: raise ValueError("Не прочитано ни одного валидного кадра.")
    except Exception as e: logging.error(f"Ошибка чтения видео: {e}", exc_info=True); frames=[]
    finally:
        if cap and cap.isOpened(): logging.debug("Releasing capture"); cap.release()
    return frames, fps

# --- Функция извлечения ОДНОГО бита ("БН" версия для PyTorch) ---
# @profile # Добавьте, если нужно профилировать
def extract_single_bit(L1_tensor: torch.Tensor, L2_tensor: torch.Tensor, ring_idx: int, n_rings: int, fn: int) -> Optional[int]:
    """
    Извлекает один бит из заданного кольца, используя ПРЕДВАРИТЕЛЬНО ВЫЧИСЛЕННЫЕ
    действительные lowpass тензоры L1 и L2. (BN-PyTorch-Optimized)
    """
    pair_index = fn // 2
    prefix = f"[BN P:{pair_index}, R:{ring_idx}]" # Префикс логов
    # logging.debug(f"{prefix} Starting extraction...") # Отладочный лог начала

    try:
        # --- Шаг 1: Проверка входных данных ---
        if L1_tensor is None or L2_tensor is None or L1_tensor.shape != L2_tensor.shape \
           or not isinstance(L1_tensor, torch.Tensor) or not isinstance(L2_tensor, torch.Tensor) \
           or L1_tensor.ndim != 2 or L2_tensor.ndim != 2:
            logging.warning(f"{prefix} Invalid L1/L2 provided.")
            return None
        if not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
             logging.warning(f"{prefix} L1/L2 not float. L1:{L1_tensor.dtype}, L2:{L2_tensor.dtype}")
             return None
        logging.debug(f"{prefix} Input tensors L1/L2 validated. Shape: {L1_tensor.shape}")

        # --- Шаг 2: Кольцевое деление (по тензорам L1, L2) ---
        r1c = ring_division(L1_tensor, n_rings, fn) # Используем тензорную ring_division
        r2c = ring_division(L2_tensor, n_rings, fn + 1)

        if not(0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
             logging.warning(f"{prefix} Invalid ring index {ring_idx}. Max index: {len(r1c)-1}")
             return None
        cd1_tensor = r1c[ring_idx] # Координаты - тензоры
        cd2_tensor = r2c[ring_idx]
        # Увеличена проверка мин. размера до 10
        if cd1_tensor is None or cd2_tensor is None or cd1_tensor.shape[0] < 10 or cd2_tensor.shape[0] < 10:
             logging.debug(f"{prefix} Ring coords None or ring too small (size1={cd1_tensor.shape[0] if cd1_tensor is not None else 'None'}, size2={cd2_tensor.shape[0] if cd2_tensor is not None else 'None'}).")
             return None
        logging.debug(f"{prefix} Ring coords obtained. Size1={cd1_tensor.shape[0]}, Size2={cd2_tensor.shape[0]}")


        # --- Шаг 3: Извлечение значений, DCT, SVD ---
        try:
            rows1, cols1 = cd1_tensor[:, 0], cd1_tensor[:, 1]
            rows2, cols2 = cd2_tensor[:, 0], cd2_tensor[:, 1]
            rv1_tensor = L1_tensor[rows1, cols1]
            rv2_tensor = L2_tensor[rows2, cols2]
        except IndexError: logging.warning(f"{prefix} Index error getting ring values."); return None

        # Синхронизация размера
        min_s = min(rv1_tensor.numel(), rv2_tensor.numel())
        if min_s == 0: logging.debug(f"{prefix} Ring values empty after min_s check."); return None
        if rv1_tensor.numel() != rv2_tensor.numel():
            logging.warning(f"{prefix} Ring tensor sizes differ ({rv1_tensor.numel()} vs {rv2_tensor.numel()}), synchronizing to {min_s}.")
            rv1_tensor = rv1_tensor[:min_s]
            rv2_tensor = rv2_tensor[:min_s]

        # --- Конвертация в NumPy для SciPy DCT/SVD ---
        rv1_np = rv1_tensor.cpu().numpy().astype(np.float32)
        rv2_np = rv2_tensor.cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(rv1_np)) or not np.all(np.isfinite(rv2_np)): logging.warning(f"{prefix} NaN/inf in ring values after conversion."); return None
        logging.debug(f"{prefix} Converted ring values to NumPy. Size={rv1_np.size}")

        # --- DCT и SVD (используя SciPy) ---
        logging.info(
            f"{prefix} [BN - Перед DCT] rv1_np (из L1_tensor): size={rv1_np.size}, min={np.min(rv1_np):.4e}, max={np.max(rv1_np):.4e}, mean={np.mean(rv1_np):.4e}, std={np.std(rv1_np):.4e}")
        # Логируем также rv2_np
        logging.info(
            f"{prefix} [BN - Перед DCT] rv2_np (из L2_tensor): size={rv2_np.size}, min={np.min(rv2_np):.4e}, max={np.max(rv2_np):.4e}, mean={np.mean(rv2_np):.4e}, std={np.std(rv2_np):.4e}")

        d1 = dct_1d(rv1_np); d2 = dct_1d(rv2_np)
        if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(d2)): logging.warning(f"{prefix} NaN/inf after DCT."); return None
        logging.debug(f"{prefix} DCT done. d1[0]={d1[0]:.4e}, d2[0]={d2[0]:.4e}")

        try: S1_vec = scipy_svd(d1.reshape(-1, 1), compute_uv=False); S2_vec = scipy_svd(d2.reshape(-1, 1), compute_uv=False)
        except np.linalg.LinAlgError: logging.warning(f"{prefix} SVD failed."); return None
        if S1_vec is None or S1_vec.size == 0 or S2_vec is None or S2_vec.size == 0: logging.warning(f"{prefix} SVD empty result."); return None
        if not np.all(np.isfinite(S1_vec)) or not np.all(np.isfinite(S2_vec)): logging.warning(f"{prefix} SVD NaN/inf."); return None

        s1 = S1_vec[0]; s2 = S2_vec[0]
        logging.debug(f"{prefix} s1={s1:.6e}, s2={s2:.6e}") # Основной лог SVD

        # --- Шаг 4: Принятие решения ---
        eps = 1e-12; threshold = 1.0
        if abs(s2) < eps: logging.warning(f"{prefix} s2 near zero ({s2:.2e}). Ratio unreliable."); return None
        ratio = s1 / s2
        extracted_bit = 1 if ratio >= threshold else 0
        logging.debug(f"{prefix} ratio={ratio:.6f} -> Bit={extracted_bit}") # Основной лог результата

        return extracted_bit
    except Exception as e:
        logging.error(f"Extract single bit BN failed (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
        return None

# --- ИЗМЕНЕННЫЙ Воркер для обработки батча ("БН" версия для PyTorch) ---
def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
    """
    Обрабатывает батч задач извлечения: выполняет DTCWT один раз на пару,
    затем вызывает extract_single_bit для выбранных колец. (BN-PyTorch-Optimized)
    """
    batch_results: Dict[int, List[Optional[int]]] = {}
    if not batch_args_list: return {}
    args_example = batch_args_list[0]
    nr = args_example.get('n_rings', N_RINGS); nrtu = args_example.get('num_rings_to_use', NUM_RINGS_TO_USE)
    cps = args_example.get('candidate_pool_size', CANDIDATE_POOL_SIZE); ec = args_example.get('embed_component', EMBED_COMPONENT)

    for args in batch_args_list:
        pair_idx = args.get('pair_idx', -1)
        f1_bgr = args.get('frame1'); f2_bgr = args.get('frame2')
        device = args.get('device'); dtcwt_fwd = args.get('dtcwt_fwd')
        if pair_idx == -1 or f1_bgr is None or f2_bgr is None or device is None or dtcwt_fwd is None:
            logging.error(f"Missing args P:{pair_idx} in _extract_batch_worker")
            batch_results[pair_idx] = [None] * nrtu; continue

        fn = 2 * pair_idx
        extracted_bits_for_pair: List[Optional[int]] = [None] * nrtu
        L1_tensor: Optional[torch.Tensor] = None; L2_tensor: Optional[torch.Tensor] = None

        try:
            # --- Шаг 1: Преобразование цвета и DTCWT ---
            if not isinstance(f1_bgr, np.ndarray) or not isinstance(f2_bgr, np.ndarray): logging.warning(f"[BN P:{pair_idx}] Input frames not numpy."); batch_results[pair_idx] = extracted_bits_for_pair; continue
            y1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb); y2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2YCrCb)
            c1 = y1[:, :, ec].astype(np.float32) / 255.0; c2 = y2[:, :, ec].astype(np.float32) / 255.0
            comp1_tensor = torch.from_numpy(c1).to(device=device, dtype=torch.float32)
            comp2_tensor = torch.from_numpy(c2).to(device=device, dtype=torch.float32)

            Yl_t, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
            Yl_t1, _ = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, fn + 1)

            if Yl_t is None or Yl_t1 is None:
                 logging.warning(f"[BN P:{pair_idx}] DTCWT forward failed (returned None).")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            # --- ИСПРАВЛЕНИЕ: Присваиваем L1/L2 ПОСЛЕ проверки Yl_t/Yl_t1 на None ---
            # Убираем лишние измерения N, C и проверяем тип
            if Yl_t.dim() > 2: L1_tensor = Yl_t.squeeze()
            elif Yl_t.dim() == 2: L1_tensor = Yl_t # Уже 2D
            else: raise ValueError(f"Unexpected Yl_t dim: {Yl_t.dim()}")

            if Yl_t1.dim() > 2: L2_tensor = Yl_t1.squeeze()
            elif Yl_t1.dim() == 2: L2_tensor = Yl_t1
            else: raise ValueError(f"Unexpected Yl_t1 dim: {Yl_t1.dim()}")

            # Проверка типа и формы после squeeze
            if not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
                 logging.error(f"[BN P:{pair_idx}] L1/L2 not float after squeeze! L1:{L1_tensor.dtype}, L2:{L2_tensor.dtype}")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue
            if L1_tensor.shape != L2_tensor.shape:
                 logging.warning(f"[BN P:{pair_idx}] L1/L2 shape mismatch after squeeze! L1:{L1_tensor.shape}, L2:{L2_tensor.shape}")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue
            # -----------------------------------------------------------------------

            # --- Шаг 2: Выбор колец ---
            coords = ring_division(L1_tensor, nr, fn) # Используем L1_tensor
            if coords is None or len(coords) != nr: logging.warning(f"[BN P:{pair_idx}] Ring division failed."); batch_results[pair_idx] = extracted_bits_for_pair; continue
            candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
            current_nrtu = nrtu
            if len(candidate_rings) < current_nrtu: logging.warning(f"[BN P:{pair_idx}] Not enough candidates."); current_nrtu=len(candidate_rings)
            if current_nrtu == 0: logging.error(f"[BN P:{pair_idx}] No candidates/rings."); batch_results[pair_idx] = []; continue

            entropies = []; min_pixels_for_entropy = 10
            for r_idx_cand in candidate_rings:
                entropy_val = -float('inf')
                if 0 <= r_idx_cand < len(coords) and isinstance(coords[r_idx_cand], torch.Tensor) and coords[r_idx_cand].shape[0] >= min_pixels_for_entropy:
                     c_tensor = coords[r_idx_cand]
                     try:
                           rows, cols = c_tensor[:, 0], c_tensor[:, 1]
                           rv_tensor = L1_tensor[rows, cols] # L1_tensor здесь определен
                           rv_np = rv_tensor.cpu().numpy()
                           shannon_entropy, _ = calculate_entropies(rv_np, fn, r_idx_cand)
                           if np.isfinite(shannon_entropy): entropy_val = shannon_entropy
                     except IndexError: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] IndexError entropy")
                     except Exception as e_entr: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] Entropy error: {e_entr}")
                entropies.append((entropy_val, r_idx_cand))
            entropies.sort(key=lambda x: x[0], reverse=True)
            selected_rings = [idx for e, idx in entropies if e > -float('inf')][:current_nrtu]

            if len(selected_rings) < current_nrtu: # Fallback
                logging.warning(f"[BN P:{pair_idx}] Fallback ring selection.")
                deterministic_fallback = candidate_rings[:current_nrtu]
                for ring in deterministic_fallback:
                    if ring not in selected_rings: selected_rings.append(ring)
                    if len(selected_rings) == current_nrtu: break
                if len(selected_rings) < current_nrtu: logging.error(f"[BN P:{pair_idx}] Fallback failed."); batch_results[pair_idx] = [None]*current_nrtu; continue
            # logging.info(f"[BN P:{pair_idx}] Selected rings: {selected_rings}")

            # --- Шаг 3: Извлечение бит ---
            extracted_bits_for_pair = [None] * len(selected_rings)
            for i, ring_idx_to_extract in enumerate(selected_rings):
                 extracted_bits_for_pair[i] = extract_single_bit(L1_tensor, L2_tensor, ring_idx_to_extract, nr, fn) # Вызываем БН-версию

            batch_results[pair_idx] = extracted_bits_for_pair

        except cv2.error as cv_err: logging.error(f"OpenCV error P:{pair_idx}: {cv_err}"); batch_results[pair_idx] = [None] * nrtu
        except Exception as e: logging.error(f"Error processing pair {pair_idx} in BN worker: {e}", exc_info=True); batch_results[pair_idx] = [None] * nrtu

    return batch_results
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
        # --- Новые параметры для PyTorch ---
        device: Optional[torch.device] = None,      # Устройство для вычислений
        dtcwt_fwd: Optional[DTCWTForward] = None, # Экземпляр прямого преобразования
        # --- Остальные параметры ---
        plb:int=PAYLOAD_LEN_BYTES,
        mw:Optional[int]=MAX_WORKERS_EXTRACT      # Макс. число воркеров
    ) -> Optional[bytes]:
    """
    Основная функция, управляющая процессом извлечения с использованием PyTorch Wavelets (БН),
    ThreadPoolExecutor, батчинга и ПОБИТОВОГО мажоритарного голосования (с гибридной логикой).
    """
    # Проверка наличия PyTorch объектов
    if not PYTORCH_WAVELETS_AVAILABLE:
        logging.critical("PyTorch Wavelets не доступна!")
        return None
    if device is None or dtcwt_fwd is None:
         logging.critical("Device или DTCWTForward не переданы в extract_watermark_from_video!")
         return None

    logging.info(f"--- Starting Extraction (PyTorch BN, Hybrid: {expect_hybrid_ecc}, Max Pkts: {max_expected_packets}) ---")
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

    # Проверяем возможность ECC для первого пакета
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

    if bp <= 0: logging.error("Bits per pair (bp) must be positive."); return None # Защита от деления на ноль
    pairs_needed = ceil(max_possible_bits / bp)
    pairs_to_process = min(total_pairs_available, pairs_needed)
    logging.info(f"Target extraction: Up to {max_expected_packets} packets.")
    logging.info(f"Frame pairs: Available={total_pairs_available}, Needed(max)={pairs_needed}, To Process={pairs_to_process}")

    if pairs_to_process == 0:
        logging.warning("Zero pairs to process based on calculations.")
        return None

    # --- Подготовка аргументов для батчей ---
    all_pairs_args = []
    skipped_pairs = 0
    for pair_idx in range(pairs_to_process):
        i1 = 2 * pair_idx; i2 = i1 + 1
        if i2 >= nf or frames[i1] is None or frames[i2] is None:
            skipped_pairs += 1
            continue
        # Передаем все необходимые параметры, включая объекты PyTorch
        args = {'pair_idx': pair_idx, 'frame1': frames[i1], 'frame2': frames[i2],
                'n_rings': nr, 'num_rings_to_use': nrtu, 'candidate_pool_size': cps,
                'embed_component': ec,
                'device': device, 'dtcwt_fwd': dtcwt_fwd} # Передаем объекты
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0: logging.warning(f"Skipped {skipped_pairs} pairs during task preparation.")
    if num_valid_tasks == 0: logging.error("No valid extraction tasks generated."); return None

    # --- Запуск ThreadPoolExecutor ---
    num_workers = mw if mw is not None and mw > 0 else (os.cpu_count() or 1)
    # Используем меньший размер батча для потенциально лучшего баланса нагрузки
    batch_size = max(1, ceil(num_valid_tasks / (num_workers * 4))) # Делим на 4*число воркеров
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i : i + batch_size] for i in range(0, num_valid_tasks, batch_size) if all_pairs_args[i:i+batch_size]] # Убираем пустые батчи

    logging.info(f"Launching {len(batched_args_list)} batches ({num_valid_tasks} pairs) using ThreadPool (mw={num_workers}, batch_size≈{batch_size})...")

    extracted_bits_map: Dict[int, List[Optional[int]]] = {}
    ppc = 0; fpe = 0 # Счетчики пар
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
             future_to_batch_idx = {executor.submit(_extract_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}
             for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                original_batch = batched_args_list[batch_idx] # Для подсчета ошибок
                try:
                    batch_results_map = future.result()
                    extracted_bits_map.update(batch_results_map)
                    ppc += len(batch_results_map)
                    # Считаем ошибки внимательнее: пара считается ошибочной, если *хотя бы один* бит None
                    fpe += sum(1 for bits in batch_results_map.values() if bits is None or None in bits)
                except Exception as e:
                     batch_size_failed = len(original_batch)
                     logging.error(f"Batch {batch_idx} (size {batch_size_failed}) execution failed: {e}", exc_info=True)
                     fpe += batch_size_failed
    except Exception as e:
        logging.critical(f"ThreadPoolExecutor critical error during extraction: {e}", exc_info=True)
        return None

    logging.info(f"Extraction task processing finished. Processed pairs confirmed: {ppc}. Pairs with errors/Nones: {fpe}.")
    if ppc == 0: logging.error("No pairs processed successfully by workers."); return None

    # --- Сборка общего потока извлеченных бит ---
    extracted_bits_all: List[Optional[int]] = []
    for pair_idx in range(pairs_to_process): # Итерируем по ВСЕМ парам, которые ДОЛЖНЫ были быть обработаны
        bits = extracted_bits_map.get(pair_idx)
        if bits is not None:
             # Проверяем, что вернулся список правильной длины (bp)
             if len(bits) == bp:
                  extracted_bits_all.extend(bits)
             else:
                  logging.warning(f"Pair {pair_idx} returned incorrect number of bits ({len(bits)} != {bp}). Adding Nones.")
                  extracted_bits_all.extend([None] * bp)
                  # fpe += 1 # Ошибку уже посчитали выше
        else:
            # Если для пары нет результата (None в карте или не было ключа)
            extracted_bits_all.extend([None] * bp)

    total_bits_collected = len(extracted_bits_all)
    valid_bits = [b for b in extracted_bits_all if b is not None]
    num_valid_bits = len(valid_bits)
    num_none_bits = total_bits_collected - num_valid_bits # Исправлено здесь
    success_rate = (num_valid_bits / total_bits_collected) * 100 if total_bits_collected > 0 else 0
    logging.info(f"Bit collection: Total collected={total_bits_collected}, Valid (non-None)={num_valid_bits} ({success_rate:.1f}%), None/Error={num_none_bits}.")
    if not valid_bits: logging.error("No valid (non-None) bits extracted."); return None

    # --- Гибридное Декодирование Пакетов ---
    all_payload_attempts_bits: List[Optional[List[int]]] = []
    decoded_success_count = 0; decode_failed_count = 0; total_corrected_symbols = 0
    num_processed_bits = 0

    print("\n--- Попытки Декодирования Пакетов ---")
    print(f"{'Pkt #':<6} | {'Type':<7} | {'ECC Status':<18} | {'Corrected':<10} | {'Payload (Hex)':<20}")
    print("-" * 68)

    for i in range(max_expected_packets):
        is_first_packet = (i == 0)
        use_ecc_for_this = is_first_packet and expect_hybrid_ecc and ecc_possible_for_first
        current_packet_len = packet_len_if_ecc if use_ecc_for_this else packet_len_if_raw
        packet_type_str = "ECC" if use_ecc_for_this else "Raw"

        start_idx = num_processed_bits
        end_idx = start_idx + current_packet_len

        if end_idx > num_valid_bits:
            logging.warning(f"Not enough valid bits remaining for potential packet {i + 1} (type {packet_type_str}, needed {current_packet_len}, have {num_valid_bits - start_idx}). Stopping decode.")
            break

        packet_candidate_bits = valid_bits[start_idx:end_idx]
        num_processed_bits += current_packet_len

        payload_bytes: Optional[bytes] = None
        payload_bits: Optional[List[int]] = None
        errors: int = -1
        status_str = f"Failed ({packet_type_str})"
        payload_hex_str = "N/A"

        if use_ecc_for_this:
            if bch_code is not None:
                payload_bytes, errors = decode_ecc(packet_candidate_bits, bch_code, plb)
                if payload_bytes is not None:
                    corrected_count = errors if errors != -1 else 0
                    status_str = f"OK (ECC: {corrected_count} fixed)"
                    if errors > 0: total_corrected_symbols += errors
                else:
                    status_str = f"Uncorrectable (ECC)" if errors == -1 else "ECC Decode Error"
                    # decode_failed_count увеличится ниже, если payload_bits останется None
            else: status_str = "ECC Code Missing"
        else: # Raw
            if len(packet_candidate_bits) >= payload_len_bits:
                 payload_candidate_bits_raw = packet_candidate_bits[:payload_len_bits]
                 packet_bytes_raw = bits_to_bytes(payload_candidate_bits_raw)
                 if packet_bytes_raw is not None and len(packet_bytes_raw) == plb:
                     payload_bytes = packet_bytes_raw; errors = 0; status_str = "OK (Raw)"
                 else: status_str = "Failed (Raw Convert)"
            else: status_str = "Failed (Raw Short)"

        if payload_bytes is not None:
            payload_hex_str = payload_bytes.hex()
            try:
                payload_np_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
                if len(payload_np_bits) == payload_len_bits:
                    payload_bits = payload_np_bits.tolist(); decoded_success_count += 1
                else: status_str += "[Len Fail]"; payload_bits = None
            except Exception as e_unpack: status_str += f"[UnpFail:{e_unpack}]"; payload_bits = None

        # Увеличиваем счетчик неудачи только если статус не начинался с Failed
        # и payload_bits в итоге None
        if payload_bits is None and not status_str.startswith("Failed"):
            decode_failed_count += 1

        # Добавляем результат (или None)
        logging.debug(f"Appending to list for Pkt {i+1}. payload_bits is None: {payload_bits is None}")
        all_payload_attempts_bits.append(payload_bits)

        # Выводим строку таблицы
        corrected_str = str(errors) if errors != -1 else "-"
        print(f"{i+1:<6} | {packet_type_str:<7} | {status_str:<18} | {corrected_str:<10} | {payload_hex_str:<20}")

    print("-" * 68)
    logging.info(f"Decode attempts summary: Total attempted packets = {len(all_payload_attempts_bits)}, Success (yielded {payload_len_bits} bits) = {decoded_success_count}, Failed/Skipped = {decode_failed_count}.")
    if ecc_possible_for_first and expect_hybrid_ecc:
         logging.info(f"Total ECC corrections reported for first packet: {total_corrected_symbols}.")


    # --- Побитовое Голосование ---
    num_attempted_packets = len(all_payload_attempts_bits)
    if num_attempted_packets == 0: logging.error("No packets were attempted for voting."); return None
    first_packet_payload = all_payload_attempts_bits[0] if num_attempted_packets > 0 else None
    valid_decoded_payloads = [p for p in all_payload_attempts_bits if p is not None]
    num_valid_packets_for_vote = len(valid_decoded_payloads)
    if num_valid_packets_for_vote == 0: logging.error(f"No valid {payload_len_bits}-bit payloads for voting."); return None

    final_payload_bits = []; logging.info(f"Performing bit-wise majority vote across {num_valid_packets_for_vote} valid packets (tie-break to first packet if valid)...")
    print("\n--- Bit-wise Voting Details ---"); print(f"{'Bit Pos':<8} | {'Votes 0':<8} | {'Votes 1':<8} | {'Winner':<8} | {'Tiebreak?':<10}"); print("-" * 50)

    for j in range(payload_len_bits):
        votes_for_0 = 0; votes_for_1 = 0
        for i in range(num_valid_packets_for_vote):
            if j < len(valid_decoded_payloads[i]):
                 if valid_decoded_payloads[i][j] == 1: votes_for_1 += 1
                 else: votes_for_0 += 1
            else: logging.warning(f"Index error during voting: bit {j}, pkt {i}")

        winner_bit: Optional[int] = None; tiebreak_used = "No"; valid_votes_count = votes_for_0 + votes_for_1

        if valid_votes_count == 0: # Случай, если все пакеты имели ошибку длины для этого бита
             logging.error(f"Bit position {j}: No valid votes found! Voting failed.")
             print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {'FAIL':<8} | {'N/A':<10}")
             final_payload_bits = None; break
        elif votes_for_1 > votes_for_0: winner_bit = 1
        elif votes_for_0 > votes_for_1: winner_bit = 0
        else: # Ничья
            tiebreak_used = "Yes"; logging.warning(f"Bit position {j}: Tie ({votes_for_0} vs {votes_for_1}). Using first pkt.")
            if first_packet_payload is not None and j < len(first_packet_payload): winner_bit = first_packet_payload[j]
            else: logging.error(f"Cannot resolve tie for bit {j}: First packet invalid!"); print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {'FAIL':<8} | {tiebreak_used:<10}"); final_payload_bits = None; break
        if winner_bit is None: logging.error(f"Winner bit None for pos {j}"); final_payload_bits = None; break # Не должно случиться

        final_payload_bits.append(winner_bit)
        print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {winner_bit:<8} | {tiebreak_used:<10}")

    print("-" * 50)
    if final_payload_bits is None: logging.error("Voting failed."); return None;
    logging.info(f"Voting complete.")

    # --- Конвертация и возврат результата ---
    final_payload_bytes = bits_to_bytes(final_payload_bits)
    if final_payload_bytes is None: logging.error("bits_to_bytes failed."); return None
    if len(final_payload_bytes) != plb:
        logging.error(f"Final length mismatch: {len(final_payload_bytes)}B != {plb}B.")
        # Попытка обрезать только если был паддинг в bits_to_bytes
        if len(final_payload_bytes) > plb and payload_len_bits % 8 != 0 and len(final_payload_bytes) - (len(final_payload_bytes) % plb) == plb:
            logging.warning(f"Attempting trim final payload due to padding.")
            final_payload_bytes = final_payload_bytes[:plb]
        else: return None
    logging.info(f"Final ID after bit-wise voting: {final_payload_bytes.hex()}")
    end_time = time.time(); logging.info(f"Extraction done. Total time: {end_time - start_time:.2f} sec.")
    return final_payload_bytes

def main():
    start_time_main = time.time()
    # --- Инициализация PyTorch и Galois ---
    if not PYTORCH_WAVELETS_AVAILABLE: print("ERROR: pytorch_wavelets not found."); return
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")
    # Определяем устройство
    if torch.cuda.is_available():
        try: device = torch.device("cuda"); torch.cuda.get_device_name(0); logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        except Exception as e: logging.warning(f"CUDA init failed ({e}). Using CPU."); device = torch.device("cpu")
    else: device = torch.device("cpu"); logging.info("Using CPU.")
    # Создаем экземпляр DTCWTForward
    dtcwt_fwd: Optional[DTCWTForward] = None
    try: dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device); logging.info("PyTorch DTCWTForward instance created.")
    except Exception as e: logging.critical(f"Failed to init DTCWTForward: {e}"); return

    # Имя входного файла
    input_base = f"watermarked_pytorch_hybrid_t{BCH_T}" # Согласуем имя
    input_video = input_base + INPUT_EXTENSION
    original_id = None

    # Загрузка оригинального ID
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r") as f: original_id = f.read().strip()
            assert original_id and len(original_id) == PAYLOAD_LEN_BYTES * 2; int(original_id, 16)
            logging.info(f"Original ID loaded: {original_id}")
        except Exception as e: logging.error(f"Read/Validate original ID failed: {e}"); original_id = None
    else: logging.warning(f"'{ORIGINAL_WATERMARK_FILE}' not found for comparison.")

    logging.info(f"--- Starting Extraction Main Process (PyTorch BN) ---")
    if not os.path.exists(input_video): logging.critical(f"Input video missing: '{input_video}'."); print(f"ERROR: Input missing: '{input_video}'"); return

    frames, fps_read = read_video(input_video)
    if not frames: logging.critical("Video read failed."); return
    logging.info(f"Read {len(frames)} frames.")

    # Вызов основной функции извлечения
    extracted_bytes = extract_watermark_from_video(
        frames=frames, nr=N_RINGS, nrtu=NUM_RINGS_TO_USE, bp=BITS_PER_PAIR,
        cps=CANDIDATE_POOL_SIZE, ec=EMBED_COMPONENT,
        expect_hybrid_ecc=True,     # Ожидаем гибридный режим
        max_expected_packets=15,    # Макс. пакетов
        ue=USE_ECC, bch_code=BCH_CODE_OBJECT,
        device=device, dtcwt_fwd=dtcwt_fwd,
        plb=PAYLOAD_LEN_BYTES, mw=MAX_WORKERS_EXTRACT
    )

    # Вывод результатов
    print(f"\n--- Extraction Results ---"); extracted_hex = None
    if extracted_bytes:
        if len(extracted_bytes) == PAYLOAD_LEN_BYTES: extracted_hex = extracted_bytes.hex(); print(f"  Payload Length OK."); print(f"  Decoded ID (Hex): {extracted_hex}"); logging.info(f"Decoded ID: {extracted_hex}")
        else: print(f"  ERROR: Decoded length mismatch! Got {len(extracted_bytes)}B, expected {PAYLOAD_LEN_BYTES}B."); logging.error(f"Decoded length mismatch!")
    else: print(f"  Extraction FAILED (No payload)."); logging.error("Extraction failed/No payload.")

    # Сравнение
    if original_id:
        print(f"  Original ID (Hex): {original_id}")
        if extracted_hex and extracted_hex == original_id: print("\n  >>> ID MATCH <<<"); logging.info("ID MATCH.")
        else: print("\n  >>> !!! ID MISMATCH or FAILED !!! <<<"); logging.warning("ID MISMATCH or Extraction Failed.")
    else: print("\n  Original ID unavailable for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_time_main = time.time() - start_time_main
    logging.info(f"--- Total Extractor Time: {total_time_main:.2f} sec ---")
    print(f"\nExtraction finished. Log: {LOG_FILENAME}")


# --- Точка Входа (__name__ == "__main__") ---
if __name__ == "__main__":
    # --- Инициализация и проверки ---
    if not PYTORCH_WAVELETS_AVAILABLE: print("ERROR: pytorch_wavelets required."); sys.exit(1)
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")
    # --- Профилирование и запуск main ---
    DO_PROFILING = False # Установите True для включения
    prof = None
    if DO_PROFILING: prof = cProfile.Profile(); prof.enable(); logging.info("cProfile enabled.")
    final_exit_code = 0
    try: main()
    except FileNotFoundError as e: print(f"\nERROR: {e}"); logging.error(f"{e}"); final_exit_code = 1
    except Exception as e: logging.critical(f"Unhandled main: {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. Log: {LOG_FILENAME}"); final_exit_code = 1
    finally:
        if DO_PROFILING and prof is not None:
            prof.disable(); stats = pstats.Stats(prof)
            print("\n--- Profiling Stats (Top 30 Cumulative Time) ---")
            try: stats.strip_dirs().sort_stats("cumulative").print_stats(30)
            except Exception as e_stats: print(f"Error printing stats: {e_stats}")
            print("-------------------------------------------------")
            pfile = f"profile_extract_pytorch_hybrid_t{BCH_T}.txt" # Имя файла профиля
            try:
                with open(pfile, "w", encoding='utf-8') as f: sf = pstats.Stats(prof, stream=f); sf.strip_dirs().sort_stats("cumulative").print_stats()
                logging.info(f"Profiling stats saved: {pfile}"); print(f"Profiling stats saved: {pfile}")
            except IOError as e: logging.error(f"Save profile failed: {e}")
            except Exception as e_prof_save: logging.error(f"Error saving profile: {e_prof_save}")
    sys.exit(final_exit_code)