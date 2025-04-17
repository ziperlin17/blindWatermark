# Файл: extractor.py (Версия: OpenCL Attempt, ThreadPool + Batches, Galois BCH, Syntax Fix V3)
import cv2
import numpy as np
import random
import logging
import time
import json
import os
# import imagehash # Не используется
import hashlib
from PIL import Image # Может быть не нужен, но оставим на всякий случай
from line_profiler.explicit_profiler import profile
from scipy.fftpack import dct # IDCT не нужен в экстракторе
from scipy.linalg import svd
import dtcwt # Импортируем dtcwt
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools
import concurrent.futures # Используем concurrent.futures
from concurrent.futures import ThreadPoolExecutor # Используем ThreadPoolExecutor
import uuid
from math import ceil
import cProfile
import pstats
from collections import Counter
import sys # Для восстановления бэкенда

# --- Переменная для хранения информации об успехе переключения бэкенда ---
DTCWT_OPENCL_ENABLED = True # Глобальный флаг

# --- Попытка импорта и инициализации Galois ---
try:
    import galois
    import numpy as np # Убедитесь, что numpy импортирован
    logging.info("galois: импортирован.")
    _test_bch_ok = False; _test_decode_ok = False; BCH_CODE_OBJECT = None
    try:
        _test_m = 8
        _test_t = 5 # Желаемое t
        _test_n = (1 << _test_m) - 1 # n = 255
        _test_d = 2 * _test_t + 1 # d = 11 (Вычисляем d из t)

        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        # --- Инициализируем через d ---
        _test_bch_galois = galois.BCH(_test_n, d=_test_d) # <--- ИСПОЛЬЗУЕМ d

        # --- ПРОВЕРКА ПАРАМЕТРОВ ПОЛУЧЕННОГО КОДА ---
        expected_k = 215
        if _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
             logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
             _test_bch_ok = True; BCH_CODE_OBJECT = _test_bch_galois
        else:
             logging.error(f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t={_test_bch_galois.t}, k={_test_bch_galois.k}.")
             _test_bch_ok = False; BCH_CODE_OBJECT = None

        # --- Тест decode ---
        if _test_bch_ok:
            _n_bits = _test_bch_galois.n; _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)
            GF2 = galois.GF(2); _dummy_cw_vec = GF2(_dummy_cw_bits)
            _msg, _flips = _test_bch_galois.decode(_dummy_cw_vec, errors=True)
            if _flips is not None: logging.info(f"galois: decode() test OK (flips={_flips})."); _test_decode_ok = True
            else: logging.info("galois: decode() test failed?"); _test_decode_ok = False
    except ValueError as ve:
         logging.error(f"galois: ОШИБКА ValueError при инициализации BCH(d={_test_d}): {ve}")
         BCH_CODE_OBJECT = None; _test_bch_ok = False
    except Exception as test_err:
         logging.info(f"galois: ОШИБКА теста инициализации/декодирования: {test_err}"); BCH_CODE_OBJECT = None; _test_bch_ok = False

    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None
    if GALOIS_AVAILABLE: logging.info("galois: Тесты пройдены.")
    else: logging.warning("galois: Тесты НЕ ПРОЙДЕНЫ. ECC будет отключен или работать некорректно.")

except ImportError: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; logging.info("galois library not found.")
except Exception as import_err: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; logging.info(f"galois: Ошибка импорта: {import_err}")

# --- Основные Параметры (Должны совпадать с эмбеддером) ---
LAMBDA_PARAM: float = 0.07 # Не используется в экстракторе напрямую, но для консистентности
ALPHA_MIN: float = 1.01 # Используется в compute_adaptive_alpha_entropy
ALPHA_MAX: float = 1.15  # Используется в compute_adaptive_alpha_entropy
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2 # Cb - ДОЛЖЕН СОВПАДАТЬ С ЭМБЕДДЕРОМ
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection' # Для логирования
PAYLOAD_LEN_BYTES: int = 8 # Ожидаемая длина ID
USE_ECC: bool = True # Ожидается ли ECC
BCH_M: int = 8
BCH_T: int = 5
MAX_PACKET_REPEATS: int = 5 # Для расчета макс. числа бит
FPS: int = 30 # Fallback
LOG_FILENAME: str = 'watermarking_extract_opencl_batched.log' # Новое имя
INPUT_EXTENSION: str = '.avi' # Ожидаемое расширение
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt' # Для сравнения
# Параметр для ThreadPoolExecutor
MAX_WORKERS_EXTRACT: Optional[int] = None # None -> Python выберет

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG) # Оставляем DEBUG для подробной информации

# --- Логирование Конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Извлечения (ThreadPool + Batches + OpenCL Attempt) ---")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit, Ожид. ECC: {USE_ECC} (Galois BCH m={BCH_M}, t={BCH_T}), Доступно/Работает: {GALOIS_AVAILABLE}")
logging.info(f"Ожид. Альфа для логирования: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS_EXTRACT or 'default'}) с батчингом.")
logging.info(f"DTCWT Бэкенд: Попытка использовать OpenCL (иначе NumPy).") # Добавлено
if USE_ECC and not GALOIS_AVAILABLE: logging.warning("ECC ожидается, но galois недоступна/не работает! Декодирование ECC невозможно.")
elif not USE_ECC: logging.info("ECC не ожидается.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error(f"NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE!")
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE != BITS_PER_PAIR.")

# --- Базовые Функции (должны быть идентичны эмбеддеру, кроме ненужных) ---
def dct_1d(s: np.ndarray) -> np.ndarray: return dct(s, type=2, norm='ortho')
# def idct_1d(...) # Не нужна в экстракторе

def dtcwt_transform(yp: np.ndarray, fn: int = -1) -> Optional[Pyramid]:
    """Применяет прямое DTCWT, используя текущий активный бэкенд dtcwt."""
    if not isinstance(yp, np.ndarray) or yp.ndim != 2: return None
    if np.any(np.isnan(yp)): logging.warning(f"[F:{fn}] Input NaN!")
    try:
        t = dtcwt.Transform2d()
        r, c = yp.shape; pr = r % 2 != 0; pc = c % 2 != 0
        ypp = np.pad(yp, ((0, pr), (0, pc)), mode='reflect') if pr or pc else yp
        py = t.forward(ypp.astype(np.float32), nlevels=1)
        if hasattr(py, 'lowpass') and py.lowpass is not None: py.padding_info = (pr, pc); return py
        else: return None
    except Exception as e: logging.error(f"[F:{fn}] DTCWT fwd err ({dtcwt.backend_name}): {e}"); return None

# def dtcwt_inverse(...) # Не нужна в экстракторе

@functools.lru_cache(maxsize=8) # Кэш для _ring_division_internal
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    # ... (Код _ring_division_internal без изменений) ...
    H, W = subband_shape;
    if H < 2 or W < 2: return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0; rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2); min_dist, max_dist = 0.0, np.max(distances)
    ring_bins = np.linspace(min_dist, max_dist + 1e-6, n_rings + 1) if max_dist >= 1e-6 else np.array([0., max_dist+1e-6])
    n_rings_eff = len(ring_bins)-1;
    if n_rings_eff <= 0: return [None] * n_rings
    ring_indices = np.digitize(distances, ring_bins) - 1; ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1); rc: List[Optional[np.ndarray]] = [None] * n_rings
    for rdx in range(n_rings_eff):
        coords = np.argwhere(ring_indices == rdx)
        if coords.shape[0] > 0:
            rc[rdx] = coords
    return rc

@functools.lru_cache(maxsize=8) # Кэш для вызова get_ring_coords_cached
def get_ring_coords_cached(ss: Tuple[int, int], nr: int) -> List[Optional[np.ndarray]]:
    return _ring_division_internal(ss, nr)
def ring_division(lp: np.ndarray, nr: int = N_RINGS, fn: int = -1) -> List[Optional[np.ndarray]]:
    """Разбивает 2D массив (lowpass подполосу) на N_RINGS концентрических колец."""
    if not isinstance(lp, np.ndarray) or lp.ndim != 2:
        logging.error(f"[Frame:{fn}] Invalid input for ring_division.")
        return [None] * nr

    # --- Определение sh ПЕРЕД try ---
    sh: Tuple[int, int] = lp.shape

    try:
        # Получаем координаты из кэша, используя sh
        cached_list = get_ring_coords_cached(sh, nr)

        # Проверка валидности кэша (можно оставить или убрать для краткости)
        if not isinstance(cached_list, list) or not all(isinstance(i, (np.ndarray, type(None))) for i in cached_list):
            logging.warning(f"[Frame:{fn}] Ring division cache returned invalid type, clearing and recalculating.")
            get_ring_coords_cached.cache_clear() # Очищаем кэш
            # Вызываем внутреннюю функцию напрямую для пересчета
            cached_list = _ring_division_internal(sh, nr)

        # Возвращаем копии массивов координат
        return [a.copy() if a is not None else None for a in cached_list]
    except Exception as e:
        logging.error(f"Ring division error Frame {fn}: {e}", exc_info=True) # Логируем ошибку
        return [None] * nr # Возвращаем список None при любой ошибке
def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    eps=1e-12; ve=0.; ee=0.
    if rv.size>0:
        rvc=np.clip(rv.copy(),0.,1.) # Работаем с копией
        h,_=np.histogram(rvc,bins=256,range=(0.,1.),density=False)
        tc=rvc.size
        if tc>0:
            p=h/tc;
            p=p[p>eps]
            if p.size>0:
                ve=-np.sum(p*np.log2(p))
                ee=-np.sum(p*np.exp(1.-p))
    # Возвращаем результат ПОСЛЕ всех вычислений
    return ve,ee
# Эта функция используется экстрактором для выбора колец (хотя само значение alpha не нужно для порога 1.0)
def compute_adaptive_alpha_entropy(rv: np.ndarray, ri: int, fn: int) -> float:
    if rv.size<10: return ALPHA_MIN
    ve,_=calculate_entropies(rv,fn,ri); lv=np.var(rv); en=np.clip(ve/MAX_THEORETICAL_ENTROPY,0.,1.); vmp=0.005; vsc=500
    tn=1./(1.+np.exp(-vsc*(lv-vmp))); we=.6; wt=.4; mf=np.clip((we*en+wt*tn),0.,1.)
    fa=ALPHA_MIN+(ALPHA_MAX-ALPHA_MIN)*mf; logging.debug(f"[F:{fn}, R:{ri}] Extractor Alpha Calc (for logging/consistency)={fa:.4f}"); return np.clip(fa,ALPHA_MIN,ALPHA_MAX)
def get_fixed_pseudo_random_rings(pi:int, nr:int, ps:int)->List[int]:
    if ps<=0:
        return []
    if ps>nr:
        ps=nr # Нельзя выбрать больше колец, чем есть

    sd=str(pi).encode('utf-8')
    hd=hashlib.sha256(sd).digest()
    sv=int.from_bytes(hd,'big')
    prng=random.Random(sv)
    try:
        ci=prng.sample(range(nr),ps)
    except ValueError: # Если вдруг nr < ps, хотя мы проверили выше
        ci=list(range(nr))

    logging.debug(f"[P:{pi}] Candidates: {ci}");
    return ci
# def calculate_perceptual_mask(...) # Не нужна в экстракторе
# def add_ecc(...) # Не нужна в экстракторе

def bits_to_bytes(bit_list: List[int]) -> Optional[bytes]:
    """Конвертирует список бит (0/1) в байты."""
    # Проверка на None удалена, предполагаем, что на вход идут только 0/1
    num_bits = len(bit_list)
    if num_bits % 8 != 0:
        # logging.warning(f"Длина битового списка ({num_bits}) не кратна 8. Дополняем нулями?")
        # Решаем НЕ дополнять, т.к. это может испортить данные ECC
        logging.error(f"Длина битового списка ({num_bits}) не кратна 8. Ошибка конвертации.")
        return None

    byte_array = bytearray()
    for i in range(0, num_bits, 8):
        byte_chunk = bit_list[i:i+8]
        try:
            byte_val = int("".join(map(str, byte_chunk)), 2)
            byte_array.append(byte_val)
        except ValueError: # Если в списке не 0/1
            logging.error(f"Невалидные символы в битовом списке при конвертации: {byte_chunk}")
            return None
    return bytes(byte_array) # Возвращаем неизменяемый bytes

def decode_ecc(packet_bits_list: List[int], bch_code: galois.BCH, expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    """Декодирует пакет бит с использованием Galois BCH."""
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

    try:
        # Конвертируем список бит в NumPy массив uint8
        packet_bits_np = np.array(packet_bits_list, dtype=np.uint8)

        # Конвертируем в FieldArray
        GF = bch_code.field
        received_vector = GF(packet_bits_np)

        # Декодируем
        try:
            # errors=True возвращает кортеж (сообщение, кол-во ошибок) или вызывает UncorrectableError
            corrected_message_vector, n_corrected_symbols = bch_code.decode(received_vector, errors=True)
            logging.info(f"Galois ECC: Декодировано, исправлено {n_corrected_symbols} ошибок.")

        except galois.errors.UncorrectableError:
            logging.warning("Galois ECC: Слишком много ошибок, не удалось декодировать пакет.")
            return None, -1 # Пакет неисправим

        # Извлекаем исправленные биты payload
        corrected_k_bits_np = corrected_message_vector.view(np.ndarray).astype(np.uint8)
        if corrected_k_bits_np.size < expected_payload_len_bits:
            logging.error(f"Decode ECC: Длина декодированного сообщения ({corrected_k_bits_np.size}) < ожидаемой ({expected_payload_len_bits}).")
            return None, -1

        # Берем только нужную часть (первые биты)
        corrected_payload_bits_np = corrected_k_bits_np[:expected_payload_len_bits]
        logging.debug(f"Decode ECC: Extracted {len(corrected_payload_bits_np)} payload bits.")

        # Конвертируем биты payload в байты
        corrected_payload_bytes = bits_to_bytes(corrected_payload_bits_np.tolist())

        if corrected_payload_bytes is None:
            logging.error("Decode ECC: Ошибка конвертации бит payload в байты.")
            return None, -1
        if len(corrected_payload_bytes) != expected_data_len_bytes:
             logging.error(f"Decode ECC: Неверная финальная длина payload ({len(corrected_payload_bytes)} байт), ожидалось {expected_data_len_bytes}.")
             return None, -1

        return corrected_payload_bytes, n_corrected_symbols

    except Exception as e:
        logging.error(f"Decode ECC: Неожиданная ошибка: {e}", exc_info=True)
        return None, -1


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

def extract_single_bit(f1:np.ndarray, f2:np.ndarray, ring_idx:int, n_rings:int, fn:int=0, embed_comp:int=EMBED_COMPONENT) -> Optional[int]:
    pair_index = fn // 2
    prefix = f"[МТ P:{pair_index}, R:{ring_idx}]" # Префикс
    try:
        if f1 is None or f2 is None: return None
        # Лог входных кадров (осторожно, может быть большим)
        # logging.debug(f"{prefix} f1 shape {f1.shape} f2 shape {f2.shape}")

        # Конвертация и DTCWT ВНУТРИ
        y1=cv2.cvtColor(f1,cv2.COLOR_BGR2YCrCb); y2=cv2.cvtColor(f2,cv2.COLOR_BGR2YCrCb)
        c1=y1[:,:,embed_comp].astype(np.float32)/255.; c2=y2[:,:,embed_comp].astype(np.float32)/255.
        p1=dtcwt_transform(c1,fn); p2=dtcwt_transform(c2,fn+1);
        if p1 is None or p2 is None or p1.lowpass is None or p2.lowpass is None: return None
        L1=np.array(p1.lowpass); L2=np.array(p2.lowpass)
        logging.debug(f"{prefix} === L1/L2 Calculated Once ===")

        logging.debug(f"{prefix} L1(in) s={L1.shape} m={np.mean(L1):.8e} v={np.var(L1):.8e} L1[0,0]={L1[0,0]:.8e}")
        logging.debug(f"{prefix} L2(in) s={L2.shape} m={np.mean(L2):.8e} v={np.var(L2):.8e} L2[0,0]={L2[0,0]:.8e}")

        # Кольца
        r1c=ring_division(L1,n_rings,fn); r2c=ring_division(L2,n_rings,fn+1);
        if not(0<=ring_idx<n_rings and ring_idx<len(r1c) and ring_idx<len(r2c)): return None
        cd1=r1c[ring_idx]; cd2=r2c[ring_idx];
        if cd1 is None or cd2 is None: return None

        # Значения
        rs1,cs1=cd1[:,0],cd1[:,1]; rv1=L1[rs1,cs1].astype(np.float32);
        rs2,cs2=cd2[:,0],cd2[:,1]; rv2=L2[rs2,cs2].astype(np.float32)
        if rv1.size==0 or rv2.size==0: return None
        min_s = min(rv1.size, rv2.size);
        if rv1.size!=rv2.size: rv1=rv1[:min_s]; rv2=rv2[:min_s]
        if min_s == 0: return None
        logging.debug(f"{prefix} rv1 s={rv1.shape} m={np.mean(rv1):.8f} data[:3]={np.array2string(rv1[:3], precision=8)}")
        logging.debug(f"{prefix} rv2 s={rv2.shape} m={np.mean(rv2):.8f} data[:3]={np.array2string(rv2[:3], precision=8)}")

        # DCT
        d1=dct_1d(rv1); d2=dct_1d(rv2);
        logging.debug(f"{prefix} d1[:3]={np.array2string(d1[:3], precision=8)}")
        logging.debug(f"{prefix} d2[:3]={np.array2string(d2[:3], precision=8)}")

        # SVD
        try: S1=svd(d1.reshape(-1,1),compute_uv=False); S2=svd(d2.reshape(-1,1),compute_uv=False)
        except np.linalg.LinAlgError: return None
        s1=S1[0] if S1.size>0 else 0.; s2=S2[0] if S2.size>0 else 0.;
        logging.debug(f"{prefix} s1={s1:.12e}, s2={s2:.12e}") # Повышенная точность

        # Решение
        eps=1e-12; threshold=1.0;
        ratio=s1/(s2+eps);
        extracted_bit = 0 if ratio >= threshold else 1;
        logging.debug(f"{prefix} ratio={ratio:.12e} -> Bit={extracted_bit}")

        return extracted_bit
    except Exception as e:
        logging.error(f"Extract single bit MT failed (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True) # Добавил exc_info
        return None
#

# --- Функция извлечения ОДНОГО бита из ОДНОГО кольца ---
# def extract_single_bit(L1: np.ndarray, L2: np.ndarray, ring_idx:int, n_rings:int, fn:int=0) -> Optional[int]:
#     """
#     Извлекает один бит из заданного кольца, используя ПРЕДВАРИТЕЛЬНО ВЫЧИСЛЕННЫЕ
#     lowpass компоненты L1 и L2.
#     """
#     pair_index = fn // 2
#     try:
#         # --- УБРАНЫ cvtColor и dtcwt_transform ---
#
#         prefix = f"[БН P:{pair_index}, R:{ring_idx}]" # Префикс для лога
#         logging.debug(f"{prefix} L1 Shape: {L1.shape}, L1[0,0]: {L1[0,0]:.6f}") # Лог входных L1/L2
#         logging.debug(f"{prefix} L2 Shape: {L2.shape}, L2[0,0]: {L2[0,0]:.6f}")
#
#         # Проверка типов L1, L2 (на всякий случай)
#         if not isinstance(L1, np.ndarray) or not isinstance(L2, np.ndarray) or L1.shape != L2.shape:
#             logging.warning(f"[P:{pair_index},R:{ring_idx}] Invalid L1/L2 provided.")
#             return None
#
#         # Кольца (вызывается на L1, L2)
#         # Важно: ring_division должна быть эффективной (с кэшем)
#         r1c = ring_division(L1, n_rings, fn)
#         r2c = ring_division(L2, n_rings, fn+1) # Для второго кадра тоже
#
#         if not(0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
#              logging.warning(f"[P:{pair_index},R:{ring_idx}] Invalid ring index.")
#              return None
#         cd1 = r1c[ring_idx]; cd2 = r2c[ring_idx]
#         if cd1 is None or cd2 is None:
#              logging.debug(f"[P:{pair_index},R:{ring_idx}] Ring coords are None.")
#              return None # Кольцо пустое или ошибка
#
#         # Значения, DCT, SVD
#         try:
#              rs1,cs1=cd1[:,0],cd1[:,1]; rv1=L1[rs1,cs1].astype(np.float32)
#              rs2,cs2=cd2[:,0],cd2[:,1]; rv2=L2[rs2,cs2].astype(np.float32)
#              logging.debug(f"{prefix} rv1[:5]: {np.array2string(rv1[:5], precision=6)}")  # Лог части данных
#
#         except IndexError:
#              logging.warning(f"[P:{pair_index},R:{ring_idx}] Index error getting ring values.")
#              return None
#
#         if rv1.size == 0 or rv2.size == 0:
#              logging.debug(f"[P:{pair_index},R:{ring_idx}] Ring values empty.")
#              return None
#
#         # Синхронизация размера (если вдруг отличаются)
#         min_s = min(rv1.size, rv2.size)
#         if rv1.size != rv2.size:
#             logging.warning(f"[P:{pair_index}, R:{ring_idx}] Ring value size mismatch ({rv1.size} vs {rv2.size}), using min {min_s}.")
#             rv1=rv1[:min_s]; rv2=rv2[:min_s]
#             if min_s == 0: return None
#
#         d1=dct_1d(rv1); d2=dct_1d(rv2)
#         try:
#              S1=svd(d1.reshape(-1,1), compute_uv=False); S2=svd(d2.reshape(-1,1), compute_uv=False)
#
#         except np.linalg.LinAlgError:
#              logging.warning(f"[P:{pair_index},R:{ring_idx}] SVD failed.")
#              return None
#
#         s1=S1[0] if S1.size>0 else 0.; s2=S2[0] if S2.size>0 else 0.
#
#         # Принятие решения (ПОРОГ = 1.0)
#         eps=1e-12; threshold=1.0
#         ratio=s1/(s2+eps)
#         extracted_bit = 0 if ratio >= threshold else 1
#         logging.debug(f"{prefix} s1={s1:.6f}, s2={s2:.6f}, ratio={ratio:.6f}") # Лог SVD/ratio
#         logging.debug(f"{prefix} -> Bit={extracted_bit}")
#
#         # Логирование (убрали вычисление alpha_for_log для скорости)
#         logging.debug(f"[P:{pair_index}, R:{ring_idx}] s1={s1:.4f}, s2={s2:.4f}, ratio={ratio:.4f} vs thr={threshold:.1f} -> Bit={extracted_bit}")
#
#         return extracted_bit
#     except Exception as e:
#         logging.error(f"Extract single bit failed (P:{pair_index}, R:{ring_idx}): {e}", exc_info=False)
#         return None


# --- Воркер для ОДНОЙ ПАРЫ (выбор колец + извлечение) ---

def _extract_single_pair_task(args: Dict[str, Any]) -> Tuple[int, List[Optional[int]]]:
    """
    Обрабатывает одну пару: выбирает кольца и вызывает ОРИГИНАЛЬНЫЙ extract_single_bit.
    (Версия с исправленным try-except в цикле энтропии)
    """
    pair_idx=args['pair_idx']; f1=args['frame1']; f2=args['frame2']; nr=args['n_rings']; nrtu=args['num_rings_to_use']; cps=args['candidate_pool_size']; ec=args['embed_component']; fn=2*pair_idx; final_rings=[]
    extracted_bits: List[Optional[int]] = [None] * nrtu
    try:
        # 1. Выбор колец (использует DTCWT ВНУТРИ себя для L1)
        cands = get_fixed_pseudo_random_rings(pair_idx, nr, cps);
        if len(cands) < nrtu: raise ValueError("Not enough candidates")

        # Блок try-except для всего процесса выбора колец
        try:
             # Вычисляем L1 ТОЛЬКО для выбора колец
             comp1_sel = f1[:,:,ec].astype(np.float32)/255.;
             p1_sel = dtcwt_transform(comp1_sel, fn);
             if p1_sel is None or p1_sel.lowpass is None: # Проверка результата DTCWT
                 raise RuntimeError("DTCWT L1 failed for selection")

             L1s = np.array(p1_sel.lowpass);
             coords = ring_division(L1s, nr, fn); # Разбиваем L1 на кольца
             entropies=[]
             min_pixels_for_entropy = 10

             for r_idx in cands:
                  e=-float('inf');
                  # Проверка индекса перед доступом к coords
                  if 0 <= r_idx < len(coords):
                       c=coords[r_idx];
                       if c is not None and c.shape[0]>=min_pixels_for_entropy:
                            # --- ИСПРАВЛЕННЫЙ блок try-except ---
                            try:
                                 rs,cs=c[:,0],c[:,1]; rv=L1s[rs,cs];
                                 se,_=calculate_entropies(rv,fn,r_idx);
                                 if np.isfinite(se):
                                      e=se
                            except Exception as entropy_e:
                                 # logging.warning(f"Entropy calc error P:{pair_idx} R:{r_idx}: {entropy_e}") # Опционально
                                 pass # Игнорируем ошибку, оставляем e = -inf
                            # --- Конец ИСПРАВЛЕННОГО блока ---
                  # Если индекс невалиден, e останется -inf
                  entropies.append((e,r_idx)) # Добавляем результат в любом случае

             entropies.sort(key=lambda x:x[0], reverse=True);
             final_rings=[idx for e,idx in entropies if e>-float('inf')][:nrtu]
             if len(final_rings) < nrtu: raise RuntimeError(f"Not enough valid rings selected {len(final_rings)}<{nrtu}")
             logging.info(f"[P:{pair_idx}] Selected rings for extraction: {final_rings}")

        except Exception as sel_err: # Ловим ошибки выбора колец
             logging.error(f"[P:{pair_idx}] Ring selection error: {sel_err}", exc_info=True)
             return pair_idx, extracted_bits # Возвращаем None биты

        # 2. Извлечение бит: вызываем ОРИГИНАЛЬНУЮ extract_single_bit, передавая f1, f2
        for i, ring_idx_to_extract in enumerate(final_rings):
            # Убедимся, что передаем корректные аргументы
            extracted_bits[i] = extract_single_bit(f1, f2, ring_idx_to_extract, nr, fn, ec)

        return pair_idx, extracted_bits

    except Exception as e: # Ловим ошибки на уровне всей задачи
        logging.error(f"Error in single pair task P:{pair_idx}: {e}", exc_info=True)
        return pair_idx, extracted_bits # Возвращаем None биты (или то, что успели извлечь)


# @profile # Добавляем профилировщик сюда
# def _extract_single_pair_task(args: Dict[str, Any]) -> Tuple[int, List[Optional[int]]]:
#     """
#     Обрабатывает одну пару: вычисляет DTCWT ОДИН РАЗ, выбирает кольца
#     и вызывает ИЗМЕНЕННУЮ extract_single_bit для извлечения бит.
#     (Версия с исправленным try-except в цикле энтропии)
#     """
#     pair_idx=args['pair_idx']; f1=args['frame1']; f2=args['frame2']; nr=args['n_rings']; nrtu=args['num_rings_to_use']; cps=args['candidate_pool_size']; ec=args['embed_component']; fn=2*pair_idx; final_rings=[]
#     extracted_bits: List[Optional[int]] = [None] * nrtu
#     try:
#         # --- Шаг 0: Преобразование цвета и DTCWT (делаем один раз) ---
#         try:
#             y1=cv2.cvtColor(f1,cv2.COLOR_BGR2YCrCb); y2=cv2.cvtColor(f2,cv2.COLOR_BGR2YCrCb)
#             comp1=y1[:,:,ec].astype(np.float32)/255.; comp2=y2[:,:,ec].astype(np.float32)/255.
#             p1 = dtcwt_transform(comp1, fn)
#             p2 = dtcwt_transform(comp2, fn+1)
#             if p1 is None or p2 is None or p1.lowpass is None or p2.lowpass is None:
#                  raise RuntimeError("DTCWT failed for one or both frames")
#             L1 = np.array(p1.lowpass); L2 = np.array(p2.lowpass)
#         except Exception as dt_err:
#              logging.error(f"[P:{pair_idx}] Initial Color/DTCWT error: {dt_err}")
#              return pair_idx, extracted_bits # Возвращаем None биты
#
#         # --- Шаг 1: Выбор колец (используя L1) ---
#         cands = get_fixed_pseudo_random_rings(pair_idx, nr, cps);
#         if len(cands) < nrtu: raise ValueError(f"Not enough candidates {len(cands)}<{nrtu}")
#         try: # Блок try для общего процесса выбора колец
#              coords = ring_division(L1, nr, fn);
#              if coords is None: raise RuntimeError("Ring division failed") # Проверка результата ring_division
#              entropies=[]
#              min_pixels_for_entropy = 10
#
#              for r_idx in cands:
#                   e = -float('inf')
#                   # Проверяем индекс и наличие координат
#                   if 0 <= r_idx < len(coords) and coords[r_idx] is not None:
#                       c = coords[r_idx]
#                       if c.shape[0] >= min_pixels_for_entropy:
#                            # --- Исправленный блок try-except ---
#                            try:
#                                 rs,cs=c[:,0],c[:,1]; rv=L1[rs,cs]
#                                 se,_=calculate_entropies(rv,fn,r_idx);
#                                 if np.isfinite(se):
#                                     e=se
#                            except Exception as entropy_e:
#                                 # logging.warning(f"[P:{pair_idx}, R:{r_idx}] Entropy calc failed: {entropy_e}")
#                                 pass # Игнорируем ошибку, e остается -inf
#                   # Добавляем результат (возможно, -inf) в список
#                   entropies.append((e,r_idx))
#              # --- Конец исправленного блока ---
#
#              entropies.sort(key=lambda x:x[0], reverse=True);
#              final_rings=[idx for e,idx in entropies if e>-float('inf')][:nrtu]
#              if len(final_rings) < nrtu: raise RuntimeError(f"Not enough valid rings selected {len(final_rings)}<{nrtu}")
#              logging.info(f"[P:{pair_idx}] Selected rings for extraction: {final_rings}")
#         except Exception as sel_err:
#              logging.error(f"[P:{pair_idx}] Ring selection error: {sel_err}", exc_info=True); # Добавил exc_info
#              return pair_idx, extracted_bits # Возвращаем None биты
#
#         # --- Шаг 2: Извлечение бит из выбранных колец, ПЕРЕДАВАЯ L1 и L2 ---
#         for i, ring_idx_to_extract in enumerate(final_rings):
#             # Вызываем ИЗМЕНЕННУЮ extract_single_bit
#             extracted_bits[i] = extract_single_bit(L1, L2, ring_idx_to_extract, nr, fn)
#
#         return pair_idx, extracted_bits
#
#     except Exception as e:
#         logging.error(f"Error in single pair task P:{pair_idx}: {e}", exc_info=True);
#         return pair_idx, extracted_bits # Возвращаем None биты в случае любой другой ошибки

# --- Воркер для обработки БАТЧА задач (ThreadPoolExecutor) ---
def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
    """Обрабатывает батч задач извлечения последовательно внутри одного ПОТОКА."""
    batch_results: Dict[int, List[Optional[int]]] = {} # Словарь {pair_idx: [bits]}
    for args in batch_args_list:
        pair_idx, bits = _extract_single_pair_task(args)
        batch_results[pair_idx] = bits
    return batch_results


# --- Основная функция извлечения (ThreadPool + Batches) ---
def extract_watermark_from_video(
        frames:List[np.ndarray], nr:int=N_RINGS, nrtu:int=NUM_RINGS_TO_USE, bp:int=BITS_PER_PAIR,
        cps:int=CANDIDATE_POOL_SIZE, ec:int=EMBED_COMPONENT, mpr:int=MAX_PACKET_REPEATS,
        ue:bool=USE_ECC, bm:int=BCH_M, bt:int=BCH_T, plb:int=PAYLOAD_LEN_BYTES,
        mw:Optional[int]=MAX_WORKERS_EXTRACT) -> Optional[bytes]:
    """Основная функция, управляющая процессом извлечения с использованием ThreadPoolExecutor и батчинга."""
    logging.info(f"Starting extraction (ThreadPool+Batches, Bits/Pair:{bp})")
    start_time=time.time(); nf=len(frames); total_pairs_available=nf//2; ppc=0; fpe=0; # Processed pairs count, failed pairs extract
    if total_pairs_available == 0: logging.error("No frame pairs to process."); return None

    # Определяем ожидаемую длину пакета
    payload_len_bits = plb * 8
    packet_len_expected = payload_len_bits # По умолчанию = длина payload
    ecc_enabled_and_valid = False
    bch_code_to_use = None

    if ue and GALOIS_AVAILABLE and BCH_CODE_OBJECT is not None:
        try:
            n = BCH_CODE_OBJECT.n; k = BCH_CODE_OBJECT.k; t_bch = BCH_CODE_OBJECT.t
            if payload_len_bits <= k:
                packet_len_expected = n # Ожидаем полный пакет ECC
                ecc_enabled_and_valid = True
                bch_code_to_use = BCH_CODE_OBJECT
                logging.info(f"Galois BCH OK: n={n}, k={k}, t={t_bch}. Expecting ECC packets ({packet_len_expected}b).")
            else: logging.warning(f"Payload size ({payload_len_bits}) > Galois k ({k}). ECC disabled.")
        except Exception as e: logging.error(f"Error getting Galois params: {e}. ECC disabled.")
    else: logging.info(f"ECC disabled or unavailable. Expecting raw payload ({packet_len_expected}b).")

    if packet_len_expected <= 0: logging.error("Invalid expected packet length."); return None

    # Определяем, сколько пар обрабатывать (исходя из max_repeats)
    pairs_to_process=min(total_pairs_available, ceil(mpr * packet_len_expected / bp));
    logging.info(f"Extracting from {pairs_to_process} pairs (max repeats:{mpr}).");
    if pairs_to_process == 0: logging.warning("Zero pairs to extract."); return None

    # --- Подготовка и запуск батчей ---
    all_pairs_args = []
    skipped_pairs = 0
    for pair_idx in range(pairs_to_process):
        i1=2*pair_idx; i2=i1+1;
        if i2>=nf or frames[i1] is None or frames[i2] is None: skipped_pairs+=1; continue
        args={'pair_idx':pair_idx, 'frame1':frames[i1], 'frame2':frames[i2], 'n_rings':nr, 'num_rings_to_use':nrtu, 'candidate_pool_size':cps, 'embed_component':ec}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args);
    if skipped_pairs > 0: logging.warning(f"Skipped {skipped_pairs} pairs preparation.")
    if num_valid_tasks == 0: logging.error("No valid extraction tasks."); return None

    num_workers = mw if mw is not None and mw>0 else (os.cpu_count() or 1)
    batch_size = max(1, ceil(num_valid_tasks / (num_workers * 2)) * 2)
    print(batch_size)
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i:i+batch_size] for i in range(0, num_valid_tasks, batch_size) if all_pairs_args[i:i+batch_size]]
    logging.info(f"Launching {num_batches} batches ({num_valid_tasks} pairs) using ThreadPool (mw={num_workers}, batch_size≈{batch_size})...")

    # Словарь для сбора результатов {pair_idx: [bits]}
    extracted_bits_map: Dict[int, List[Optional[int]]] = {}
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch_idx = {executor.submit(_extract_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                try:
                    batch_results_map = future.result() # Получаем словарь {pair_idx: [bits]}
                    extracted_bits_map.update(batch_results_map) # Обновляем общий словарь
                    ppc += len(batch_results_map) # Считаем обработанные пары
                    # Считаем пары с ошибками (где есть None в битах)
                    fpe += sum(1 for bits in batch_results_map.values() if bits is None or None in bits)
                except Exception as e:
                     logging.error(f"Batch {batch_idx} failed: {e}", exc_info=True)
                     # Считаем все пары из упавшего батча ошибками
                     fpe += len(batched_args_list[batch_idx])
    except Exception as e: logging.critical(f"ThreadPoolExecutor error: {e}", exc_info=True); return None

    logging.info(f"Extraction finished. Processed pairs:{ppc}. Pairs with errors/Nones:{fpe}.");
    if ppc == 0: logging.error("No pairs processed successfully."); return None

    # --- Сборка общего потока бит ---
    extracted_bits_all: List[Optional[int]] = []
    for pair_idx in range(pairs_to_process): # Итерируем по ожидаемым индексам пар
        bits = extracted_bits_map.get(pair_idx) # Получаем результат для этой пары
        if bits is not None:
            extracted_bits_all.extend(bits) # Добавляем биты
        else:
            # Если для пары нет результата (была пропущена или ошибка), добавляем None-заполнители
            extracted_bits_all.extend([None] * bp)

    total_bits_collected = len(extracted_bits_all)
    valid_bits = [b for b in extracted_bits_all if b is not None]
    error_rate = (total_bits_collected - len(valid_bits)) / total_bits_collected if total_bits_collected > 0 else 0
    logging.info(f"Total bits collected: {total_bits_collected}. Valid bits: {len(valid_bits)} ({100*(1-error_rate):.1f}% success rate).")

    if not valid_bits: logging.error("No valid bits extracted."); return None

    # --- Декодирование и Голосование ---
    # Используем только валидные биты
    num_potential_packets=len(valid_bits)//packet_len_expected if packet_len_expected > 0 else 0
    logging.info(f"Attempting to decode {num_potential_packets} potential packets ({packet_len_expected} bits each) from valid bits...");
    decoded_payloads: List[bytes] = []; decoded_success_count = 0; decode_failed_count = 0; ecc_corrected_symbols = 0

    for i in range(num_potential_packets):
        start_idx=i*packet_len_expected; end_idx=start_idx+packet_len_expected;
        if end_idx > len(valid_bits): break # Не хватает бит на полный пакет

        packet_candidate_bits = valid_bits[start_idx:end_idx]
        payload:Optional[bytes]=None; errors:int=-1

        if ecc_enabled_and_valid and bch_code_to_use is not None:
            payload, errors = decode_ecc(packet_candidate_bits, bch_code_to_use, plb)
        else: # Без ECC
             packet_bytes = bits_to_bytes(packet_candidate_bits)
             if packet_bytes is not None and len(packet_bytes) >= plb:
                 payload=packet_bytes[:plb]; errors=0 # Считаем 0 ошибок
             else: payload=None; errors=-1

        if payload is not None and len(payload)==plb:
            decoded_payloads.append(payload); decoded_success_count+=1;
            if errors > 0: ecc_corrected_symbols += errors
        else:
            decode_failed_count+=1

    logging.info(f"Decode summary: Success={decoded_success_count}, Failed={decode_failed_count}. Total ECC symbol corrections: {ecc_corrected_symbols}.");
    if not decoded_payloads: logging.error("No valid payloads decoded after ECC/processing."); return None

    # Голосование
    payload_counts=Counter(decoded_payloads); logging.info("Voting results:");
    for pld,c in payload_counts.most_common(5): logging.info(f"  ID {pld.hex()}: {c} votes")
    most_common_payload, winner_votes = payload_counts.most_common(1)[0];
    confidence=winner_votes/decoded_success_count if decoded_success_count > 0 else 0.
    logging.info(f"Winner selected: {most_common_payload.hex()} with {winner_votes}/{decoded_success_count} votes ({confidence:.1%}).");
    final_payload_bytes=most_common_payload;

    end_time=time.time(); logging.info(f"Extraction done. Total time: {end_time-start_time:.2f} sec.")
    return final_payload_bytes

# --- Основная Функция (main) ---
def main():
    global BCH_CODE_OBJECT, DTCWT_OPENCL_ENABLED # Используем флаг OpenCL

    start_time_main = time.time()
    backend_name_str = 'opencl' if DTCWT_OPENCL_ENABLED else 'numpy'
    print(backend_name_str)
    # Имя входного файла должно соответствовать выходному имени эмбеддера
    input_base = f"watermarked_galois_t4_{backend_name_str}_thr_batched"
    input_video = input_base + INPUT_EXTENSION
    original_id = None

    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE,"r") as f: original_id=f.read().strip();
            assert original_id and len(original_id)==PAYLOAD_LEN_BYTES*2, "Invalid original ID format"
            int(original_id, 16) # Проверка, что это hex
            logging.info(f"Original ID loaded: {original_id}")
        except Exception as e: logging.error(f"Read original ID failed: {e}"); original_id=None
    else: logging.warning(f"{ORIGINAL_WATERMARK_FILE} not found.")

    logging.info(f"--- Starting Extraction Main Process (ThreadPool + Batches + {backend_name_str.upper()} DTCWT) ---")
    if not os.path.exists(input_video): logging.critical(f"Input missing: '{input_video}'."); print(f"ERROR: Input missing."); return

    frames,fps = read_video(input_video);
    if not frames: logging.critical("Failed to read video."); return
    logging.info(f"Read {len(frames)} frames.")

    # Вызов основной функции извлечения
    extracted_bytes = extract_watermark_from_video(
        frames=frames, nr=N_RINGS, nrtu=NUM_RINGS_TO_USE, bp=BITS_PER_PAIR, cps=CANDIDATE_POOL_SIZE,
        ec=EMBED_COMPONENT, mpr=MAX_PACKET_REPEATS, ue=USE_ECC, bm=BCH_M, bt=BCH_T,
        plb=PAYLOAD_LEN_BYTES, mw=MAX_WORKERS_EXTRACT)

    print(f"\n--- Extraction Results ---"); extracted_hex=None
    if extracted_bytes:
        if len(extracted_bytes)==PAYLOAD_LEN_BYTES:
            extracted_hex=extracted_bytes.hex(); print(f"  Payload OK."); print(f"  Decoded ID (Hex): {extracted_hex}"); logging.info(f"Decoded ID: {extracted_hex}")
        else: print(f"  ERROR: Payload length mismatch! Got {len(extracted_bytes)}B."); logging.error(f"Payload length mismatch!")
    else: print(f"  Extraction FAILED."); logging.error("Extraction failed.")

    # Сравнение с оригиналом, если он есть
    if original_id:
        print(f"  Original ID (Hex): {original_id}")
        if extracted_hex and extracted_hex==original_id: print("\n  >>> ID MATCH <<<"); logging.info("ID MATCH.")
        else: print("\n  >>> !!! ID MISMATCH or FAILED !!! <<<"); logging.warning("ID MISMATCH.")
    else: print("\n  Original ID unavailable for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_time_main = time.time()-start_time_main; logging.info(f"--- Total Extractor Time: {total_time_main:.2f} sec ---")
    print(f"\nExtraction finished. Log: {LOG_FILENAME}")


# --- Точка Входа ---
if __name__ == "__main__":
    # --- Попытка переключить бэкенд DTCWT на OpenCL ---
    original_dtcwt_backend = 'numpy' # Значение по умолчанию
    DTCWT_OPENCL_ENABLED = False # Сбрасываем флаг перед проверкой

    try:
        # import dtcwt # Уже импортирован выше
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
                DTCWT_OPENCL_ENABLED = True # Устанавливаем флаг в True при успехе
            except Exception as e_init:
                 logging.warning(f"OpenCL backend test failed: {e_init}. Falling back to NumPy.", exc_info=False)
                 if dtcwt.backend_name == 'opencl': dtcwt.pop_backend() # Откатываем push
                 DTCWT_OPENCL_ENABLED = False # Оставляем False
        else:
            logging.warning("push_backend('opencl') did not change backend! Using NumPy.")
            DTCWT_OPENCL_ENABLED = False
            if dtcwt.backend_name != 'numpy': dtcwt.push_backend('numpy') # Ставим NumPy, если что-то странное
    except ImportError: logging.warning("dtcwt library not found."); DTCWT_OPENCL_ENABLED = False
    except ValueError as e_push: logging.warning(f"Failed switch to OpenCL: {e_push}. Using NumPy."); DTCWT_OPENCL_ENABLED = False
    except Exception as e_ocl: logging.warning(f"Error setting/testing OpenCL: {e_ocl}. Using NumPy.", exc_info=True); DTCWT_OPENCL_ENABLED = False
    # БЛОК finally ЗДЕСЬ НЕ НУЖЕН для проверки консистентности, она не так важна до вызова main

    # --- Логируем финальный активный бэкенд ПЕРЕД вызовом main ---
    try:
        final_backend_before_main = dtcwt.backend_name
        logging.info(f"Active DTCWT backend before calling main: {final_backend_before_main}")
        # Дополнительная проверка на всякий случай
        if final_backend_before_main == 'opencl' and not DTCWT_OPENCL_ENABLED:
            logging.error("Inconsistency detected before main: OpenCL active but flag is False! Forcing NumPy.")
            dtcwt.push_backend('numpy')
            DTCWT_OPENCL_ENABLED = False
        elif final_backend_before_main != 'opencl' and DTCWT_OPENCL_ENABLED:
            logging.error("Inconsistency detected before main: NumPy active but OpenCL flag is True! Resetting flag.")
            DTCWT_OPENCL_ENABLED = False
    except Exception as e_check:
         logging.error(f"Error checking backend before main: {e_check}")

    # --- Основной блок запуска ---
    if USE_ECC and not GALOIS_AVAILABLE: # Проверка Galois (если используется)
        print("\nWARNING: USE_ECC=True, but galois unavailable/failed. ECC decoding disabled.")

    profiler = cProfile.Profile(); profiler.enable()
    final_exit_code = 0 # Код возврата по умолчанию
    try:
        main() # Вызываем main С ТЕКУЩИМ АКТИВНЫМ БЭКЕНДОМ
        print(f"\n--- DTCWT Backend Used During Main: {'OpenCL (Forward Only)' if DTCWT_OPENCL_ENABLED else 'NumPy'} ---")
    except FileNotFoundError as e:
         print(f"\nERROR: Input file not found: {e}")
         logging.error(f"Input file not found: {e}", exc_info=True)
         final_exit_code = 1 # Устанавливаем код ошибки
    except Exception as e:
         logging.critical(f"Unhandled exception (Extractor): {e}", exc_info=True)
         print(f"\nCRITICAL ERROR: {e}. See log.")
         final_exit_code = 1 # Устанавливаем код ошибки
    finally:
        profiler.disable(); stats = pstats.Stats(profiler)
        print("\n--- Profiling Stats (Top 30 Cumulative Time) ---")
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        print("-------------------------------------------------")
        backend_str = 'opencl' if DTCWT_OPENCL_ENABLED else 'numpy'
        profile_file = f"profile_extract_{backend_str}_batched_galois_t{BCH_T}.txt"
        try:
            with open(profile_file, "w") as f: stats_file = pstats.Stats(profiler, stream=f); stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling stats saved: {profile_file}"); print(f"Profiling stats saved: {profile_file}")
        except IOError as e: logging.error(f"Could not save profiling stats: {e}")

        # --- Блок восстановления исходного бэкенда ПОСЛЕ ВСЕГО ---
        # --- Этот блок теперь не мешает работе main ---
        try:
            # Проверяем, что dtcwt был импортирован и original_dtcwt_backend существует
            if 'dtcwt' in sys.modules and 'original_dtcwt_backend' in locals() and dtcwt.backend_name != original_dtcwt_backend:
                logging.info(f"Attempting to restore original dtcwt backend: {original_dtcwt_backend}")
                # Восстанавливаем, пока не совпадет или стек не опустеет
                max_pops = 10 # Защита от бесконечного цикла
                pop_count = 0
                while dtcwt.backend_name != original_dtcwt_backend and len(getattr(dtcwt, '_backend_stack', [])) > 0 and pop_count < max_pops:
                     dtcwt.pop_backend()
                     pop_count += 1
                # Если все равно не совпало, ставим принудительно
                if dtcwt.backend_name != original_dtcwt_backend:
                     dtcwt.push_backend(original_dtcwt_backend)
                     # Очищаем лишний push, если он добавился
                     if hasattr(dtcwt,'_backend_stack') and len(dtcwt._backend_stack)>1 and dtcwt._backend_stack[0]==original_dtcwt_backend: dtcwt._backend_stack.pop(0);
                logging.info(f"DTCWT backend restored to: {dtcwt.backend_name}")
        except Exception as e_restore:
            logging.warning(f"Could not restore backend: {e_restore}")

    sys.exit(final_exit_code) # Выходим с кодом ошибки, если он был установлен
