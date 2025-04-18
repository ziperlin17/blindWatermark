# Файл: embedder_opencl_batched.py (Версия: ThreadPool + Batches + OpenCL Attempt - ИСПРАВЛЕННЫЙ)
import cv2
import numpy as np
import random
import logging
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
# import imagehash
import hashlib
from PIL import Image
from line_profiler.explicit_profiler import profile
from scipy.fftpack import dct, idct
from scipy.linalg import svd
import dtcwt
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools
import uuid
from math import ceil
import cProfile
import pstats

# --- Переменная для хранения информации об успехе переключения бэкенда ---
DTCWT_OPENCL_ENABLED = False # Глобальный флаг

# --- Попытка импорта и инициализации Galois ---
try:
    import galois
    logging.info("galois: импортирован.")
    _test_bch_ok = False; _test_decode_ok = False; BCH_CODE_OBJECT = None
    try:
        _test_m = 8
        _test_t = 5 # <--- Желаемое t
        _test_n = (1 << _test_m) - 1 # n = 255
        _test_d = 2 * _test_t + 1 # d = 11 (Вычисляем d из t)

        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        # --- Инициализируем через d ---
        _test_bch_galois = galois.BCH(_test_n, d=_test_d) # <--- ИСПОЛЬЗУЕМ d

        # --- ПРОВЕРКА ПАРАМЕТРОВ ПОЛУЧЕННОГО КОДА ---
        expected_k = 215 # Ожидаемое k для t=5
        # Проверяем, что ПОЛУЧЕННОЕ t совпадает с желаемым
        if _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
             logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
             _test_bch_ok = True; BCH_CODE_OBJECT = _test_bch_galois
        else:
             logging.error(f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t={_test_bch_galois.t}, k={_test_bch_galois.k}.")
             _test_bch_ok = False; BCH_CODE_OBJECT = None

        # --- Тест decode (остается таким же) ---
        if _test_bch_ok:
            _n_bits = _test_bch_galois.n; _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)
            GF2 = galois.GF(2); _dummy_cw_vec = GF2(_dummy_cw_bits)
            _msg, _flips = _test_bch_galois.decode(_dummy_cw_vec, errors=True)
            if _flips is not None: logging.info(f"galois: decode() test OK (flips={_flips})."); _test_decode_ok = True
            else: logging.warning("galois: decode() test failed?"); _test_decode_ok = False # Изменено на warning
    except ValueError as ve: # Ловим ошибки инициализации BCH
         logging.error(f"galois: ОШИБКА ValueError при инициализации BCH(d={_test_d}): {ve}")
         BCH_CODE_OBJECT = None; _test_bch_ok = False
    except Exception as test_err: # Ловим другие ошибки тестов
         logging.error(f"galois: ОШИБКА теста инициализации/декодирования: {test_err}", exc_info=True) # Добавлен exc_info
         BCH_CODE_OBJECT = None; _test_bch_ok = False

    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None
    if GALOIS_AVAILABLE: logging.info("galois: Тесты пройдены.")
    else: logging.warning("galois: Тесты НЕ ПРОЙДЕНЫ. ECC будет отключен или работать некорректно.")

except ImportError: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; logging.info("galois library not found.")
except Exception as import_err: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; logging.info(f"galois: Ошибка импорта: {import_err}")

# --- Основные Параметры ---
LAMBDA_PARAM: float = 0.05
ALPHA_MIN: float = 1.02
ALPHA_MAX: float = 1.21
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2 # Cb
USE_PERCEPTUAL_MASKING: bool = True
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection'
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 5
MAX_PACKET_REPEATS: int = 5
FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed_opencl_batched.log'
OUTPUT_CODEC: str = 'MJPG'
OUTPUT_EXTENSION: str = '.avi'
SELECTED_RINGS_FILE: str = 'selected_rings_embed_opencl_batched.json'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS: Optional[int] = None

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование Конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Встраивания (ThreadPool + Batches + OpenCL Attempt) ---")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Payload: {PAYLOAD_LEN_BYTES * 8}bit, ECC: {effective_use_ecc} (Galois BCH m={BCH_M}, t={BCH_T}), Max Repeats: {MAX_PACKET_REPEATS}")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(f"Маскировка: {USE_PERCEPTUAL_MASKING} (Lambda={LAMBDA_PARAM}), Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS or 'default'}) с батчингом.")
logging.info(f"DTCWT Бэкенд: Попытка использовать OpenCL (иначе NumPy).")
if USE_ECC and not GALOIS_AVAILABLE: logging.warning("ECC вкл, но galois недоступна/не работает! Встраивание БЕЗ ECC.")
elif not USE_ECC: logging.info("ECC выкл.")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG', 'DIVX']: logging.warning(f"Используется кодек с потерями '{OUTPUT_CODEC}'.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE:
    logging.error(f"NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE! Исправлено.")
    NUM_RINGS_TO_USE = CANDIDATE_POOL_SIZE
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE != BITS_PER_PAIR.")

# --- Базовые Функции ---
def dct_1d(s: np.ndarray) -> np.ndarray:
    return dct(s, type=2, norm='ortho')

def idct_1d(c: np.ndarray) -> np.ndarray:
    return idct(c, type=2, norm='ortho')

def dtcwt_transform(yp: np.ndarray, fn: int = -1) -> Optional[Pyramid]:
    if not isinstance(yp, np.ndarray) or yp.ndim != 2:
         logging.error(f"[Frame:{fn}] Invalid input type/dims for dtcwt_transform: {type(yp)}, {yp.ndim if hasattr(yp, 'ndim') else 'N/A'}")
         return None
    if np.any(np.isnan(yp)): logging.warning(f"[Frame:{fn}] Input DTCWT contains NaN!")
    try:
        t = dtcwt.Transform2d()
        r, c = yp.shape; pr = r % 2 != 0; pc = c % 2 != 0
        ypp = np.pad(yp, ((0, pr), (0, pc)), mode='reflect') if pr or pc else yp
        py = t.forward(ypp.astype(np.float32), nlevels=1)
        if hasattr(py, 'lowpass') and py.lowpass is not None:
             py.padding_info = (pr, pc); return py
        else: logging.error(f"[Frame:{fn}] DTCWT forward did not return lowpass."); return None
    except Exception as e: logging.error(f"[Frame:{fn}] DTCWT forward error (backend: {dtcwt.backend_name}): {e}", exc_info=True); return None

def dtcwt_inverse(py: Pyramid, fn: int = -1) -> Optional[np.ndarray]:
    """
    Применяет обратное DTCWT, используя текущий активный бэкенд dtcwt.
    (Версия для работы с пирамидой с highpasses=() )
    """
    logging.debug(f"[F:{fn}] Entering dtcwt_inverse. Pyramid type: {type(py)}. Backend: {dtcwt.backend_name}")

    if not isinstance(py, dtcwt.Pyramid):
         logging.error(f"[F:{fn}] Input is not a dtcwt.Pyramid based object.")
         return None
    if not hasattr(py, 'lowpass') or py.lowpass is None:
         logging.error(f"[F:{fn}] Input pyramid missing 'lowpass' or it is None.")
         return None

    lp_shape = getattr(py.lowpass, 'shape', 'N/A')
    lp_dtype = getattr(py.lowpass, 'dtype', 'N/A')
    lp_is_np = isinstance(py.lowpass, np.ndarray)
    logging.debug(f"[F:{fn}] Input Lowpass shape: {lp_shape}, dtype: {lp_dtype}, Is NumPy: {lp_is_np}")

    hp_info = "None or Empty"
    hp_level_count = 0
    if hasattr(py, 'highpasses') and py.highpasses is not None:
         try: hp_level_count = len(py.highpasses); hp_info = f"Tuple len={hp_level_count}"
         except TypeError: hp_info = f"Not iterable (type={type(py.highpasses)})"
    logging.debug(f"[F:{fn}] Input Highpasses info: {hp_info}")

    try:
        t = dtcwt.Transform2d()
        py_processed = py

        # Попытка конвертации в NumPy Pyramid, если нужно
        if not isinstance(py_processed, dtcwt.numpy.Pyramid):
            logging.debug(f"[F:{fn}] Input not numpy.Pyramid. Attempting conversion...")
            try:
                lp_np = np.array(py_processed.lowpass).copy()
                # Создаем ПУСТОЙ кортеж для highpasses, так как мы передаем highpasses=()
                hp_np = ()
                sc_np = tuple(np.array(s).copy() for s in py_processed.scales) if hasattr(py_processed, 'scales') and py_processed.scales is not None else None
                py_processed = dtcwt.numpy.Pyramid(lp_np, hp_np, scales=sc_np)
                if hasattr(py, 'padding_info'): setattr(py_processed, 'padding_info', getattr(py, 'padding_info'))
                logging.info(f"[F:{fn}] Converted to numpy.Pyramid (empty highpasses).")
            except Exception as e_conv:
                 logging.warning(f"[F:{fn}] Failed to convert: {e_conv}. Using original type {type(py)}.", exc_info=True)

        # Вызов t.inverse() БЕЗ gain_mask
        logging.debug(f"[F:{fn}] Calling t.inverse() with pyramid (type: {type(py_processed)})...")
        start_inverse = time.perf_counter()
        rp = t.inverse(py_processed) # Передаем пирамиду с пустым highpasses=()
        end_inverse = time.perf_counter()
        logging.debug(f"[F:{fn}] t.inverse() call finished in {end_inverse - start_inverse:.4f} seconds.")

        # Проверка результата
        if rp is None: logging.error(f"[F:{fn}] t.inverse() returned None!"); return None
        if not isinstance(rp, np.ndarray): rp = np.array(rp)
        rp = rp.astype(np.float32); logging.debug(f"[F:{fn}] Result shape: {rp.shape}, dtype: {rp.dtype}")

        # Обработка паддинга
        pr, pc = getattr(py_processed, 'padding_info', (False, False)); logging.debug(f"[F:{fn}] Padding: pr={pr}, pc={pc}")
        rows_rp, cols_rp = rp.shape; end_row = rows_rp - pr if pr else rows_rp; end_col = cols_rp - pc if pc else cols_rp; logging.debug(f"[F:{fn}] Target shape: ({end_row}, {end_col})")
        if end_row < 0 or end_col < 0 or end_row > rows_rp or end_col > cols_rp: logging.error(f"[F:{fn}] Invalid target shape"); return None
        ry = rp[:end_row, :end_col].copy(); logging.debug(f"[F:{fn}] Result after unpadding shape: {ry.shape}")
        if np.any(np.isnan(ry)): logging.warning(f"[F:{fn}] NaN after inverse!")
        return ry

    except ValueError as ve: logging.error(f"[F:{fn}] ValueError during DTCWT inverse: {ve}", exc_info=True); return None
    except IndexError as ie: logging.error(f"[F:{fn}] IndexError during DTCWT inverse (accessing empty Yh?): {ie}", exc_info=True); return None
    except Exception as e: logging.error(f"[F:{fn}] DTCWT inverse error (backend: {dtcwt.backend_name}): {e}", exc_info=True); return None


@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    H, W = subband_shape
    if H < 2 or W < 2: return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2)
    min_dist, max_dist = 0.0, np.max(distances)
    ring_bins = np.linspace(min_dist, max_dist + 1e-6, n_rings + 1) if max_dist >= 1e-6 else np.array([0., max_dist+1e-6])
    n_rings_eff = len(ring_bins)-1
    if n_rings_eff <= 0: return [None] * n_rings
    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    rc: List[Optional[np.ndarray]] = [None] * n_rings
    for rdx in range(n_rings_eff):
        coords = np.argwhere(ring_indices == rdx)
        if coords.shape[0] > 0:
            rc[rdx] = coords
    return rc

@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(ss: Tuple[int, int], nr: int) -> List[Optional[np.ndarray]]:
    return _ring_division_internal(ss, nr)

def ring_division(lp: np.ndarray, nr: int = N_RINGS, fn: int = -1) -> List[Optional[np.ndarray]]:
    if not isinstance(lp, np.ndarray) or lp.ndim != 2:
        logging.error(f"[Frame:{fn}] Invalid input for ring_division.")
        return [None] * nr
    sh=lp.shape
    try:
        cached_list = get_ring_coords_cached(sh, nr)
        # Возвращаем копии
        return [a.copy() if a is not None else None for a in cached_list]
    except Exception as e:
        logging.error(f"Ring division error Frame {fn}: {e}", exc_info=True)
        return [None] * nr

def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    eps=1e-12; ve=0.; ee=0.
    if rv.size>0:
        # Убедимся, что работаем с копией, чтобы clip не менял исходные данные
        rv_processed = rv.copy()
        min_val, max_val = np.min(rv_processed), np.max(rv_processed)
        if min_val < 0.0 or max_val > 1.0:
             rv_processed = np.clip(rv_processed, 0.0, 1.0)
        h,_=np.histogram(rv_processed, bins=256, range=(0., 1.), density=False)
        tc=rv_processed.size
        if tc>0:
             p=h/tc; p=p[p>eps]
             if p.size>0:
                  ve=-np.sum(p*np.log2(p))
                  ee=-np.sum(p*np.exp(1.-p))
    return ve,ee

def compute_adaptive_alpha_entropy(rv: np.ndarray, ri: int, fn: int) -> float:
    if rv.size<10: return ALPHA_MIN
    ve,_=calculate_entropies(rv,fn,ri); lv=np.var(rv); en=np.clip(ve/MAX_THEORETICAL_ENTROPY,0.,1.); vmp=0.005; vsc=500
    tn=1./(1.+np.exp(-vsc*(lv-vmp))); we=.6; wt=.4; mf=np.clip((we*en+wt*tn),0.,1.)
    fa=ALPHA_MIN+(ALPHA_MAX-ALPHA_MIN)*mf; logging.debug(f"[F:{fn}, R:{ri}] Alpha={fa:.4f} (E={ve:.3f},V={lv:.6f})")
    return np.clip(fa,ALPHA_MIN,ALPHA_MAX)

def get_fixed_pseudo_random_rings(pi:int, nr:int, ps:int)->List[int]:
    # Эта функция выглядела синтаксически правильно, разбиение на строки для читаемости
    if ps<=0: return []
    if ps>nr: ps=nr
    sd=str(pi).encode('utf-8')
    hd=hashlib.sha256(sd).digest()
    sv=int.from_bytes(hd,'big')
    prng=random.Random(sv)
    try:
        ci=prng.sample(range(nr),ps)
    except ValueError: # Если nr < ps (не должно случаться из-за проверки выше)
        ci=list(range(nr))
    logging.debug(f"[P:{pi}] Candidates: {ci}");
    return ci

def calculate_perceptual_mask(ip: np.ndarray, fn: int = -1) -> Optional[np.ndarray]:
    if not isinstance(ip, np.ndarray) or ip.ndim != 2: return None;
    try:
        pf=ip.astype(np.float32); gx=cv2.Sobel(pf,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(pf,cv2.CV_32F,0,1,ksize=3)
        gm=np.sqrt(gx**2+gy**2); ks=(11,11); s=5; lm=cv2.GaussianBlur(pf,ks,s); lms=cv2.GaussianBlur(pf**2,ks,s)
        lv=np.sqrt(np.maximum(lms-lm**2,0)); cm=np.maximum(gm,lv); eps=1e-9; mc=np.max(cm)
        mn=cm/(mc+eps) if mc>eps else np.zeros_like(cm);
        # logging.debug(f"Mask F{fn}: range {np.min(mn):.2f}-{np.max(mn):.2f}") # Можно раскомментировать
        return np.clip(mn,0.,1.).astype(np.float32)
    except Exception as e: logging.error(f"Mask error F{fn}: {e}"); return np.ones_like(ip, dtype=np.float32)

def add_ecc(data_bits: np.ndarray, bch_code: galois.BCH) -> Optional[np.ndarray]:
    # Исправленная версия с try-except
    if not GALOIS_AVAILABLE or bch_code is None: return data_bits;
    k=bch_code.k; n=bch_code.n;
    if data_bits.size > k: logging.error(f"Data {data_bits.size} > k {k}"); return None;
    pad_len = k-data_bits.size;
    msg_bits=np.pad(data_bits,(0,pad_len),'constant') if pad_len>0 else data_bits.astype(np.uint8); # Убедимся в типе
    try: # Добавляем try
        GF=bch_code.field; msg_vec=GF(msg_bits); cw_vec=bch_code.encode(msg_vec);
        pkt_bits=cw_vec.view(np.ndarray).astype(np.uint8);
        assert pkt_bits.size == n, f"Galois size {pkt_bits.size}!=n {n}"; # Используем assert
        logging.info(f"Galois ECC: Data({data_bits.size}b->{k}b) -> Packet({pkt_bits.size}b).")
        return pkt_bits
    except Exception as e: # Добавляем except
        logging.error(f"Galois encode error: {e}", exc_info=True); # Добавил exc_info
        return None # Возвращаем None при ошибке

def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # Исправленная версия с правильной структурой try-except-finally
    logging.info(f"Reading: {video_path}"); frames: List[np.ndarray] = []; fps = float(FPS); cap = None; h, w = -1, -1;
    try:
        assert os.path.exists(video_path), f"Not found: {video_path}";
        cap = cv2.VideoCapture(video_path);
        assert cap.isOpened(), f"Cannot open {video_path}";
        fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS);
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
        logging.info(f"Props: {w}x{h}@{fps:.2f}~{fc}f");
        rc,nc,ic=0,0,0;
        while True:
            ret,f=cap.read();
            if not ret:
                break # Выход из цикла while
            if f is None:
                nc+=1;
                continue # К следующей итерации while
            if f.ndim==3 and f.shape[:2]==(h,w) and f.dtype==np.uint8:
                frames.append(f); rc+=1;
            else:
                ic+=1; logging.warning(f"Skipped invalid frame #{rc+nc+ic}");
        logging.info(f"Read loop finished. V:{rc},N:{nc},I:{ic}");
        assert rc>0, "No valid frames read."
    except Exception as e:
        logging.error(f"Read error: {e}", exc_info=True);
        frames=[] # Очищаем кадры при ошибке
        # fps остается значением по умолчанию или тем, что успели прочитать
    finally:
        # Блок finally выполняется всегда
        if cap and cap.isOpened(): # Проверяем, что cap был создан и все еще открыт
            logging.debug("Releasing video capture resource.")
            cap.release()
    # Return должен быть после finally, но внутри функции
    return frames, fps

def write_video(frames: List[np.ndarray], out_path: str, fps: float, codec: str = OUTPUT_CODEC):
    # Исправленная версия с правильной структурой try-except-finally
    if not frames: logging.error("No frames"); return;
    logging.info(f"Writing: {out_path}(FPS:{fps:.2f},Codec:{codec})");
    writer = None; # Инициализируем writer как None
    try:
        fv = next((f for f in frames if f is not None and f.ndim==3), None);
        assert fv is not None, "No valid frames";
        h,w,_=fv.shape; fourcc=cv2.VideoWriter_fourcc(*codec); base,_=os.path.splitext(out_path);
        out_path=base+OUTPUT_EXTENSION; writer=cv2.VideoWriter(out_path, fourcc, fps, (w,h)); wc=codec;
        if not writer.isOpened(): # Отступ правильный
            logging.error(f"Fail codec {codec}");
            fbk='MJPG';
            if OUTPUT_EXTENSION.lower()=='.avi' and codec.upper()!=fbk: # Отступ правильный
                 logging.warning(f"Fallback {fbk}");
                 fourcc_fbk=cv2.VideoWriter_fourcc(*fbk); # Переменная должна быть другая
                 writer=cv2.VideoWriter(out_path,fourcc_fbk,fps,(w,h)); wc=fbk;
                 assert writer.isOpened(), "Writer failed even with fallback";
            else: # Если не AVI или fallback не нужен
                 assert writer.isOpened(), "Writer failed"; # Просто проверяем исходную ошибку

        # Этот блок теперь правильно внутри try
        wc_ok,sc_ok=0,0; bf=np.zeros((h,w,3),dtype=np.uint8);
        for i,f in enumerate(frames):
            if f is not None and f.shape[:2]==(h,w) and f.dtype==np.uint8:
                writer.write(f); wc_ok+=1;
            else:
                writer.write(bf); sc_ok+=1; logging.warning(f"Skip invalid frame {i}");
        logging.info(f"Write done({wc}). W:{wc_ok},S:{sc_ok}");
    except Exception as e:
        logging.error(f"Write error: {e}", exc_info=True)
    finally:
        # Блок finally выполняется всегда
        if writer and writer.isOpened(): # Проверяем, что writer был создан и открыт
             logging.debug("Releasing video writer resource.")
             writer.release()

# --- Функция встраивания в пару кадров (embed_frame_pair) ---
# --- (Копипаста из предыдущего ответа, она была корректна) ---
def embed_frame_pair(
        frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bits: List[int],
        selected_ring_indices: List[int], n_rings: int, frame_number: int,
        use_perceptual_masking: bool, embed_component: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Встраивает биты в выбранные кольца пары кадров.
    (Версия с ручным upsampling lowpass перед вызовом inverse для обхода бага dtcwt)
    """
    pair_index = frame_number // 2
    if len(bits) != len(selected_ring_indices):
        logging.error(f"[P:{pair_index}] Mismatch bits/rings.")
        return None, None
    if not bits: return frame1_bgr, frame2_bgr

    try:
        # 1. Проверка и YCrCb
        if frame1_bgr is None or frame2_bgr is None: return None, None
        f1_ycrcb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb); f2_ycrcb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2YCrCb)
        comp1 = f1_ycrcb[:, :, embed_component].astype(np.float32)/255.0; comp2 = f2_ycrcb[:, :, embed_component].astype(np.float32)/255.0
        Y1,Cr1,Cb1 = f1_ycrcb[:,:,0],f1_ycrcb[:,:,1],f1_ycrcb[:,:,2]; Y2,Cr2,Cb2 = f2_ycrcb[:,:,0],f2_ycrcb[:,:,1],f2_ycrcb[:,:,2]

        # 2. Прямое DTCWT (nlevels=1)
        pyramid1_orig = dtcwt_transform(comp1, frame_number); pyramid2_orig = dtcwt_transform(comp2, frame_number + 1)
        if pyramid1_orig is None or pyramid2_orig is None or pyramid1_orig.lowpass is None or pyramid2_orig.lowpass is None: return None, None

        # 3. Извлечение и модификация Lowpass
        L1 = np.array(pyramid1_orig.lowpass).copy(); L2 = np.array(pyramid2_orig.lowpass).copy()
        padding_info1 = getattr(pyramid1_orig, 'padding_info', (False, False))
        padding_info2 = getattr(pyramid2_orig, 'padding_info', (False, False))

        # --- Код модификации L1, L2 (цикл по кольцам, DCT, SVD, IDCT, маска) ---
        ring_coords1 = ring_division(L1, n_rings, frame_number); ring_coords2 = ring_division(L2, n_rings, frame_number + 1)
        perceptual_mask = None
        if use_perceptual_masking: mask_full_res = calculate_perceptual_mask(comp1, frame_number);
        if mask_full_res is not None: perceptual_mask = cv2.resize(mask_full_res, (L1.shape[1], L1.shape[0])) if mask_full_res.shape != L1.shape else mask_full_res.astype(np.float32);
        else: perceptual_mask = np.ones_like(L1)
        modifications_count = 0 # Счетчик модификаций
        for ring_idx, bit in zip(selected_ring_indices, bits):
            if not (0 <= ring_idx < n_rings and ring_idx < len(ring_coords1) and ring_idx < len(ring_coords2)): continue
            coords1 = ring_coords1[ring_idx]; coords2 = ring_coords2[ring_idx]
            if coords1 is None or coords2 is None: continue
            rows1, cols1 = coords1[:,0], coords1[:,1]; rows2, cols2 = coords2[:,0], coords2[:,1]
            v1 = L1[rows1, cols1].astype(np.float32); v2 = L2[rows2, cols2].astype(np.float32)
            if v1.size == 0 or v2.size == 0: continue
            min_s = min(v1.size, v2.size);
            if v1.size != v2.size:
                v1,v2 = v1[:min_s], v2[:min_s]; rows1,cols1 = rows1[:min_s],cols1[:min_s]; rows2,cols2 = rows2[:min_s],cols2[:min_s]
                if min_s == 0: continue
            alpha = compute_adaptive_alpha_entropy(v1, ring_idx, frame_number); d1=dct_1d(v1); d2=dct_1d(v2)
            try: U1, S1v, Vt1 = svd(d1.reshape(-1,1), False); U2, S2v, Vt2 = svd(d2.reshape(-1,1), False)
            except np.linalg.LinAlgError: continue
            s1 = S1v[0] if S1v.size>0 else 0.; s2 = S2v[0] if S2v.size>0 else 0.;
            eps=1e-12; ratio = s1/(s2+eps); ns1,ns2=s1,s2; modified=False; a2=alpha*alpha; inv_a=1/(alpha+eps)
            if bit==0:
                if ratio<alpha: ns1=(s1*a2+alpha*s2)/(a2+1); ns2=(alpha*s1+s2)/(a2+1); modified=True
            else:
                if ratio>=inv_a: ns1=(s1+alpha*s2)/(1+a2); ns2=(alpha*s1+a2*s2)/(1+a2); modified=True
            if modified:
                modifications_count+=1
                try:
                    vt1_val=Vt1[0,0] if Vt1.shape==(1,1) else 1.0; vt2_val=Vt2[0,0] if Vt2.shape==(1,1) else 1.0
                    d1m = (U1 * ns1 * vt1_val).flatten(); d2m = (U2 * ns2 * vt2_val).flatten()
                    v1m=idct_1d(d1m); v2m=idct_1d(d2m)
                except Exception as recon_err: logging.warning(f"[P:{pair_index},R:{ring_idx}] Recon err: {recon_err}"); continue
                if v1m.size != v1.size or v2m.size != v2.size: continue
                delta1=v1m-v1; delta2=v2m-v2; mf1=np.ones_like(delta1); mf2=np.ones_like(delta2)
                if use_perceptual_masking and perceptual_mask is not None:
                    try: mv1=perceptual_mask[rows1,cols1]; mv2=perceptual_mask[rows2,cols2]; mf1*=(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mv1); mf2*=(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mv2)
                    except Exception as mask_err: logging.warning(f"Mask apply error P:{pair_index} R:{ring_idx}: {mask_err}")
                try: L1[rows1,cols1] += delta1*mf1; L2[rows2,cols2] += delta2*mf2
                except IndexError: logging.warning(f"Delta apply index error P:{pair_index} R:{ring_idx}"); continue
        # --- Конец цикла модификации ---

        # --- Обход бага dtcwt.inverse для nlevels=1 ---
        # 4. Ручной Upsampling L1 и L2
        rows_L, cols_L = L1.shape
        target_shape_up = (rows_L * 2, cols_L * 2)
        try:
             L1_up = cv2.resize(L1, (target_shape_up[1], target_shape_up[0]), interpolation=cv2.INTER_LINEAR)
             L2_up = cv2.resize(L2, (target_shape_up[1], target_shape_up[0]), interpolation=cv2.INTER_LINEAR)
             logging.debug(f"[P:{pair_index}] Manually upsampled L1/L2 to {L1_up.shape}")
        except Exception as e_resize:
             logging.error(f"[P:{pair_index}] Failed to upsample L1/L2: {e_resize}")
             return None, None

        # 5. Создание НУЛЕВОГО highpass УВЕЛИЧЕННОГО размера
        zeros_highpass_up = np.zeros(target_shape_up + (6,), dtype=np.complex64)

        # 6. Создание пирамиды для inverse с УВЕЛИЧЕННЫМИ lowpass и highpass
        pyramid1_for_inverse = dtcwt.Pyramid(L1_up, highpasses=(zeros_highpass_up,))
        pyramid2_for_inverse = dtcwt.Pyramid(L2_up, highpasses=(zeros_highpass_up.copy(),))
        # НЕ устанавливаем padding_info для этой временной пирамиды

        # 7. Вызов dtcwt_inverse (БЕЗ gain_mask)
        c1m = dtcwt_inverse(pyramid1_for_inverse, frame_number)
        c2m = dtcwt_inverse(pyramid2_for_inverse, frame_number + 1)
        # --- Конец обхода бага ---

        if c1m is None or c2m is None:
             logging.error(f"[P:{pair_index}] DTCWT inverse failed after upsampling trick.")
             return None, None

        # 8. Удаление паддинга из результата inverse, используя ОРИГИНАЛЬНЫЙ padding_info
        try:
            pr1, pc1 = padding_info1; pr2, pc2 = padding_info2
            rows_c1m, cols_c1m = c1m.shape; end_row1 = rows_c1m - pr1 if pr1 else rows_c1m; end_col1 = cols_c1m - pc1 if pc1 else cols_c1m
            rows_c2m, cols_c2m = c2m.shape; end_row2 = rows_c2m - pr2 if pr2 else rows_c2m; end_col2 = cols_c2m - pc2 if pc2 else cols_c2m
            if end_row1 < 0 or end_col1 < 0 or end_row1 > rows_c1m or end_col1 > cols_c1m: raise ValueError("Invalid unpadding size c1m")
            if end_row2 < 0 or end_col2 < 0 or end_row2 > rows_c2m or end_col2 > cols_c2m: raise ValueError("Invalid unpadding size c2m")
            c1m_unpadded = c1m[:end_row1, :end_col1]
            c2m_unpadded = c2m[:end_row2, :end_col2]
            logging.debug(f"[P:{pair_index}] Unpadded inverse results to {c1m_unpadded.shape}, {c2m_unpadded.shape}")
        except Exception as e_unpad:
             logging.error(f"[P:{pair_index}] Error during unpadding: {e_unpad}")
             return None, None

        # 9. Постобработка и сборка кадра
        tsh=(Y1.shape[0],Y1.shape[1]);
        if c1m_unpadded.shape != tsh: c1m_final = cv2.resize(c1m_unpadded, (tsh[1], tsh[0]), interpolation=cv2.INTER_LINEAR)
        else: c1m_final = c1m_unpadded
        if c2m_unpadded.shape != tsh: c2m_final = cv2.resize(c2m_unpadded, (tsh[1], tsh[0]), interpolation=cv2.INTER_LINEAR)
        else: c2m_final = c2m_unpadded
        c1s = np.clip(c1m_final * 255.0, 0, 255).astype(np.uint8); c2s = np.clip(c2m_final * 255.0, 0, 255).astype(np.uint8)
        ny1 = np.stack((Y1,Cr1,Cb1), axis=-1); ny2 = np.stack((Y2,Cr2,Cb2), axis=-1)
        ny1[:,:,embed_component] = c1s; ny2[:,:,embed_component] = c2s
        f1m = cv2.cvtColor(ny1, cv2.COLOR_YCrCb2BGR); f2m = cv2.cvtColor(ny2, cv2.COLOR_YCrCb2BGR)
        logging.debug(f"--- Embed Finish P:{pair_index}, Mods:{modifications_count} ---")
        return f1m, f2m

    except cv2.error as cv_err:
        logging.error(f"OpenCV error P:{pair_index}: {cv_err}", exc_info=True); return None, None
    except MemoryError as mem_err:
        logging.error(f"Memory error P:{pair_index}: {mem_err}", exc_info=True); return None, None
    except Exception as e:
        logging.error(f"Critical error P:{pair_index}: {e}", exc_info=True); return None, None

# --- Воркер для одной пары кадров ---
def _embed_single_pair_task(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """
    Обрабатывает одну пару кадров: выбирает кольца на основе энтропии из пула кандидатов
    и вызывает embed_frame_pair.
    """
    pair_idx = args['pair_idx']
    fn = 2 * pair_idx
    bits = args['bits']
    f1 = args['frame1'] # Используем только первый кадр для выбора колец
    f2 = args['frame2']
    nr = args['n_rings']
    nrtu = args['num_rings_to_use']
    cps = args['candidate_pool_size']
    ec = args['embed_component']
    upm = args['use_perceptual_masking']
    selected_rings = [] # Инициализируем на случай ошибки

    try:
        # --- ШАГ 1: Получение детерминированного пула кандидатов ---
        candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
        if len(candidate_rings) < nrtu:
            raise ValueError(f"Not enough candidates {len(candidate_rings)}<{nrtu} for pair {pair_idx}")

        # --- ШАГ 2: Адаптивный выбор из пула по энтропии ---
        # Используем *первый* кадр пары для расчета энтропии
        comp1_sel = f1[:, :, ec].astype(np.float32) / 255.0
        pyr1 = dtcwt_transform(comp1_sel, fn)
        if pyr1 is None or pyr1.lowpass is None:
            raise RuntimeError(f"DTCWT L1 failed for selection in pair {pair_idx}")

        L1s = pyr1.lowpass
        # Убедимся, что L1s это NumPy массив для корректной индексации
        if not isinstance(L1s, np.ndarray):
            L1s = np.array(L1s)

        # Получаем координаты колец для L1s
        coords = ring_division(L1s, nr, fn)
        if coords is None or len(coords) != nr:
             raise RuntimeError(f"Ring division failed for L1s in pair {pair_idx}")

        # Вычисляем энтропию для колец-кандидатов
        entropies = []
        min_pixels_for_entropy = 10 # Минимальное кол-во пикселей для расчета
        for r_idx in candidate_rings:
            entropy_val = -float('inf') # Значение по умолчанию, если расчет не удался
            # Проверяем валидность индекса и наличие координат
            if 0 <= r_idx < len(coords) and coords[r_idx] is not None:
                 c = coords[r_idx]
                 if c.shape[0] >= min_pixels_for_entropy:
                      try:
                           rs, cs = c[:, 0], c[:, 1]
                           rv = L1s[rs, cs] # Получаем значения пикселей кольца
                           # Вычисляем энтропию (только Шеннона, вторая не нужна для сортировки)
                           shannon_entropy, _ = calculate_entropies(rv, fn, r_idx)
                           if np.isfinite(shannon_entropy):
                                entropy_val = shannon_entropy
                      except IndexError:
                           logging.warning(f"IndexError during entropy calculation P:{pair_idx} R:{r_idx}")
                      except Exception as entropy_e:
                           logging.warning(f"Entropy calc error P:{pair_idx} R:{r_idx}: {entropy_e}")
                           # Оставляем entropy_val = -inf
            entropies.append((entropy_val, r_idx)) # Сохраняем (энтропия, индекс_кольца)

        # Сортируем по УБЫВАНИЮ энтропии
        entropies.sort(key=lambda x: x[0], reverse=True)

        # Выбираем 'nrtu' колец с наивысшей валидной энтропией
        selected_rings = [idx for e, idx in entropies if e > -float('inf')][:nrtu]

        # Проверяем, достаточно ли колец выбрано
        if len(selected_rings) < nrtu:
            # Если не хватило колец с валидной энтропией, можно дополнить детерминированно
            # или вызвать ошибку. Выберем дополнение как запасной вариант.
            logging.warning(f"[P:{pair_idx}] Not enough rings with valid entropy ({len(selected_rings)}<{nrtu}). Falling back.")
            deterministic_fallback = candidate_rings[:nrtu]
            # Добавляем недостающие кольца, избегая дубликатов
            for ring in deterministic_fallback:
                if ring not in selected_rings:
                    selected_rings.append(ring)
                if len(selected_rings) == nrtu:
                    break
            # Если и так не хватило (не должно случиться при cps>=nrtu), то ошибка
            if len(selected_rings) < nrtu:
                 raise RuntimeError(f"Fallback failed, still not enough rings for pair {pair_idx}")
            logging.warning(f"[P:{pair_idx}] Selected rings after fallback: {selected_rings}")
        else:
             logging.info(f"[P:{pair_idx}] Selected rings (Entropy based): {selected_rings}")


        # --- ШАГ 3: Вызов встраивания с выбранными кольцами ---
        mod_f1, mod_f2 = embed_frame_pair(f1, f2, bits, selected_rings, nr, fn, upm, ec)

        # Возвращаем результат
        return fn, mod_f1, mod_f2, selected_rings

    except Exception as e:
        logging.error(f"Error in single pair task (Embed - Entropy Sel.) P:{pair_idx}: {e}", exc_info=True)
        return fn, None, None, [] # Возвращаем пустой список колец при ошибке


# --- Воркер для обработки БАТЧА задач ---
def _embed_batch_worker(batch_args_list: List[Dict]) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]]:
    batch_results = []
    for args in batch_args_list:
        result = _embed_single_pair_task(args)
        batch_results.append(result)
    return batch_results

# --- Основная функция встраивания (ThreadPool + Batches) ---
@profile
def embed_watermark_in_video(
        frames: List[np.ndarray], packet_bits: np.ndarray, n_rings: int = N_RINGS, num_rings_to_use: int = NUM_RINGS_TO_USE,
        bits_per_pair: int = BITS_PER_PAIR, candidate_pool_size: int = CANDIDATE_POOL_SIZE, max_packet_repeats: int = MAX_PACKET_REPEATS,
        fps: float = FPS, max_workers: Optional[int] = MAX_WORKERS,
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING, embed_component: int = EMBED_COMPONENT):
    num_frames=len(frames); total_pairs=num_frames//2; packet_len_bits=packet_bits.size
    if total_pairs==0 or packet_len_bits==0: logging.warning("Skip embed: no pairs or no data"); return frames[:]
    pairs_needed=ceil(max_packet_repeats*packet_len_bits/bits_per_pair); pairs_to_process=min(total_pairs, pairs_needed); total_bits=pairs_to_process*bits_per_pair
    num_reps = ceil(total_bits/packet_len_bits) if packet_len_bits>0 else 1; bits_flat=np.tile(packet_bits,num_reps)[:total_bits]
    logging.info(f"Embed Start (ThreadPool+Batches): {total_bits}b({bits_per_pair}/p) in {pairs_to_process} pairs. Pkt:{packet_len_bits}b. Reps:{max_packet_repeats}t,{num_reps:.1f}a.")
    start_time=time.time(); watermarked_frames=frames[:]; rings_log: Dict[int,List[int]]={}; pc,ec,uc=0,0,0; skipped_pairs=0; all_pairs_args=[]

    for pair_idx in range(pairs_to_process):
        i1=2*pair_idx; i2=i1+1
        if i2>=num_frames or frames[i1] is None or frames[i2] is None: skipped_pairs+=1; continue
        sbi=pair_idx*bits_per_pair; ebi=sbi+bits_per_pair; cb=bits_flat[sbi:ebi].tolist()
        if len(cb)!=bits_per_pair: skipped_pairs+=1; continue
        args={'pair_idx':pair_idx, 'frame1':frames[i1], 'frame2':frames[i2], 'bits':cb, 'n_rings':n_rings, 'num_rings_to_use':num_rings_to_use, 'candidate_pool_size':candidate_pool_size, 'frame_number':i1, 'use_perceptual_masking':use_perceptual_masking, 'embed_component':embed_component}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args);
    if skipped_pairs>0: logging.warning(f"Skipped {skipped_pairs} pairs.");
    if num_valid_tasks==0: logging.error("No valid tasks."); return watermarked_frames

    num_workers = max_workers if max_workers is not None and max_workers>0 else (os.cpu_count() or 1)
    batch_size = max(1, ceil(num_valid_tasks / (num_workers * 2))) # Делим на удвоенное число воркеров
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i:i+batch_size] for i in range(0, num_valid_tasks, batch_size) if all_pairs_args[i:i+batch_size]]
    logging.info(f"Launching {num_batches} batches ({num_valid_tasks} pairs) using ThreadPool (mw={num_workers}, batch_size≈{batch_size})...")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch_idx = {executor.submit(_embed_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]; original_batch = batched_args_list[batch_idx]
                try:
                    batch_results = future.result()
                    if len(batch_results) != len(original_batch): ec += len(original_batch); logging.error(f"Batch {batch_idx} size mismatch!"); continue;
                    for i, single_res in enumerate(batch_results):
                        if single_res and len(single_res)==4:
                            fn_res, mod_f1, mod_f2, sel_rings = single_res; pair_idx = original_batch[i]['pair_idx']; i1=2*pair_idx; i2=i1+1
                            if sel_rings: rings_log[pair_idx] = sel_rings
                            if mod_f1 is not None and mod_f2 is not None:
                                if i1<len(watermarked_frames): watermarked_frames[i1]=mod_f1; uc+=1
                                if i2<len(watermarked_frames): watermarked_frames[i2]=mod_f2; uc+=1
                                pc+=1
                            else: ec+=1
                        else: ec+=1
                except Exception as e: ec+=len(original_batch); logging.error(f"Batch {batch_idx} failed: {e}", exc_info=True)
    except Exception as e: logging.critical(f"ThreadPoolExecutor error: {e}", exc_info=True); return frames[:]

    logging.info(f"Batch processing done. OK pairs:{pc}, Err/Skip pairs:{ec+skipped_pairs}. Updated frames:{uc}.")
    # Исправленный блок сохранения лога колец
    if rings_log:
        try: # Добавляем try
            ser_log={str(k):v for k,v in rings_log.items()};
            with open(SELECTED_RINGS_FILE,'w') as f:
                 json.dump(ser_log, f, indent=4);
            logging.info(f"Rings log saved: {SELECTED_RINGS_FILE}")
        except Exception as e: # Добавляем except
             logging.error(f"Save rings log failed: {e}", exc_info=True) # Логируем ошибку
    else: logging.warning("Rings log empty.")
    end_time=time.time(); logging.info(f"Embed done. Time: {end_time-start_time:.2f}s.")
    return watermarked_frames


# --- Основная Функция (main) ---
def main():
    global BCH_CODE_OBJECT, DTCWT_OPENCL_ENABLED
    start_time_main = time.time()
    input_video = "input.mp4"
    backend_name_str = 'opencl' if DTCWT_OPENCL_ENABLED else 'numpy'
    base_output_filename = f"watermarked_galois_t4_{backend_name_str}_thr_batched"
    output_video = base_output_filename + OUTPUT_EXTENSION
    logging.info(f"--- Starting Embedding Main Process (ThreadPool + Batches + {backend_name_str.upper()} DTCWT) ---")

    frames, fps_read = read_video(input_video)
    if not frames: logging.critical("Video read failed."); return
    fps_to_use = float(FPS) if fps_read <= 0 else fps_read
    if len(frames) < 2: logging.critical("Not enough frames."); return

    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES); original_id_hex = original_id_bytes.hex()
    logging.info(f"Generated Payload ID ({PAYLOAD_LEN_BYTES * 8} bit, Hex): {original_id_hex}")

    packet_bits_to_embed: Optional[np.ndarray] = None
    effective_ecc_enabled = USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is not None
    if effective_ecc_enabled:
        try:
            bch_k = BCH_CODE_OBJECT.k; payload_len_bits = PAYLOAD_LEN_BYTES*8; assert payload_len_bits <= bch_k, f"Payload {payload_len_bits} > k {bch_k}";
            payload_bits = np.unpackbits(np.frombuffer(original_id_bytes,dtype=np.uint8)); packet_bits_to_embed = add_ecc(payload_bits, BCH_CODE_OBJECT); assert packet_bits_to_embed is not None, "add_ecc failed"
            logging.info(f"Using ECC packet ({packet_bits_to_embed.size} bits).")
        except Exception as e: logging.error(f"ECC prep failed: {e}. Using raw."); effective_ecc_enabled = False; packet_bits_to_embed = None
    if packet_bits_to_embed is None:
        packet_bits_to_embed = np.unpackbits(np.frombuffer(original_id_bytes,dtype=np.uint8)); logging.info(f"Using raw payload ({packet_bits_to_embed.size} bits).")

    try:
        with open(ORIGINAL_WATERMARK_FILE, "w") as f: f.write(original_id_hex)
        logging.info(f"Original ID saved: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e: logging.error(f"Save ID failed: {e}")

    watermarked_frames = embed_watermark_in_video(
        frames=frames, packet_bits=packet_bits_to_embed, n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE,
        bits_per_pair=BITS_PER_PAIR, candidate_pool_size=CANDIDATE_POOL_SIZE, max_packet_repeats=MAX_PACKET_REPEATS,
        fps=fps_to_use, max_workers=MAX_WORKERS,
        use_perceptual_masking=USE_PERCEPTUAL_MASKING, embed_component=EMBED_COMPONENT)

    if watermarked_frames and len(watermarked_frames) == len(frames):
        write_video(frames=watermarked_frames, out_path=output_video, fps=fps_to_use, codec=OUTPUT_CODEC)
        logging.info(f"Watermarked video saved: {output_video}")
        try:
            if os.path.exists(output_video): logging.info(f"Output size: {os.path.getsize(output_video)/(1024*1024):.2f} MB")
            else: logging.error(f"Output file missing!")
        except OSError as e: logging.error(f"Get size failed: {e}")
    else: logging.error("Embedding failed. No output.")

    logging.info(f"--- Embedding Main Process Finished ({'OpenCL Fwd' if DTCWT_OPENCL_ENABLED else 'NumPy'} DTCWT) ---")
    total_time_main = time.time() - start_time_main
    logging.info(f"--- Total Main Function Time: {total_time_main:.2f} seconds ---")
    print(f"\nEmbedding process (ThreadPool + Batches, {'OpenCL Fwd' if DTCWT_OPENCL_ENABLED else 'NumPy'} DTCWT) finished.")
    print(f"Output: {output_video}")
    print(f"Log: {LOG_FILENAME}")
    print(f"ID file: {ORIGINAL_WATERMARK_FILE}")
    print(f"Rings file: {SELECTED_RINGS_FILE}")
    print("\nRun extractor.py to extract and verify.")

# --- Точка Входа ---
if __name__ == "__main__":
    original_dtcwt_backend = 'numpy' # Сохраняем исходный бэкенд
    try:
        import dtcwt
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
                 logging.warning(f"OpenCL backend test failed: {e_init}. Falling back to NumPy.", exc_info=False)
                 if dtcwt.backend_name == 'opencl': dtcwt.pop_backend()
                 DTCWT_OPENCL_ENABLED = False
        else:
            logging.warning("push_backend('opencl') did not change backend! Using NumPy.")
            DTCWT_OPENCL_ENABLED = False
            if dtcwt.backend_name != 'numpy': dtcwt.push_backend('numpy')
    except ImportError: logging.warning("dtcwt library not found."); DTCWT_OPENCL_ENABLED = False
    except ValueError as e_push: logging.warning(f"Failed switch to OpenCL: {e_push}. Using NumPy."); DTCWT_OPENCL_ENABLED = False
    except Exception as e_ocl: logging.warning(f"Error setting/testing OpenCL: {e_ocl}. Using NumPy.", exc_info=True); DTCWT_OPENCL_ENABLED = False
    finally: # Проверка консистентности флага и бэкенда
        try:
             if 'dtcwt' in sys.modules:
                  final_backend = dtcwt.backend_name
                  if final_backend == 'opencl' and not DTCWT_OPENCL_ENABLED:
                       logging.error("Inconsistency: OpenCL active but flag False!"); dtcwt.push_backend('numpy')
                  elif final_backend != 'opencl' and DTCWT_OPENCL_ENABLED:
                       logging.error("Inconsistency: NumPy active but OpenCL flag True!"); DTCWT_OPENCL_ENABLED = False
                  logging.info(f"Final active DTCWT backend for main: {dtcwt.backend_name}")
        except Exception as e_final_check: logging.error(f"Error final backend check: {e_final_check}")

    if USE_ECC and not GALOIS_AVAILABLE:
        print("\nERROR: ECC required but galois failed/unavailable.")
    else:
        prof = cProfile.Profile(); prof.enable()
        try:
            main()
            print(f"\n--- DTCWT Backend Used: {'OpenCL (Forward Only)' if DTCWT_OPENCL_ENABLED else 'NumPy'} ---")
        except FileNotFoundError as e: print(f"\nERROR: {e}"); logging.error(f"{e}", exc_info=True)
        except Exception as e: logging.critical(f"Unhandled main: {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. Log: {LOG_FILENAME}")
        finally:
            prof.disable(); stats = pstats.Stats(prof)
            print("\n--- Profiling Stats (Top 30 Cumulative Time) ---")
            stats.strip_dirs().sort_stats("cumulative").print_stats(30)
            print("-------------------------------------------------")
            backend_str = 'opencl' if DTCWT_OPENCL_ENABLED else 'numpy'
            pfile = f"profile_embed_{backend_str}_batched_galois_t{BCH_T}.txt"
            try:
                with open(pfile, "w") as f: sf = pstats.Stats(prof, stream=f); sf.strip_dirs().sort_stats("cumulative").print_stats()
                logging.info(f"Profiling stats saved: {pfile}"); print(f"Profiling stats saved: {pfile}")
            except IOError as e: logging.error(f"Save profile failed: {e}"); print(f"Warning: Could not save profile stats.")

            # --- Восстановление исходного бэкенда dtcwt ---
            try:
                if 'dtcwt' in sys.modules and 'original_dtcwt_backend' in locals() and dtcwt.backend_name != original_dtcwt_backend:
                    logging.info(f"Restoring original dtcwt backend: {original_dtcwt_backend}")
                    # Восстанавливаем, пока не совпадет или стек не опустеет
                    while dtcwt.backend_name != original_dtcwt_backend and len(getattr(dtcwt, '_backend_stack', [])) > 0:
                         dtcwt.pop_backend()
                    # Если все равно не совпало, ставим принудительно
                    if dtcwt.backend_name != original_dtcwt_backend:
                         dtcwt.push_backend(original_dtcwt_backend)
                         # Очищаем лишний push
                         if hasattr(dtcwt, '_backend_stack') and len(dtcwt._backend_stack) > 1: dtcwt._backend_stack.pop(0)
                    logging.info(f"DTCWT backend restored to: {dtcwt.backend_name}")
            except Exception as e_restore: logging.warning(f"Could not restore backend: {e_restore}")