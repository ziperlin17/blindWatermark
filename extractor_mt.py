import cProfile
import concurrent
import pstats
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import random
import logging
import time
import json
import os
import hashlib
# Убираем PIL, если не используется
# import imagehash
# line_profiler / cProfile опционально
# from line_profiler import profile
# import cProfile
# import pstats
# --- PyTorch импорты ---
import torch
import torch.nn.functional as F
from galois import BCH
from line_profiler import profile

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    # Определим классы-пустышки для предотвращения ошибок импорта
    class DTCWTForward: pass
    class DTCWTInverse: pass
    logging.error("ОШИБКА: Библиотека pytorch_wavelets не найдена! Установите: pip install pytorch_wavelets")
try:
    # Используем нашу новую библиотеку для DCT/IDCT
    import torch_dct as dct_torch # Импортируем под псевдонимом
    TORCH_DCT_AVAILABLE = True
except ImportError:
    TORCH_DCT_AVAILABLE = False
    logging.error("ОШИБКА: Библиотека torch-dct не найдена! Установите: pip install torch-dct")
# --------------------------
from typing import List, Tuple, Optional, Dict, Any
# functools убран
import uuid
from math import ceil
from collections import Counter
import sys
# --- Galois импорты ---
try:
    import galois
    BCH_TYPE = galois.BCH; GALOIS_IMPORTED = True; logging.info("galois library imported.")
except ImportError:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.info("galois library not found.")
except Exception as import_err:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.error(f"Galois import error: {import_err}", exc_info=True)

# --- Глобальные Параметры (СОГЛАСОВАНЫ с Embedder) ---
LAMBDA_PARAM: float = 0.05 # Не используется в экстракторе, но оставляем для консистентности
ALPHA_MIN: float = 1.13   # Не используется в экстракторе
ALPHA_MAX: float = 1.28   # Не используется в экстракторе
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0 # Используется в calculate_entropies
EMBED_COMPONENT: int = 2 # Cb
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection' # Метод выбора колец
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True         # Ожидать ли ECC в принципе (для первого пакета)
BCH_M: int = 8
BCH_T: int = 9             # <--- СОГЛАСОВАНО С Embedder
FPS: int = 30              # Не используется, но оставляем
LOG_FILENAME: str = 'watermarking_extract_pytorch.log' # Новое имя
INPUT_EXTENSION: str = '.mp4' # Согласовано с embedder
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS_EXTRACT: Optional[int] = 14 # Настраивается по CPU
# max_expected_packets определяется в функции extract_watermark_from_video

# --- Инициализация Галуа (СОГЛАСОВАНА с Embedder) ---
BCH_CODE_OBJECT: Optional[BCH_TYPE] = None
GALOIS_AVAILABLE = False
if GALOIS_IMPORTED:
    _test_bch_ok = False; _test_decode_ok = False
    try:
        _test_m = BCH_M; _test_t = BCH_T; _test_n = (1 << _test_m) - 1; _test_d = 2 * _test_t + 1
        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)
        # Определяем ожидаемое k
        if _test_t == 5: expected_k = 215
        elif _test_t == 7: expected_k = 201
        elif _test_t == 9: expected_k = 187 # <--- Для t=9
        elif _test_t == 11: expected_k = 173
        elif _test_t == 15: expected_k = 131
        else: logging.error(f"Неизвестное k для t={_test_t}"); expected_k = -1

        if expected_k != -1 and hasattr(_test_bch_galois, 't') and hasattr(_test_bch_galois, 'k') \
           and _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
             logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
             _test_bch_ok = True; BCH_CODE_OBJECT = _test_bch_galois
        else: logging.error(f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t=getattr(_test_bch_galois, 't', 'N/A'), k=getattr(_test_bch_galois, 'k', 'N/A').")

        if _test_bch_ok and BCH_CODE_OBJECT is not None:
            try:
                _n_bits = BCH_CODE_OBJECT.n; _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8); GF2 = galois.GF(2); _dummy_cw_vec = GF2(_dummy_cw_bits)
                _msg, _flips = BCH_CODE_OBJECT.decode(_dummy_cw_vec, errors=True)
                _test_decode_ok = (_flips is not None or _flips == 0); logging.info(f"galois: decode() test {'OK' if _test_decode_ok else 'failed'}.")
            except Exception as decode_err: logging.error(f"galois: decode() test failed: {decode_err}", exc_info=True); _test_decode_ok = False
    except Exception as test_err: logging.error(f"galois: ОШИБКА теста: {test_err}", exc_info=True); BCH_CODE_OBJECT = None; _test_bch_ok = False
    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None
if GALOIS_AVAILABLE: logging.info("galois: Готов к использованию.")
else: logging.warning("galois: НЕ ДОСТУПЕН.")

# --- Настройка логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO, format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование конфигурации ---
logging.info(f"--- Запуск Скрипта Извлечения (PyTorch Wavelets & DCT) ---")
logging.info(f"PyTorch Wavelets Доступно: {PYTORCH_WAVELETS_AVAILABLE}")
logging.info(f"Torch DCT Доступно: {TORCH_DCT_AVAILABLE}")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit")
logging.info(f"ECC Ожидается (для 1-го пак.): {USE_ECC}, Доступен/Работает: {GALOIS_AVAILABLE} (BCH m={BCH_M}, t={BCH_T})")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, N_RINGS_Total={N_RINGS}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS_EXTRACT or 'default'}) с батчингом.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error(f"NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE! Проверьте настройки.")
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE != BITS_PER_PAIR.")

# --- Базовые Функции (Модифицированные или Новые) ---

def dct1d_torch(s_tensor: torch.Tensor) -> torch.Tensor:
    """1D DCT-II используя torch-dct."""
    if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct не доступен")
    return dct_torch.dct(s_tensor, norm='ortho')

def svd_torch_s1(tensor_1d: torch.Tensor) -> Optional[torch.Tensor]:
    """Применяет SVD и возвращает только первое сингулярное число как тензор."""
    try:
        tensor_2d = tensor_1d.unsqueeze(-1)
        s_values = torch.linalg.svdvals(tensor_2d)
        if s_values is None or s_values.numel() == 0: return None
        if not torch.isfinite(s_values[0]): return None
        return s_values[0] # Возвращаем тензор (скаляр)
    except Exception as e:
        logging.error(f"PyTorch SVD error: {e}", exc_info=True)
        return None

def dtcwt_pytorch_forward(yp_tensor: torch.Tensor, xfm: DTCWTForward, device: torch.device, fn: int = -1) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Применяет прямое DTCWT PyTorch к одному каналу (2D тензору)."""
    if not PYTORCH_WAVELETS_AVAILABLE: logging.error("PTW unavailable."); return None, None
    if not isinstance(yp_tensor, torch.Tensor) or yp_tensor.ndim != 2: logging.error(f"[F:{fn}] Invalid input tensor."); return None, None
    try:
        yp_tensor = yp_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        xfm = xfm.to(device)
        with torch.no_grad(): Yl, Yh = xfm(yp_tensor)
        if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh: logging.error(f"[F:{fn}] DTCWTForward invalid result."); return None, None
        return Yl, Yh
    except Exception as e: logging.error(f"[F:{fn}] PT DTCWT fwd error: {e}"); return None, None

def ring_division(lp_tensor: torch.Tensor, nr: int = N_RINGS, fn: int = -1) -> List[Optional[torch.Tensor]]:
    """Разбивает 2D PyTorch тензор на N колец (версия из embedder)."""
    if not isinstance(lp_tensor, torch.Tensor) or lp_tensor.ndim != 2: logging.error(f"[F:{fn}] Invalid input for ring_division."); return [None] * nr
    H, W = lp_tensor.shape; device = lp_tensor.device
    if H < 2 or W < 2: logging.warning(f"[F:{fn}] Tensor too small ({H}x{W})"); return [None] * nr
    try:
        rr, cc = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0; distances = torch.sqrt((rr - center_r)**2 + (cc - center_c)**2)
        min_dist, max_dist = torch.tensor(0.0, device=device), torch.max(distances)
        if max_dist < 1e-9: ring_bins = torch.tensor([0.0]*(nr + 1), device=device); ring_bins[1:] = max_dist + 1e-6
        else: ring_bins = torch.linspace(min_dist.item(), (max_dist + 1e-6).item(), nr + 1, device=device)
        ring_indices = torch.zeros_like(distances, dtype=torch.long) - 1
        for i in range(nr):
            mask = (distances >= ring_bins[i]) & (distances < ring_bins[i+1] if i < nr-1 else distances <= ring_bins[i+1])
            ring_indices[mask] = i
        ring_indices[distances < ring_bins[1]] = 0
        rings: List[Optional[torch.Tensor]] = [None] * nr
        for rdx in range(nr):
            coords_tensor = torch.nonzero(ring_indices == rdx, as_tuple=False)
            if coords_tensor.shape[0] > 0: rings[rdx] = coords_tensor.long()
        return rings
    except Exception as e: logging.error(f"Ring division PT error F{fn}: {e}"); return [None] * nr

# --- Функции, работающие с NumPy (остаются) ---
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
                ee = -np.sum(p*np.exp(1.-p)); collision_entropy = ee # Ваша старая ee
    return shannon_entropy, collision_entropy

def get_fixed_pseudo_random_rings(pi: int, nr: int, ps: int) -> List[int]:
    if ps <= 0: return []
    if ps > nr: ps = nr
    seed_str = str(pi).encode('utf-8'); hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big'); prng = random.Random(seed_int)
    try: candidate_indices = prng.sample(range(nr), ps)
    except ValueError: candidate_indices = list(range(nr)); prng.shuffle(candidate_indices); candidate_indices = candidate_indices[:ps]
    logging.debug(f"[P:{pi}] Candidates: {candidate_indices}");
    return candidate_indices

def bits_to_bytes(bit_list: List[Optional[int]]) -> Optional[bytes]:
    valid_bits = [b for b in bit_list if b is not None and b in (0, 1)]
    num_bits = len(valid_bits)
    if num_bits == 0: return b''
    remainder = num_bits % 8
    if remainder != 0: padding_len = 8 - remainder; valid_bits.extend([0] * padding_len); num_bits += padding_len
    byte_array = bytearray()
    for i in range(0, num_bits, 8):
        byte_chunk = valid_bits[i:i+8]
        try:
            if len(byte_chunk) != 8: logging.error(f"Byte chunk error: {byte_chunk}"); return None
            byte_val = int("".join(map(str, byte_chunk)), 2); byte_array.append(byte_val)
        except ValueError: logging.error(f"Invalid symbols in bit chunk: {byte_chunk}"); return None
    return bytes(byte_array)

def decode_ecc(packet_bits_list: List[int], bch_code: Optional[BCH_TYPE], expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    if not GALOIS_AVAILABLE or bch_code is None: logging.error("ECC decode called but unavailable."); return None, -1
    n_corrected = -1
    try:
        n=bch_code.n; k=bch_code.k; expected_payload_bits=expected_data_len_bytes*8
        if len(packet_bits_list) != n: logging.error(f"Decode ECC: Bad len {len(packet_bits_list)}!={n}"); return None, -1
        if expected_payload_bits > k: logging.error(f"Decode ECC: Payload {expected_payload_bits}>k {k}"); return None, -1
        packet_bits_np=np.array(packet_bits_list, dtype=np.uint8); GF=bch_code.field; rx_vec=GF(packet_bits_np)
        try: corr_msg_vec, n_corrected = bch_code.decode(rx_vec, errors=True)
        except galois.errors.UncorrectableError: logging.warning("Galois ECC: Uncorrectable."); return None, -1
        corr_k_bits=corr_msg_vec.view(np.ndarray).astype(np.uint8)
        if corr_k_bits.size < expected_payload_bits: logging.error(f"Decode ECC: Decoded len {corr_k_bits.size} < {expected_payload_bits}"); return None, n_corrected
        payload_bits=corr_k_bits[:expected_payload_bits]; payload_bytes = bits_to_bytes(payload_bits.tolist())
        if payload_bytes is None: logging.error("Decode ECC: bits_to_bytes failed."); return None, n_corrected
        logging.info(f"Galois ECC: Decoded, corrected {n_corrected} errors.")
        return payload_bytes, n_corrected
    except Exception as e: logging.error(f"Decode ECC unexpected error: {e}", exc_info=True); return None, -1

def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # ... (Полный код функции read_video без изменений) ...
    logging.info(f"Reading: {video_path}"); frames: List[np.ndarray] = []; fps = float(FPS); cap = None; h, w = -1, -1;
    try:
        if not os.path.exists(video_path): raise FileNotFoundError(f"File not found: {video_path}")
        cap = cv2.VideoCapture(video_path); assert cap.isOpened(), f"Cannot open {video_path}"
        fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
        if w <= 0 or h <= 0: ret, f_chk = cap.read(); assert ret and f_chk is not None, "Cannot read frame to get size"; h,w,_ = f_chk.shape; cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logging.info(f"Props: {w}x{h}@{fps:.2f}~{fc if fc>0 else '?'}f"); rc,nc,ic=0,0,0;
        while True:
            ret,f=cap.read();
            if not ret: break
            if f is None: nc+=1; continue
            if f.ndim==3 and f.shape[:2]==(h,w) and f.dtype==np.uint8: frames.append(f); rc+=1;
            else: ic+=1; logging.warning(f"Skipped invalid frame #{rc+nc+ic}. Shape:{f.shape}, dtype:{f.dtype}");
        logging.info(f"Read loop done. Valid:{rc}, None:{nc}, Invalid:{ic}"); assert rc>0, "No valid frames"
    except Exception as e: logging.error(f"Read error: {e}", exc_info=True); frames=[]
    finally:
        if cap and cap.isOpened(): cap.release()
    return frames, fps


def extract_single_bit(
        # --- НОВЫЕ/ИЗМЕНЕННЫЕ аргументы ---
        comp1_tensor: torch.Tensor,     # Исходный тензор компонента кадра 1
        comp2_tensor: torch.Tensor,     # Исходный тензор компонента кадра 2
        dtcwt_fwd: 'DTCWTForward',      # Объект прямого преобразования
        device: torch.device,           # Устройство
        # --- Остальные аргументы ---
        ring_idx: int,
        n_rings: int,
        fn: int
    ) -> Optional[int]:
    """
    Извлекает один бит из заданного кольца (МТ-архитектура).
    Выполняет DTCWT для КАЖДОГО кадра ПРИ КАЖДОМ вызове.
    Использует PyTorch DCT/SVD.
    """
    pair_index = fn // 2
    prefix = f"[MT P:{pair_index}, R:{ring_idx}]" # Префикс для логов

    try:
        # --- Шаг 1: Проверка базовых входных данных ---
        if comp1_tensor is None or comp2_tensor is None or comp1_tensor.shape != comp2_tensor.shape \
           or dtcwt_fwd is None or device is None \
           or not (0 <= ring_idx < n_rings):
            logging.warning(f"{prefix} Invalid input args.")
            return None

        # --- Шаг 2: Прямое DTCWT (для КАЖДОГО кадра, КАЖДЫЙ раз) ---
        Yl_t_full, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
        Yl_t1_full, _ = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, fn + 1)

        if Yl_t_full is None or Yl_t1_full is None:
            logging.warning(f"{prefix} DTCWT forward failed inside extract_single_bit.")
            return None

        # Извлекаем LL-компоненты и проверяем размерность
        if Yl_t_full.dim() > 2: Yl_t = Yl_t_full.squeeze(0).squeeze(0)
        elif Yl_t_full.dim() == 2: Yl_t = Yl_t_full
        else: raise ValueError(f"Invalid Yl_t dim: {Yl_t_full.dim()}")

        if Yl_t1_full.dim() > 2: Yl_t1 = Yl_t1_full.squeeze(0).squeeze(0)
        elif Yl_t1_full.dim() == 2: Yl_t1 = Yl_t1_full
        else: raise ValueError(f"Invalid Yl_t1 dim: {Yl_t1_full.dim()}")

        if not torch.is_floating_point(Yl_t) or not torch.is_floating_point(Yl_t1):
            raise TypeError(f"Yl/Yl_t1 not float! Yl:{Yl_t.dtype}, Yl_t1:{Yl_t1.dtype}")
        if Yl_t.shape != Yl_t1.shape:
            raise ValueError(f"Yl/Yl_t1 shape mismatch! Yl:{Yl_t.shape}, Yl_t1:{Yl_t1.shape}")
        # logging.debug(f"{prefix} Internal DTCWT done.")

        # --- Шаг 3: Кольцевое деление (по свежим Yl_t, Yl_t1) ---
        r1c = ring_division(Yl_t, n_rings, fn)
        r2c = ring_division(Yl_t1, n_rings, fn + 1)

        if r1c is None or r2c is None or ring_idx >= len(r1c) or ring_idx >= len(r2c):
             logging.warning(f"{prefix} Invalid ring index or ring_division failed.")
             return None
        cd1_tensor = r1c[ring_idx]; cd2_tensor = r2c[ring_idx]
        min_ring_size = 10
        if cd1_tensor is None or cd2_tensor is None or cd1_tensor.shape[0] < min_ring_size or cd2_tensor.shape[0] < min_ring_size:
             logging.debug(f"{prefix} Ring coords None or ring too small (<{min_ring_size}).")
             return None

        # --- Шаг 4: Извлечение значений, DCT, SVD (НА ТЕНЗОРАХ) ---
        try:
            rows1, cols1 = cd1_tensor[:, 0], cd1_tensor[:, 1]
            rows2, cols2 = cd2_tensor[:, 0], cd2_tensor[:, 1]
            # Извлекаем из свежих Yl_t, Yl_t1
            rv1_tensor = Yl_t[rows1, cols1].to(dtype=torch.float32)
            rv2_tensor = Yl_t1[rows2, cols2].to(dtype=torch.float32)

            min_s = min(rv1_tensor.numel(), rv2_tensor.numel())
            if min_s == 0: return None
            if rv1_tensor.numel() != rv2_tensor.numel():
                rv1_tensor = rv1_tensor[:min_s]; rv2_tensor = rv2_tensor[:min_s]

            # logging.debug(f"{prefix} rv1 stats...") # Логи можно вернуть при необходимости
            # logging.debug(f"{prefix} rv2 stats...")

            # PyTorch DCT
            if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct not available")
            d1_tensor = dct1d_torch(rv1_tensor)
            d2_tensor = dct1d_torch(rv2_tensor)
            if not torch.isfinite(d1_tensor).all() or not torch.isfinite(d2_tensor).all(): return None
            # logging.debug(f"{prefix} DCT done...")

            # PyTorch SVD
            s1_tensor = svd_torch_s1(d1_tensor)
            s2_tensor = svd_torch_s1(d2_tensor)
            if s1_tensor is None or s2_tensor is None: return None
            # logging.debug(f"{prefix} SVD done...")

            s1 = s1_tensor.item(); s2 = s2_tensor.item()

        except RuntimeError as torch_err:
             logging.error(f"{prefix} PyTorch runtime error during Tensor DCT/SVD: {torch_err}", exc_info=True); return None
        except IndexError:
             logging.warning(f"{prefix} Index error getting ring tensor values."); return None
        except Exception as e:
             logging.error(f"{prefix} Error in Tensor DCT/SVD part: {e}", exc_info=True); return None

        # --- Шаг 5: Принятие решения ---
        eps = 1e-12; threshold = 1.0
        if abs(s2) < eps:
             logging.warning(f"{prefix} s2={s2:.2e} is close to zero. Unreliable ratio.")
             return None

        ratio = s1 / s2
        comparison_result = (ratio >= threshold)
        extracted_bit = 0 if comparison_result else 1

        logging.info(f"{prefix} Decision: s1={s1:.8e}, s2={s2:.8e}, ratio={ratio:.8f}, th={threshold}, comp={comparison_result}, bit={extracted_bit}")

        return extracted_bit

    except Exception as e:
        logging.error(f"Unexpected error in extract_single_bit MT (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
        return None


# @profile
# def extract_single_bit(L1_tensor: torch.Tensor, L2_tensor: torch.Tensor, ring_idx: int, n_rings: int, fn: int) -> Optional[int]:
#     """
#     Извлекает один бит (PyTorch DCT/SVD, ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ).
#     (BN-PyTorch-Optimized-Corrected - V2 Log)
#     """
#     pair_index = fn // 2
#     prefix = f"[BN P:{pair_index}, R:{ring_idx}]"
#
#     try:
#         # --- Шаг 1: Проверка входных данных ---
#         if L1_tensor is None or L2_tensor is None or L1_tensor.shape != L2_tensor.shape \
#            or not isinstance(L1_tensor, torch.Tensor) or not isinstance(L2_tensor, torch.Tensor) \
#            or L1_tensor.ndim != 2 or L2_tensor.ndim != 2 \
#            or not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
#             logging.warning(f"{prefix} Invalid L1/L2 provided.")
#             return None
#         device = L1_tensor.device
#
#         # --- Шаг 2: Кольцевое деление ---
#         r1c = ring_division(L1_tensor, n_rings, fn)
#         r2c = ring_division(L2_tensor, n_rings, fn + 1)
#         if r1c is None or r2c is None \
#            or not(0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
#              logging.warning(f"{prefix} Invalid ring index or ring_division failed.")
#              return None
#         cd1_tensor = r1c[ring_idx]; cd2_tensor = r2c[ring_idx]
#         min_ring_size = 10
#         if cd1_tensor is None or cd2_tensor is None or cd1_tensor.shape[0] < min_ring_size or cd2_tensor.shape[0] < min_ring_size:
#              logging.debug(f"{prefix} Ring coords None or ring too small (<{min_ring_size}).")
#              return None
#
#         # --- Шаг 3: Извлечение значений, DCT, SVD (НА ТЕНЗОРАХ) ---
#         try:
#             rows1, cols1 = cd1_tensor[:, 0], cd1_tensor[:, 1]
#             rows2, cols2 = cd2_tensor[:, 0], cd2_tensor[:, 1]
#             rv1_tensor = L1_tensor[rows1, cols1].to(dtype=torch.float32)
#             rv2_tensor = L2_tensor[rows2, cols2].to(dtype=torch.float32)
#             min_s = min(rv1_tensor.numel(), rv2_tensor.numel())
#             if min_s == 0: return None
#             if rv1_tensor.numel() != rv2_tensor.numel():
#                 rv1_tensor = rv1_tensor[:min_s]; rv2_tensor = rv2_tensor[:min_s]
#
#             # *** ЛОГ: Статистика входных значений кольца ***
#             logging.debug(f"{prefix} rv1 stats: size={rv1_tensor.numel()}, mean={rv1_tensor.mean():.6e}, std={rv1_tensor.std():.6e}")
#             logging.debug(f"{prefix} rv2 stats: size={rv2_tensor.numel()}, mean={rv2_tensor.mean():.6e}, std={rv2_tensor.std():.6e}")
#
#             # --- PyTorch DCT ---
#             if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct not available")
#             d1_tensor = dct1d_torch(rv1_tensor)
#             d2_tensor = dct1d_torch(rv2_tensor)
#             if not torch.isfinite(d1_tensor).all() or not torch.isfinite(d2_tensor).all(): return None
#             # *** ЛОГ: Первые DCT коэффициенты ***
#             logging.debug(f"{prefix} DCT done. d1[0]={d1_tensor[0]:.6e}, d2[0]={d2_tensor[0]:.6e}")
#
#
#             # --- PyTorch SVD ---
#             s1_tensor = svd_torch_s1(d1_tensor)
#             s2_tensor = svd_torch_s1(d2_tensor)
#             if s1_tensor is None or s2_tensor is None: return None
#             # *** ЛОГ: Сингулярные числа (тензоры) с высокой точностью ***
#             logging.debug(f"{prefix} SVD done. s1_tensor={s1_tensor.item():.8e}, s2_tensor={s2_tensor.item():.8e}")
#
#             # Конвертируем в Python float для финального расчета и сравнения
#             s1 = s1_tensor.item()
#             s2 = s2_tensor.item()
#
#         except RuntimeError as torch_err:
#              logging.error(f"{prefix} PyTorch runtime error during Tensor DCT/SVD: {torch_err}", exc_info=True); return None
#         except IndexError:
#              logging.warning(f"{prefix} Index error getting ring tensor values."); return None
#         except Exception as e:
#              logging.error(f"{prefix} Error in Tensor DCT/SVD processing part: {e}", exc_info=True); return None
#
#         # --- Шаг 4: Принятие решения ---
#         eps = 1e-12; threshold = 1.0
#         if abs(s2) < eps:
#              logging.warning(f"{prefix} s2={s2:.2e} is close to zero. Unreliable ratio.")
#              return None
#
#         ratio = s1 / s2
#         # --- Явное вычисление и логирование сравнения ---
#         comparison_result = (ratio >= threshold)
#         extracted_bit = 0 if comparison_result else 1
#
#         # *** ЛОГ: Детальная информация для принятия решения ***
#         logging.info(f"{prefix} Decision: s1={s1:.8e}, s2={s2:.8e}, ratio={ratio:.8f}, threshold={threshold}, comparison (ratio >= threshold)={comparison_result}, extracted_bit={extracted_bit}")
#
#         return extracted_bit
#
#     except Exception as e:
#         logging.error(f"Unexpected error in extract_single_bit (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
#         return None
#

def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
    """
    Обрабатывает батч задач извлечения в МТ-архитектуре.
    Выполняет DTCWT для выбора колец, затем вызывает extract_single_bit,
    который выполняет DTCWT для каждого кадра/кольца.
    """
    batch_results: Dict[int, List[Optional[int]]] = {}
    if not batch_args_list: return {}

    args_example = batch_args_list[0]
    nr = args_example.get('n_rings', N_RINGS)
    nrtu = args_example.get('num_rings_to_use', NUM_RINGS_TO_USE)
    cps = args_example.get('candidate_pool_size', CANDIDATE_POOL_SIZE)
    ec = args_example.get('embed_component', EMBED_COMPONENT)
    device = args_example.get('device')
    dtcwt_fwd = args_example.get('dtcwt_fwd') # Получаем объект DTCWT

    if device is None or dtcwt_fwd is None:
         logging.error("Device или DTCWTForward не переданы в _extract_batch_worker!")
         for args in batch_args_list:
              pair_idx = args.get('pair_idx', -1)
              if pair_idx != -1: batch_results[pair_idx] = [None] * nrtu
         return batch_results

    for args in batch_args_list:
        pair_idx = args.get('pair_idx', -1)
        f1_bgr = args.get('frame1')
        f2_bgr = args.get('frame2')

        current_nrtu = args.get('num_rings_to_use', NUM_RINGS_TO_USE)
        extracted_bits_for_pair: List[Optional[int]] = [None] * current_nrtu

        if pair_idx == -1 or f1_bgr is None or f2_bgr is None:
            logging.error(f"Недостаточно аргументов для обработки pair_idx={pair_idx if pair_idx != -1 else 'unknown'}")
            batch_results[pair_idx] = extracted_bits_for_pair; continue

        fn = 2 * pair_idx
        selected_rings = [] # Список для выбранных колец

        try:
            # --- Шаг 1: Преобразование цвета (для выбора колец и передачи дальше) ---
            if not isinstance(f1_bgr, np.ndarray) or not isinstance(f2_bgr, np.ndarray):
                 logging.warning(f"[MT Worker P:{pair_idx}] Input frames not numpy arrays.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            y1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
            y2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2YCrCb)
            c1_np = y1[:, :, ec].copy().astype(np.float32) / 255.0
            c2_np = y2[:, :, ec].copy().astype(np.float32) / 255.0
            comp1_tensor = torch.from_numpy(c1_np).to(device=device)
            comp2_tensor = torch.from_numpy(c2_np).to(device=device)

            # --- Шаг 2: Выбор колец (Требует DTCWT для ПЕРВОГО кадра) ---
            # ВАЖНО: Этот DTCWT нужен только для синхронизации выбора колец
            Yl_t_select, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
            if Yl_t_select is None:
                 logging.warning(f"[MT Worker P:{pair_idx}] DTCWT forward failed for ring selection.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue
            if Yl_t_select.dim() > 2: Yl_t_select = Yl_t_select.squeeze(0).squeeze(0)
            # --- Конец DTCWT для выбора колец ---

            coords = ring_division(Yl_t_select, nr, fn)
            if coords is None or len(coords) != nr:
                 logging.warning(f"[MT Worker P:{pair_idx}] Ring division failed.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
            if len(candidate_rings) < current_nrtu:
                logging.warning(f"[MT Worker P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{current_nrtu}.")
                current_nrtu = len(candidate_rings)
            if current_nrtu == 0:
                logging.error(f"[MT Worker P:{pair_idx}] No candidates to select rings from.")
                batch_results[pair_idx] = []; continue

            entropies = []; min_pixels = 10
            L1_numpy_for_entropy = Yl_t_select.cpu().numpy() # Используем результат DTCWT для выбора колец
            for r_idx_cand in candidate_rings:
                entropy_val = -float('inf')
                if 0 <= r_idx_cand < len(coords) and isinstance(coords[r_idx_cand], torch.Tensor) and coords[r_idx_cand].shape[0] >= min_pixels:
                     c_tensor = coords[r_idx_cand]; rows_t, cols_t = c_tensor[:, 0], c_tensor[:, 1]
                     rows_np, cols_np = rows_t.cpu().numpy(), cols_t.cpu().numpy()
                     try:
                         rv_np = L1_numpy_for_entropy[rows_np, cols_np]
                         shannon_entropy, _ = calculate_entropies(rv_np, fn, r_idx_cand)
                         if np.isfinite(shannon_entropy): entropy_val = shannon_entropy
                     except Exception as e_entr: logging.warning(f"[MT P:{pair_idx} R_cand:{r_idx_cand}] Entropy error: {e_entr}")
                entropies.append((entropy_val, r_idx_cand))

            entropies.sort(key=lambda x: x[0], reverse=True)
            selected_rings = [idx for e, idx in entropies if e > -float('inf')][:current_nrtu]

            if len(selected_rings) < current_nrtu: # Fallback
                logging.warning(f"[MT Worker P:{pair_idx}] Fallback ring selection ({len(selected_rings)}<{current_nrtu}).")
                deterministic_fallback = candidate_rings[:current_nrtu]
                needed = current_nrtu - len(selected_rings)
                for ring in deterministic_fallback:
                    if needed == 0: break
                    if ring not in selected_rings: selected_rings.append(ring); needed -= 1
                if len(selected_rings) < current_nrtu:
                     logging.error(f"[MT Worker P:{pair_idx}] Fallback failed.")
                     current_nrtu = len(selected_rings) # Используем то, что есть
            logging.info(f"[MT Worker P:{pair_idx}] Selected {len(selected_rings)} rings: {selected_rings}")

            # --- Шаг 3: Извлечение бит из выбранных колец ---
            extracted_bits_for_pair = [None] * len(selected_rings) # Размер по фактическому числу колец
            for i, ring_idx_to_extract in enumerate(selected_rings):
                 # Вызываем МТ-версию extract_single_bit, передавая ТЕНЗОРЫ и объект DTCWT
                 extracted_bits_for_pair[i] = extract_single_bit(
                     comp1_tensor, comp2_tensor, dtcwt_fwd, device, # Новые аргументы
                     ring_idx_to_extract, nr, fn # Старые аргументы
                 )

            # Дополняем None, если колец извлекли меньше, чем ожидали изначально (nrtu)
            while len(extracted_bits_for_pair) < nrtu:
                 extracted_bits_for_pair.append(None)

            batch_results[pair_idx] = extracted_bits_for_pair[:nrtu] # Возвращаем результат нужной длины

        except Exception as e:
            logging.error(f"Error processing pair {pair_idx} in MT worker: {e}", exc_info=True)
            batch_results[pair_idx] = [None] * nrtu # Заполняем None при ошибке

    return batch_results


# def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
#     """
#     Обрабатывает батч задач извлечения: выполняет DTCWT один раз на пару,
#     затем вызывает extract_single_bit для выбранных колец. (BN-PyTorch-Optimized)
#     """
#     batch_results: Dict[int, List[Optional[int]]] = {}
#     if not batch_args_list: return {}
#
#     # Получаем общие параметры из первого аргумента
#     args_example = batch_args_list[0]
#     nr = args_example.get('n_rings', N_RINGS)
#     nrtu = args_example.get('num_rings_to_use', NUM_RINGS_TO_USE)
#     cps = args_example.get('candidate_pool_size', CANDIDATE_POOL_SIZE)
#     ec = args_example.get('embed_component', EMBED_COMPONENT)
#     device = args_example.get('device')
#     dtcwt_fwd = args_example.get('dtcwt_fwd')
#
#     # Проверяем наличие критически важных общих аргументов
#     if device is None or dtcwt_fwd is None:
#          logging.error("Device или DTCWTForward не переданы в _extract_batch_worker!")
#          # Возвращаем пустой результат для всех пар в батче
#          for args in batch_args_list:
#               pair_idx = args.get('pair_idx', -1)
#               if pair_idx != -1:
#                    batch_results[pair_idx] = [None] * nrtu
#          return batch_results
#
#     # Итерируем по парам в батче
#     for args in batch_args_list:
#         pair_idx = args.get('pair_idx', -1)
#         f1_bgr = args.get('frame1')
#         f2_bgr = args.get('frame2')
#
#         # Инициализируем результат для текущей пары
#         # Убедимся, что nrtu валидно (на случай, если selected_rings будет короче)
#         current_nrtu = args.get('num_rings_to_use', NUM_RINGS_TO_USE)
#         extracted_bits_for_pair: List[Optional[int]] = [None] * current_nrtu
#
#         # Проверяем индивидуальные аргументы пары
#         if pair_idx == -1 or f1_bgr is None or f2_bgr is None:
#             logging.error(f"Недостаточно аргументов для обработки pair_idx={pair_idx if pair_idx != -1 else 'unknown'}")
#             batch_results[pair_idx] = extracted_bits_for_pair # Записываем None результат
#             continue # Переходим к следующей паре
#
#         fn = 2 * pair_idx
#         L1_tensor: Optional[torch.Tensor] = None # Переменные для действительных L1, L2
#         L2_tensor: Optional[torch.Tensor] = None
#
#         try:
#             # --- Шаг 1: Преобразование цвета и DTCWT ---
#             if not isinstance(f1_bgr, np.ndarray) or not isinstance(f2_bgr, np.ndarray):
#                  logging.warning(f"[BN Worker P:{pair_idx}] Input frames not numpy arrays.")
#                  batch_results[pair_idx] = extracted_bits_for_pair; continue
#
#             # Конвертация BGR -> YCrCb -> Компонент -> Тензор [0,1]
#             y1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
#             y2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2YCrCb)
#             # Используем .copy() для избежания проблем с read-only
#             c1_np = y1[:, :, ec].copy().astype(np.float32) / 255.0
#             c2_np = y2[:, :, ec].copy().astype(np.float32) / 255.0
#             comp1_tensor = torch.from_numpy(c1_np).to(device=device)
#             comp2_tensor = torch.from_numpy(c2_np).to(device=device)
#
#             # Прямое DTCWT
#             Yl_t, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
#             Yl_t1, _ = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, fn + 1)
#
#             if Yl_t is None or Yl_t1 is None:
#                  logging.warning(f"[BN Worker P:{pair_idx}] DTCWT forward failed.")
#                  batch_results[pair_idx] = extracted_bits_for_pair; continue
#
#             # --- Извлекаем LL и проверяем ---
#             if Yl_t.dim() > 2: L1_tensor = Yl_t.squeeze(0).squeeze(0)
#             elif Yl_t.dim() == 2: L1_tensor = Yl_t
#             else: raise ValueError(f"Invalid Yl_t dim: {Yl_t.dim()}")
#
#             if Yl_t1.dim() > 2: L2_tensor = Yl_t1.squeeze(0).squeeze(0)
#             elif Yl_t1.dim() == 2: L2_tensor = Yl_t1
#             else: raise ValueError(f"Invalid Yl_t1 dim: {Yl_t1.dim()}")
#
#             if not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
#                  raise TypeError(f"L1/L2 not float! L1:{L1_tensor.dtype}, L2:{L2_tensor.dtype}")
#             if L1_tensor.shape != L2_tensor.shape:
#                  raise ValueError(f"L1/L2 shape mismatch! L1:{L1_tensor.shape}, L2:{L2_tensor.shape}")
#
#             # --- Шаг 2: Выбор колец (используем L1_tensor) ---
#             coords = ring_division(L1_tensor, nr, fn) # PyTorch ring_division
#             if coords is None or len(coords) != nr:
#                  logging.warning(f"[BN Worker P:{pair_idx}] Ring division failed.")
#                  batch_results[pair_idx] = extracted_bits_for_pair; continue
#
#             candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
#             # Используем current_nrtu, определенное ранее
#             if len(candidate_rings) < current_nrtu:
#                 logging.warning(f"[BN Worker P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{current_nrtu}.")
#                 current_nrtu = len(candidate_rings) # Обновляем nrtu до числа кандидатов
#             if current_nrtu == 0:
#                 logging.error(f"[BN Worker P:{pair_idx}] No candidates to select rings from.")
#                 batch_results[pair_idx] = []; continue # Возвращаем пустой список бит
#
#             # Выбор по энтропии
#             entropies = []; min_pixels = 10
#             L1_numpy_for_entropy = L1_tensor.cpu().numpy() # Конвертируем один раз
#             for r_idx_cand in candidate_rings:
#                 entropy_val = -float('inf')
#                 if 0 <= r_idx_cand < len(coords) and isinstance(coords[r_idx_cand], torch.Tensor) and coords[r_idx_cand].shape[0] >= min_pixels:
#                      c_tensor = coords[r_idx_cand]; rows_t, cols_t = c_tensor[:, 0], c_tensor[:, 1]
#                      # Конвертируем индексы тензора в NumPy для индексации NumPy массива
#                      rows_np, cols_np = rows_t.cpu().numpy(), cols_t.cpu().numpy()
#                      try:
#                          rv_np = L1_numpy_for_entropy[rows_np, cols_np]
#                          shannon_entropy, _ = calculate_entropies(rv_np, fn, r_idx_cand)
#                          if np.isfinite(shannon_entropy): entropy_val = shannon_entropy
#                      except IndexError: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] IndexError entropy")
#                      except Exception as e_entr: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] Entropy error: {e_entr}")
#                 entropies.append((entropy_val, r_idx_cand))
#
#             entropies.sort(key=lambda x: x[0], reverse=True)
#             # Выбираем не более current_nrtu колец
#             selected_rings = [idx for e, idx in entropies if e > -float('inf')][:current_nrtu]
#
#             # Fallback, если нужно
#             if len(selected_rings) < current_nrtu:
#                 logging.warning(f"[BN Worker P:{pair_idx}] Fallback ring selection ({len(selected_rings)}<{current_nrtu}).")
#                 deterministic_fallback = candidate_rings[:current_nrtu]
#                 needed = current_nrtu - len(selected_rings)
#                 for ring in deterministic_fallback:
#                     if needed == 0: break
#                     if ring not in selected_rings:
#                         selected_rings.append(ring)
#                         needed -= 1
#                 # Если и после fallback не хватает (маловероятно при правильной логике)
#                 if len(selected_rings) < current_nrtu:
#                      logging.error(f"[BN Worker P:{pair_idx}] Fallback failed, not enough rings ({len(selected_rings)}<{current_nrtu}).")
#                      # Устанавливаем nrtu равным числу фактически выбранных колец
#                      current_nrtu = len(selected_rings)
#                      extracted_bits_for_pair = [None] * current_nrtu # Корректируем размер выхода
#                      # Не делаем continue, попробуем извлечь из того, что есть
#
#             # Логируем финально выбранные кольца
#             logging.info(f"[BN Worker P:{pair_idx}] Selected {len(selected_rings)} rings for extraction: {selected_rings}")
#
#             # --- Шаг 3: Извлечение бит из выбранных колец ---
#             # Убедимся, что размер списка бит соответствует числу выбранных колец
#             extracted_bits_for_pair = [None] * len(selected_rings)
#             for i, ring_idx_to_extract in enumerate(selected_rings):
#                  # Передаем действительные L1_tensor, L2_tensor
#                  extracted_bits_for_pair[i] = extract_single_bit(L1_tensor, L2_tensor, ring_idx_to_extract, nr, fn)
#
#             # Если изначально ожидали nrtu бит, а извлекли меньше, дополняем None
#             while len(extracted_bits_for_pair) < nrtu:
#                  extracted_bits_for_pair.append(None)
#
#             batch_results[pair_idx] = extracted_bits_for_pair[:nrtu] # Гарантируем нужный размер
#
#         except cv2.error as cv_err:
#              logging.error(f"OpenCV error P:{pair_idx} in BN worker: {cv_err}", exc_info=True)
#              batch_results[pair_idx] = [None] * nrtu
#         except RuntimeError as torch_err: # Ловим ошибки PyTorch отдельно
#             logging.error(f"PyTorch runtime error P:{pair_idx} in BN worker: {torch_err}", exc_info=True)
#             batch_results[pair_idx] = [None] * nrtu
#         except Exception as e:
#             logging.error(f"Unexpected error processing pair {pair_idx} in BN worker: {e}", exc_info=True)
#             batch_results[pair_idx] = [None] * nrtu
#
#     return batch_results
#


# --- Основная функция извлечения (использует новый _extract_batch_worker) ---
# @profile
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
        bch_code:Optional[BCH_TYPE]=BCH_CODE_OBJECT, # Глобальный объект BCH
        # --- Параметры PyTorch ---
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
    # Проверка наличия PyTorch объектов и библиотек
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        logging.critical("Отсутствуют PyTorch Wavelets или Torch DCT!")
        return None
    if device is None or dtcwt_fwd is None:
         logging.critical("Device или DTCWTForward не переданы в extract_watermark_from_video!")
         return None
    # Проверка ECC, если он требуется
    if ue and expect_hybrid_ecc and not GALOIS_AVAILABLE:
        logging.error("ECC требуется для гибридного режима, но Galois недоступен!")
        # Примечание: код ниже все равно переключится на Raw, но лучше сообщить об ошибке сразу
        # return None # Можно раскомментировать для прерывания

    logging.info(f"--- Starting Extraction (PyTorch BN, Hybrid Expected: {expect_hybrid_ecc}, Max Packets: {max_expected_packets}) ---")
    start_time = time.time()
    nf = len(frames)
    total_pairs_available = nf // 2

    if total_pairs_available == 0:
        logging.error("Нет пар кадров для обработки.")
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
                # Ошибка конфигурации, если ожидали гибрид
                if expect_hybrid_ecc:
                     logging.error("Cannot proceed with hybrid mode: ECC not possible for first packet due to k.")
                     return None
        except AttributeError: logging.warning("ECC check: BCH object is dummy."); ecc_possible_for_first = False
        except Exception as e: logging.error(f"ECC check: Error getting Galois params: {e}."); ecc_possible_for_first = False
    else:
        logging.info("ECC check: Disabled or unavailable for first packet.")

    # Корректируем флаг ожидания гибрида, если ECC невозможен
    if expect_hybrid_ecc and not ecc_possible_for_first:
        logging.warning("Hybrid mode requested but ECC not possible/available. Switching to Raw mode for all packets.")
        expect_hybrid_ecc = False # Все пакеты будут Raw

    # --- Вычисляем, сколько пар кадров нужно обработать ---
    max_possible_bits = 0
    if expect_hybrid_ecc: # Значит ecc_possible_for_first тоже True
        max_possible_bits = packet_len_if_ecc + max(0, max_expected_packets - 1) * packet_len_if_raw
    else: # Ожидаем только один тип пакета
        current_packet_len_for_calc = packet_len_if_ecc if ue and ecc_possible_for_first else packet_len_if_raw
        max_possible_bits = max_expected_packets * current_packet_len_for_calc

    if bp <= 0: logging.error("Bits per pair (bp) must be positive."); return None
    pairs_needed = ceil(max_possible_bits / bp) if max_possible_bits > 0 else 0
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
                'device': device, 'dtcwt_fwd': dtcwt_fwd} # Передаем объекты PyTorch
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0: logging.warning(f"Skipped {skipped_pairs} pairs during task preparation.")
    if num_valid_tasks == 0: logging.error("No valid extraction tasks generated."); return None

    # --- Запуск ThreadPoolExecutor ---
    num_workers = mw if mw is not None and mw > 0 else (os.cpu_count() or 1)
    # Используем меньший размер батча для потенциально лучшего баланса нагрузки
    batch_size = max(1, ceil(num_valid_tasks / (num_workers * 4)))
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i : i + batch_size] for i in range(0, num_valid_tasks, batch_size) if all_pairs_args[i:i+batch_size]] # Убираем пустые батчи
    actual_num_batches = len(batched_args_list)

    logging.info(f"Launching {actual_num_batches} batches ({num_valid_tasks} pairs) using ThreadPool (mw={num_workers}, batch_size≈{batch_size})...")

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

    logging.info(f"Extraction task processing finished. Pairs processed: {ppc}. Pairs with errors/Nones: {fpe}.")
    if ppc == 0: logging.error("No pairs processed successfully by workers."); return None

    # --- Сборка общего потока извлеченных бит ---
    extracted_bits_all: List[Optional[int]] = []
    for pair_idx in range(pairs_to_process): # Итерируем по всем парам, что ДОЛЖНЫ были обработаться
        bits = extracted_bits_map.get(pair_idx)
        # Добавляем результат или None-заполнители
        if bits and isinstance(bits, list) and len(bits) == bp: # Проверяем тип и длину
            extracted_bits_all.extend(bits)
        else:
             if bits is None: logging.debug(f"Pair {pair_idx}: No result in map, adding Nones.")
             else: logging.warning(f"Pair {pair_idx}: Incorrect bits format/len ({type(bits)}, len={len(bits) if isinstance(bits, list) else 'N/A'}). Adding Nones.")
             extracted_bits_all.extend([None] * bp)

    total_bits_collected = len(extracted_bits_all)
    # Фильтруем None и невалидные значения (не 0 и не 1)
    valid_bits = [b for b in extracted_bits_all if b is not None and b in (0, 1)]
    num_valid_bits = len(valid_bits)
    num_error_bits = total_bits_collected - num_valid_bits # Количество None или не 0/1
    success_rate = (num_valid_bits / total_bits_collected) * 100 if total_bits_collected > 0 else 0
    logging.info(f"Bit collection: Total attempted={total_bits_collected}, Valid(0/1)={num_valid_bits} ({success_rate:.1f}%), Error/None={num_error_bits}.")
    if not valid_bits: logging.error("No valid (0/1) bits extracted."); return None

    # --- Гибридное Декодирование Пакетов ---
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
        # Важно: Сдвигаем указатель *после* проверки, чтобы не выйти за границы при последнем пакете
        num_processed_bits += current_packet_len

        # --- Пытаемся получить payload ---
        payload_bytes: Optional[bytes] = None
        payload_bits: Optional[List[int]] = None
        errors: int = -1
        status_str = f"Failed ({packet_type_str})" # Статус по умолчанию
        payload_hex_str = "N/A"

        if use_ecc_for_this:
            if bch_code is not None: # Проверка на всякий случай
                payload_bytes, errors = decode_ecc(packet_candidate_bits, bch_code, plb)
                if payload_bytes is not None:
                    corrected_count = errors if errors != -1 else 0 # Если -1, то неисправимо, но payload получен (странно, но обработаем)
                    status_str = f"OK (ECC: {corrected_count} fixed)"
                    if errors > 0: total_corrected_symbols += errors # Суммируем только явно > 0
                else:
                    # errors тут будет -1 если неисправимо
                    status_str = f"Uncorrectable(ECC)" if errors == -1 else "ECC Decode Error"
            else: status_str = "ECC Code Missing" # Сюда не должны попадать
        else: # Raw
            # Проверяем, достаточно ли бит для raw payload
            if len(packet_candidate_bits) >= payload_len_bits:
                 payload_candidate_bits_raw = packet_candidate_bits[:payload_len_bits]
                 packet_bytes_raw = bits_to_bytes(payload_candidate_bits_raw)
                 if packet_bytes_raw is not None and len(packet_bytes_raw) == plb:
                     payload_bytes = packet_bytes_raw; errors = 0; status_str = "OK (Raw)"
                 else: status_str = "Failed (Raw Convert)"
            else: status_str = "Failed (Raw Short)" # Не хватило бит

        # --- Получаем биты из байт (если удалось) ---
        if payload_bytes is not None:
            payload_hex_str = payload_bytes.hex() # Показываем hex, даже если unpack не удастся
            try:
                payload_np_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
                if len(payload_np_bits) == payload_len_bits:
                    payload_bits = payload_np_bits.tolist(); decoded_success_count += 1
                else:
                    logging.warning(f"Packet {i+1} unpacked to len {len(payload_np_bits)} != {payload_len_bits}.")
                    status_str += "[Len Fail]"; payload_bits = None # Считаем неудачей
            except Exception as e_unpack:
                logging.error(f"Error unpacking bits for packet {i+1}: {e_unpack}")
                status_str += "[Unpack Fail]"; payload_bits = None # Считаем неудачей
        # else: # Если payload_bytes None, то payload_bits уже None

        # Увеличиваем счетчик неудачи, если payload_bits все еще None
        if payload_bits is None:
             decode_failed_count += 1

        # Добавляем результат (payload_bits или None) в список
        all_payload_attempts_bits.append(payload_bits)

        # Выводим строку таблицы
        corrected_str = str(errors) if errors >= 0 else "-" # Используем '-' для неисправимых/ошибок/raw
        print(f"{i+1:<6} | {packet_type_str:<7} | {status_str:<18} | {corrected_str:<10} | {payload_hex_str:<20}")

    # --- Конец цикла декодирования ---
    print("-" * 68)
    logging.info(f"Decode attempts summary: Total attempted packets = {len(all_payload_attempts_bits)}, Success (yielded {payload_len_bits} bits) = {decoded_success_count}, Failed/Uncorrectable = {decode_failed_count}.")
    if ecc_possible_for_first and expect_hybrid_ecc:
         logging.info(f"Total ECC corrections reported for first packet: {total_corrected_symbols}.")

    # --- Побитовое Голосование с Приоритетом Первого Пакета ---
    num_attempted_packets = len(all_payload_attempts_bits)
    if num_attempted_packets == 0:
        logging.error("No packets were attempted for voting."); return None
    # Получаем результат первого пакета ДО фильтрации None
    first_packet_payload = all_payload_attempts_bits[0] if num_attempted_packets > 0 else None

    # Фильтруем валидные пакеты *только для самого голосования*
    valid_decoded_payloads = [p for p in all_payload_attempts_bits if p is not None and len(p) == payload_len_bits]
    num_valid_packets_for_vote = len(valid_decoded_payloads)

    if num_valid_packets_for_vote == 0:
        logging.error(f"No valid {payload_len_bits}-bit payloads available for voting."); return None

    final_payload_bits = []
    logging.info(f"Performing bit-wise majority vote across {num_valid_packets_for_vote} valid packets (tie-break to first packet attempt if valid)...")
    print("\n--- Bit-wise Voting Details ---"); print(f"{'Bit Pos':<8} | {'Votes 0':<8} | {'Votes 1':<8} | {'Winner':<8} | {'Tiebreak?':<10}"); print("-" * 50)

    for j in range(payload_len_bits): # Итерация по позициям бит
        votes_for_0 = 0; votes_for_1 = 0
        # Голосуем только по валидным пакетам
        for i in range(num_valid_packets_for_vote):
            # Проверка индекса (хотя не должна быть нужна после проверки длины payload_bits)
            if j < len(valid_decoded_payloads[i]):
                 if valid_decoded_payloads[i][j] == 1: votes_for_1 += 1
                 else: votes_for_0 += 1
            else: logging.warning(f"Index error during voting: bit {j}, valid_pkt_idx {i}")

        winner_bit: Optional[int] = None; tiebreak_used = "No"
        valid_votes_count = votes_for_0 + votes_for_1 # Общее число голосов за эту позицию

        if valid_votes_count == 0: # Если вдруг все валидные пакеты имели неверную длину
             logging.error(f"Bit position {j}: No valid votes found! Voting failed.")
             print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {'FAIL':<8} | {'N/A':<10}")
             final_payload_bits = None; break # Прерываем голосование
        elif votes_for_1 > votes_for_0: winner_bit = 1
        elif votes_for_0 > votes_for_1: winner_bit = 0
        else: # Ничья
            tiebreak_used = "Yes"; logging.warning(f"Bit position {j}: Tie ({votes_for_0} vs {votes_for_1}). Using first pkt attempt.")
            # Используем результат первой *попытки* декодирования (может быть None)
            if first_packet_payload is not None and j < len(first_packet_payload):
                 winner_bit = first_packet_payload[j]
            else:
                 logging.error(f"Cannot resolve tie for bit {j}: First packet attempt invalid or index error!"); print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {'FAIL':<8} | {tiebreak_used:<10}"); final_payload_bits = None; break

        if winner_bit is None: logging.error(f"Winner bit None for pos {j}"); final_payload_bits = None; break

        final_payload_bits.append(winner_bit)
        print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {winner_bit:<8} | {tiebreak_used:<10}")

    # --- Конец цикла голосования ---
    print("-" * 50)
    if final_payload_bits is None: logging.error("Voting failed."); return None;
    logging.info(f"Voting complete.")

    # --- Конвертация и возврат результата ---
    final_payload_bytes = bits_to_bytes(final_payload_bits)
    if final_payload_bytes is None: logging.error("bits_to_bytes failed."); return None
    if len(final_payload_bytes) != plb:
        logging.error(f"Final length mismatch: {len(final_payload_bytes)}B != {plb}B.")
        # Попытка обрезать, если был паддинг в bits_to_bytes (маловероятно для 64 бит)
        if len(final_payload_bytes) > plb and payload_len_bits % 8 == 0 : # Если исходное число бит кратно 8, обрезка не нужна
             return None
        elif len(final_payload_bytes) > plb and len(final_payload_bytes) - (len(final_payload_bytes) % plb) == plb:
             logging.warning(f"Attempting trim final payload due to padding.")
             final_payload_bytes = final_payload_bytes[:plb]
        else: return None # Ошибка длины
    logging.info(f"Final ID after bit-wise voting: {final_payload_bytes.hex()}")
    end_time = time.time(); logging.info(f"Extraction done. Total time: {end_time - start_time:.2f} sec.")
    return final_payload_bytes

# --- Функция main (Адаптированная для PyTorch BN) ---
def main():
    start_time_main = time.time()
    # --- Инициализация PyTorch и Galois ---
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE: print("ERROR: PyTorch libraries required."); return
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else: logging.info("Using CPU.")
    # Создаем экземпляр DTCWTForward (DTCWTInverse больше не нужен в экстракторе)
    dtcwt_fwd: Optional[DTCWTForward] = None
    try:
        dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
        logging.info("PyTorch DTCWTForward instance created.")
    except Exception as e: logging.critical(f"Failed to init DTCWTForward: {e}"); return

    # --- Имя входного файла ---
    input_base = f"output_h264_28" # Имя файла зависит от BCH_T
    input_video = input_base + INPUT_EXTENSION
    original_id = None

    # Загрузка оригинального ID
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r") as f: original_id = f.read().strip()
            assert original_id and len(original_id) == PAYLOAD_LEN_BYTES * 2; int(original_id, 16)
            logging.info(f"Original ID loaded: {original_id}")
        except Exception as e: logging.error(f"Read/Validate original ID failed: {e}"); original_id = None
    else: logging.warning(f"'{ORIGINAL_WATERMARK_FILE}' not found.")

    logging.info(f"--- Starting Extraction Main Process (PyTorch BN) ---")
    if not os.path.exists(input_video): logging.critical(f"Input video missing: '{input_video}'."); print(f"ERROR: Input missing: '{input_video}'"); return

    frames, fps_read = read_video(input_video)
    if not frames: logging.critical("Video read failed."); return
    logging.info(f"Read {len(frames)} frames.")

    # --- Вызов основной функции извлечения ---
    extracted_bytes = extract_watermark_from_video(
        frames=frames,
        nr=N_RINGS, nrtu=NUM_RINGS_TO_USE, bp=BITS_PER_PAIR,
        cps=CANDIDATE_POOL_SIZE, ec=EMBED_COMPONENT,
        expect_hybrid_ecc=True,     # Ожидаем гибридный режим
        max_expected_packets=15,    # Макс. пакетов
        ue=USE_ECC, bch_code=BCH_CODE_OBJECT, # Передаем объект BCH
        device=device, dtcwt_fwd=dtcwt_fwd,   # Передаем объекты PyTorch
        plb=PAYLOAD_LEN_BYTES, mw=MAX_WORKERS_EXTRACT
    )

    # --- Вывод результатов ---
    print(f"\n--- Extraction Results ---"); extracted_hex = None
    if extracted_bytes:
        if len(extracted_bytes) == PAYLOAD_LEN_BYTES: extracted_hex = extracted_bytes.hex(); print(f"  Payload Length OK."); print(f"  Decoded ID (Hex): {extracted_hex}"); logging.info(f"Decoded ID: {extracted_hex}")
        else: print(f"  ERROR: Decoded length mismatch! {len(extracted_bytes)}B != {PAYLOAD_LEN_BYTES}B."); logging.error(f"Decoded length mismatch!")
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
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE: print("ERROR: PyTorch libraries required."); sys.exit(1)
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")
    # --- Профилирование и запуск main ---
    DO_PROFILING = False # Включить/выключить
    prof = None
    if DO_PROFILING: prof = cProfile.Profile(); prof.enable(); logging.info("cProfile enabled.")
    final_exit_code = 0
    try: main()
    except FileNotFoundError as e: print(f"\nERROR: {e}"); logging.error(f"{e}"); final_exit_code = 1
    except Exception as e: logging.critical(f"Unhandled main: {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. Log: {LOG_FILENAME}"); final_exit_code = 1
    finally: # Сохранение профиля
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
    sys.exit(final_exit_code)

# --- Функция main (Адаптированная для PyTorch) ---
def main():
    start_time_main = time.time()
    # Инициализация PyTorch и Galois
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE: print("ERROR: Required PyTorch libraries not found."); return
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else: logging.info("Using CPU.")
    # Создаем экземпляр DTCWTForward
    dtcwt_fwd: Optional[DTCWTForward] = None
    try:
        dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
        logging.info("PyTorch DTCWTForward instance created.")
    except Exception as e: logging.critical(f"Failed to init DTCWTForward: {e}"); return

    # Имя входного файла
    input_base = f"watermarked_pytorch_hybrid_t{BCH_T}" # Имя должно совпадать с выходом embedder
    input_video = input_base + INPUT_EXTENSION
    original_id = None

    # Загрузка оригинального ID
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r") as f: original_id = f.read().strip()
            assert original_id and len(original_id) == PAYLOAD_LEN_BYTES * 2; int(original_id, 16)
            logging.info(f"Original ID loaded: {original_id}")
        except Exception as e: logging.error(f"Read/Validate original ID failed: {e}"); original_id = None
    else: logging.warning(f"'{ORIGINAL_WATERMARK_FILE}' not found.")

    logging.info(f"--- Starting Extraction Main Process (PyTorch BN) ---")
    if not os.path.exists(input_video): logging.critical(f"Input video missing: '{input_video}'."); print(f"ERROR: Input missing: '{input_video}'"); return

    frames, fps_read = read_video(input_video)
    if not frames: logging.critical("Video read failed."); return
    logging.info(f"Read {len(frames)} frames.")

    # --- Вызов основной функции извлечения ---
    extracted_bytes = extract_watermark_from_video(
        frames=frames,
        nr=N_RINGS, nrtu=NUM_RINGS_TO_USE, bp=BITS_PER_PAIR,
        cps=CANDIDATE_POOL_SIZE, ec=EMBED_COMPONENT,
        expect_hybrid_ecc=True,     # <--- Указываем, что ожидаем гибрид
        max_expected_packets=15,    # <--- Макс. пакетов
        ue=USE_ECC, bch_code=BCH_CODE_OBJECT, # Передаем объект BCH
        device=device, dtcwt_fwd=dtcwt_fwd,   # Передаем объекты PyTorch
        plb=PAYLOAD_LEN_BYTES, mw=MAX_WORKERS_EXTRACT
    )

    # --- Вывод результатов ---
    print(f"\n--- Extraction Results ---"); extracted_hex = None
    if extracted_bytes:
        if len(extracted_bytes) == PAYLOAD_LEN_BYTES: extracted_hex = extracted_bytes.hex(); print(f"  Payload Length OK."); print(f"  Decoded ID (Hex): {extracted_hex}"); logging.info(f"Decoded ID: {extracted_hex}")
        else: print(f"  ERROR: Decoded length mismatch! {len(extracted_bytes)}B != {PAYLOAD_LEN_BYTES}B."); logging.error(f"Decoded length mismatch!")
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
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE: print("ERROR: PyTorch libraries required."); sys.exit(1)
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")
    # --- Профилирование и запуск main ---
    DO_PROFILING = False # Включить/выключить
    prof = None
    if DO_PROFILING: prof = cProfile.Profile(); prof.enable(); logging.info("cProfile enabled.")
    final_exit_code = 0
    try: main()
    except FileNotFoundError as e: print(f"\nERROR: {e}"); logging.error(f"{e}"); final_exit_code = 1
    except Exception as e: logging.critical(f"Unhandled main: {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. Log: {LOG_FILENAME}"); final_exit_code = 1
    finally: # Сохранение профиля
        if DO_PROFILING and prof is not None:
             prof.disable(); stats = pstats.Stats(prof)
             print("\n--- Profiling Stats (Top 30 Cumulative Time) ---")
             try: stats.strip_dirs().sort_stats("cumulative").print_stats(30)
             except Exception as e_stats: print(f"Error printing stats: {e_stats}")
             print("-------------------------------------------------")
             pfile = f"profile_extract_pytorch_hybrid_t{BCH_T}.txt"
             try:
                 with open(pfile, "w", encoding='utf-8') as f: sf = pstats.Stats(prof, stream=f); sf.strip_dirs().sort_stats("cumulative").print_stats()
                 logging.info(f"Profiling stats saved: {pfile}"); print(f"Profiling stats saved: {pfile}")
             except IOError as e: logging.error(f"Save profile failed: {e}")
    sys.exit(final_exit_code)
