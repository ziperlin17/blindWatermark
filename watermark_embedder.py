# Файл: embedder.py (Версия: ThreadPool + Batches, Galois BCH, m=8, t=4, Syntax Fix V8)
import cv2
import numpy as np
import random
import logging
import time
import concurrent.futures # Используем concurrent.futures
from concurrent.futures import ThreadPoolExecutor # Конкретно ThreadPoolExecutor
import json
import os
# import imagehash
import hashlib
from PIL import Image
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools
import uuid
from math import ceil
import cProfile
import pstats

# --- Попытка импорта и инициализации Galois ---
try:
    import galois
    print("galois: импортирован.")
    _test_bch_ok = False
    _test_encode_ok = False
    BCH_CODE_OBJECT = None
    try:
        _test_m = 8; _test_t = 4; _test_n = (1 << _test_m) - 1; _test_d = 2 * _test_t + 1
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)
        if _test_bch_galois.t == _test_t:
             print(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) initialized OK.")
             _test_bch_ok = True
             BCH_CODE_OBJECT = _test_bch_galois
        else:
             print(f"galois BCH init mismatch: expected t={_test_t}, got t={_test_bch_galois.t}")
             _test_bch_ok = False; BCH_CODE_OBJECT = None
        if _test_bch_ok:
            _k_bits = _test_bch_galois.k
            _dummy_msg_bits = np.zeros(_k_bits, dtype=np.uint8)
            GF2 = galois.GF(2)
            _dummy_msg_vec = GF2(_dummy_msg_bits)
            _codeword = _test_bch_galois.encode(_dummy_msg_vec)
            print("galois: encode() test OK.")
            _test_encode_ok = True
    except Exception as test_err:
         print(f"galois: ОШИБКА при инициализации/тесте: {test_err}")
         BCH_CODE_OBJECT = None
    GALOIS_AVAILABLE = _test_bch_ok and _test_encode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None
    if GALOIS_AVAILABLE: print("galois: Тесты инициализации и encode пройдены.")
    else: print("galois: Не прошел базовые тесты. ECC будет отключен.")
except ImportError: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; print("galois library not found.")
except Exception as import_err: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; print(f"galois: Ошибка импорта: {import_err}")

# --- Основные Параметры ---
LAMBDA_PARAM: float = 0.1
ALPHA_MIN: float = 1.005
ALPHA_MAX: float = 1.12
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
BCH_T: int = 4
MAX_PACKET_REPEATS: int = 5
FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed_threaded_batched.log' # Новое имя лога
OUTPUT_CODEC: str = 'XVID'
OUTPUT_EXTENSION: str = '.avi'
SELECTED_RINGS_FILE: str = 'selected_rings_embed_threaded_batched.json' # Новое имя файла
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
# Параметр для ThreadPoolExecutor
MAX_WORKERS: Optional[int] = None # None -> Python выберет оптимальное число потоков

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
# Возвращаем информацию о потоке
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование Конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Встраивания (ThreadPool + Batches) ---")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Payload: {PAYLOAD_LEN_BYTES * 8}bit, ECC: {effective_use_ecc} (Galois BCH m={BCH_M}, t={BCH_T}), Max Repeats: {MAX_PACKET_REPEATS}")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(f"Маскировка: {USE_PERCEPTUAL_MASKING} (Lambda={LAMBDA_PARAM}), Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS or 'default'}) с батчингом.")
if USE_ECC and not GALOIS_AVAILABLE: logging.warning("ECC вкл, но galois недоступна/не работает! Встраивание БЕЗ ECC.")
elif not USE_ECC: logging.info("ECC выкл.")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG', 'DIVX']: logging.warning(f"Используется кодек с потерями '{OUTPUT_CODEC}'.")
# ... остальные проверки параметров ...

# --- Базовые Функции (dct_1d, idct_1d, dtcwt_transform, dtcwt_inverse, ring_division, etc.) ---
# --- Эти функции остаются БЕЗ ИЗМЕНЕНИЙ по сравнению с предыдущей полной версией ---
# --- (Копирую их сюда для полноты файла) ---
def dct_1d(s: np.ndarray) -> np.ndarray:
    return dct(s, type=2, norm='ortho')

def idct_1d(c: np.ndarray) -> np.ndarray:
    return idct(c, type=2, norm='ortho')

def dtcwt_transform(yp: np.ndarray, fn: int = -1) -> Optional[Pyramid]:
    if np.any(np.isnan(yp)): logging.warning(f"[Frame:{fn}] Input DTCWT contains NaN!")
    try:
        t = Transform2d(); r, c = yp.shape; pr = r % 2 != 0; pc = c % 2 != 0
        ypp = np.pad(yp, ((0, pr), (0, pc)), mode='reflect') if pr or pc else yp
        py = t.forward(ypp.astype(np.float32), nlevels=1)
        if hasattr(py, 'lowpass') and py.lowpass is not None: py.padding_info = (pr, pc); return py
        else: logging.error(f"[Frame:{fn}] DTCWT no lowpass."); return None
    except Exception as e: logging.error(f"[Frame:{fn}] DTCWT forward error: {e}", exc_info=True); return None

def dtcwt_inverse(py: Pyramid, fn: int = -1) -> Optional[np.ndarray]:
    if not isinstance(py, Pyramid) or not hasattr(py, 'lowpass'): return None
    try:
        t = Transform2d(); rp = t.inverse(py).astype(np.float32)
        pr, pc = getattr(py, 'padding_info', (False, False))
        ry = rp[:rp.shape[0]-pr, :rp.shape[1]-pc] if pr or pc else rp
        if np.any(np.isnan(ry)): logging.warning(f"[Frame:{fn}] NaN after inverse DTCWT!")
        return ry
    except Exception as e: logging.error(f"[Frame:{fn}] DTCWT inverse error: {e}", exc_info=True); return None

@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    H, W = subband_shape
    if H < 2 or W < 2: return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2)
    min_dist, max_dist = 0.0, np.max(distances)
    if max_dist < 1e-6: ring_bins = np.array([0.0, max_dist + 1e-6]); n_rings_eff = 1
    else: ring_bins = np.linspace(min_dist, max_dist + 1e-6, n_rings + 1); n_rings_eff = n_rings
    if len(ring_bins) < 2: return [None] * n_rings
    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    ring_coords_list: List[Optional[np.ndarray]] = [None] * n_rings
    for rdx in range(n_rings_eff):
        coords = np.argwhere(ring_indices == rdx)
        if coords.shape[0] > 0: ring_coords_list[rdx] = coords
    return ring_coords_list

@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(ss: Tuple[int, int], nr: int) -> List[Optional[np.ndarray]]:
    return _ring_division_internal(ss, nr)

def ring_division(lp: np.ndarray, nr: int = N_RINGS, fn: int = -1) -> List[Optional[np.ndarray]]:
    if not isinstance(lp, np.ndarray) or lp.ndim != 2: return [None] * nr
    sh = lp.shape
    try:
        cached_list = get_ring_coords_cached(sh, nr)
        if not isinstance(cached_list, list) or not all(isinstance(i, (np.ndarray, type(None))) for i in cached_list):
            get_ring_coords_cached.cache_clear(); cached_list = _ring_division_internal(sh, nr)
        return [a.copy() if a is not None else None for a in cached_list]
    except Exception as e: logging.error(f"[Frame:{fn}] Ring division error: {e}"); return [None]*nr

def calculate_entropies(ring_values: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    eps = 1e-12; shannon_entropy = 0.0; exp_entropy = 0.0
    if ring_values.size > 0:
        min_val, max_val = np.min(ring_values), np.max(ring_values)
        rvc = np.clip(ring_values, 0.0, 1.0) if min_val < 0.0 or max_val > 1.0 else ring_values
        hist, _ = np.histogram(rvc, bins=256, range=(0., 1.), density=False); tc = rvc.size
        if tc > 0:
            p = hist / tc; p = p[p > eps]
            if p.size > 0:
                shannon_entropy = -np.sum(p * np.log2(p))
                exp_entropy = -np.sum(p * np.exp(1. - p))
    return shannon_entropy, exp_entropy

def compute_adaptive_alpha_entropy(ring_values: np.ndarray, ring_index: int, frame_number: int) -> float:
    if ring_values.size < 10: return ALPHA_MIN
    shannon_entropy, _ = calculate_entropies(ring_values, frame_number, ring_index)
    local_variance = np.var(ring_values)
    entropy_norm = np.clip(shannon_entropy / MAX_THEORETICAL_ENTROPY, 0.0, 1.0)
    vmp = 0.005; vsc = 500
    texture_norm = 1.0 / (1.0 + np.exp(-vsc * (local_variance - vmp)))
    we = 0.6; wt = 0.4
    mf = np.clip((we * entropy_norm + wt * texture_norm), 0.0, 1.0)
    final_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * mf
    logging.debug(f"[F:{frame_number}, R:{ring_index}] Alpha Calc: E={shannon_entropy:.3f}, V={local_variance:.6f}, MF={mf:.3f} -> alpha={final_alpha:.4f}")
    return np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)

def get_fixed_pseudo_random_rings(pair_index: int, num_total_rings: int, pool_size: int) -> List[int]:
    if pool_size <= 0: return []
    if pool_size > num_total_rings: pool_size = num_total_rings
    seed_str = str(pair_index).encode('utf-8'); hd = hashlib.sha256(seed_str).digest(); sv = int.from_bytes(hd, 'big')
    prng = random.Random(sv)
    try: candidate_indices = prng.sample(range(num_total_rings), pool_size)
    except ValueError: candidate_indices = list(range(num_total_rings))
    logging.debug(f"[Pair:{pair_index}] Generated candidate rings: {candidate_indices}")
    return candidate_indices

def calculate_perceptual_mask(input_plane: np.ndarray, fn: int = -1) -> Optional[np.ndarray]:
    if not isinstance(input_plane, np.ndarray) or input_plane.ndim != 2: return None
    try:
        pf = input_plane.astype(np.float32); gx = cv2.Sobel(pf, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(pf, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx**2 + gy**2); ks = (11, 11); s = 5; lm = cv2.GaussianBlur(pf, ks, s); lms = cv2.GaussianBlur(pf**2, ks, s)
        lv = np.sqrt(np.maximum(lms - lm**2, 0)); cm = np.maximum(gm, lv); eps = 1e-9; mv = np.max(cm)
        mn = cm / (mv + eps) if mv > eps else np.zeros_like(cm); mn = np.clip(mn, 0.0, 1.0)
        logging.debug(f"[Frame:{fn}] Perceptual mask calculated. Range: [{np.min(mn):.3f}-{np.max(mn):.3f}], Mean: {np.mean(mn):.3f}")
        return mn.astype(np.float32)
    except Exception as e: logging.error(f"[Frame:{fn}] Mask error: {e}"); return np.ones_like(input_plane, dtype=np.float32)

def add_ecc(data_bits: np.ndarray, bch_code: galois.BCH) -> Optional[np.ndarray]:
    if not GALOIS_AVAILABLE or bch_code is None: return data_bits
    k = bch_code.k; n = bch_code.n
    if data_bits.size > k: logging.error(f"Data({data_bits.size})>k({k})"); return None
    pad_len = k - data_bits.size
    msg_bits = np.pad(data_bits, (0, pad_len), 'constant') if pad_len > 0 else data_bits
    try:
        GF = bch_code.field; msg_vec = GF(msg_bits.astype(np.uint8)); cw_vec = bch_code.encode(msg_vec)
        pkt_bits = cw_vec.view(np.ndarray).astype(np.uint8)
        if pkt_bits.size != n: logging.error(f"Galois size {pkt_bits.size}!=n {n}"); return None
        logging.info(f"Galois ECC: Data({data_bits.size}b->{k}b) -> Packet({pkt_bits.size}b).")
        return pkt_bits
    except Exception as e: logging.error(f"Galois encode error: {e}"); return None

def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    logging.info(f"Reading video: {video_path}")
    frames: List[np.ndarray] = []; fps = float(FPS); cap = None; h, w = -1, -1
    try:
        if not os.path.exists(video_path): raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path);
        if not cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); logging.info(f"Props: {w}x{h} @ {fps:.2f} FPS, ~{fc} frames.")
        rc, nc, ic = 0, 0, 0
        while True:
            ret, f = cap.read();
            if not ret: break
            if f is None: nc+=1; continue
            if f.ndim == 3 and f.shape[:2] == (h, w) and f.dtype == np.uint8: frames.append(f); rc+=1
            else: logging.warning(f"Skipped invalid frame #{rc+nc+ic}: shape={f.shape}, dtype={f.dtype}"); ic+=1
        logging.info(f"Read done. Valid: {rc}, None: {nc}, Invalid: {ic}")
    except Exception as e: logging.error(f"Read video error: {e}", exc_info=True)
    finally:
        if cap: cap.release()
    if not frames: logging.error(f"No valid frames read from {video_path}")
    return frames, fps

def write_video(frames: List[np.ndarray], out_path: str, fps: float, codec: str = OUTPUT_CODEC):
    if not frames: logging.error("No frames to write."); return
    logging.info(f"Writing video: {out_path} (FPS:{fps:.2f}, Codec:{codec})")
    writer = None
    try:
        first_valid = next((f for f in frames if f is not None and f.ndim == 3), None)
        if first_valid is None: logging.error("No valid frames to get size."); return
        h, w, c = first_valid.shape; logging.info(f"Output res: {w}x{h}")
        fourcc = cv2.VideoWriter_fourcc(*codec); base, _ = os.path.splitext(out_path); out_path_corrected = base + OUTPUT_EXTENSION
        if out_path_corrected != out_path: logging.info(f"Correcting ext to '{OUTPUT_EXTENSION}'. Path: {out_path_corrected}"); out_path = out_path_corrected
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h)); wc = codec
        if not writer.isOpened():
            logging.error(f"Failed codec {codec}.")
            if OUTPUT_EXTENSION.lower() == '.avi' and codec.upper() != 'MJPG':
                fbk = 'MJPG'; logging.warning(f"Fallback to {fbk}."); fourcc_fbk = cv2.VideoWriter_fourcc(*fbk)
                writer = cv2.VideoWriter(out_path, fourcc_fbk, fps, (w, h)); wc = fbk
            if not writer.isOpened(): logging.critical("Writer failed."); return
        wc_ok, sc_ok = 0, 0; bf = np.zeros((h, w, 3), dtype=np.uint8)
        for i, frame in enumerate(frames):
            if frame is not None and frame.shape[:2] == (h, w) and frame.dtype == np.uint8: writer.write(frame); wc_ok+=1
            else: logging.warning(f"Skipping invalid frame #{i}. Writing black."); writer.write(bf); sc_ok+=1
        logging.info(f"Write done ({wc}). Written: {wc_ok}, Skipped/Black: {sc_ok}")
    except Exception as e: logging.error(f"Video write error: {e}", exc_info=True)
    finally:
        if writer: writer.release()


# --- Функция встраивания в пару кадров (embed_frame_pair) ---
# --- Остается БЕЗ ИЗМЕНЕНИЙ (с исправленным восстановлением DCT) ---
def embed_frame_pair(
        frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bits: List[int],
        selected_ring_indices: List[int], n_rings: int = N_RINGS, frame_number: int = 0,
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING, embed_component: int = EMBED_COMPONENT
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Встраивает биты в выбранные кольца пары кадров. (Версия с исправленным восстановлением DCT)"""
    pair_index = frame_number // 2
    if len(bits) != len(selected_ring_indices):
        logging.error(f"[P:{pair_index}] Mismatch bits ({len(bits)}) and rings ({len(selected_ring_indices)}).")
        return None, None
    if not bits: return frame1_bgr, frame2_bgr

    try:
        if frame1_bgr is None or frame2_bgr is None or frame1_bgr.shape != frame2_bgr.shape: return None, None
        try: f1_ycrcb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb); f2_ycrcb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2YCrCb)
        except cv2.error as e: logging.error(f"[P:{pair_index}] BGR->YCrCb Error: {e}"); return None, None
        try:
            comp1 = f1_ycrcb[:, :, embed_component].astype(np.float32)/255.0; comp2 = f2_ycrcb[:, :, embed_component].astype(np.float32)/255.0
            Y1, Cr1, Cb1 = f1_ycrcb[:,:,0], f1_ycrcb[:,:,1], f1_ycrcb[:,:,2]; Y2, Cr2, Cb2 = f2_ycrcb[:,:,0], f2_ycrcb[:,:,1], f2_ycrcb[:,:,2]
        except IndexError: logging.error(f"[P:{pair_index}] Invalid component index: {embed_component}."); return None, None

        pyramid1 = dtcwt_transform(comp1, frame_number); pyramid2 = dtcwt_transform(comp2, frame_number + 1)
        if pyramid1 is None or pyramid2 is None or pyramid1.lowpass is None or pyramid2.lowpass is None: return None, None
        L1 = pyramid1.lowpass.copy(); L2 = pyramid2.lowpass.copy()
        ring_coords1 = ring_division(L1, n_rings, frame_number); ring_coords2 = ring_division(L2, n_rings, frame_number + 1)
        perceptual_mask = None
        if use_perceptual_masking:
            mask_full_res = calculate_perceptual_mask(comp1, frame_number)
            if mask_full_res is not None:
                 if mask_full_res.shape != L1.shape: perceptual_mask = cv2.resize(mask_full_res, (L1.shape[1], L1.shape[0]), interpolation=cv2.INTER_LINEAR)
                 else: perceptual_mask = mask_full_res
            else: perceptual_mask = np.ones_like(L1, dtype=np.float32)

        modifications_count = 0
        for ring_idx, bit_to_embed in zip(selected_ring_indices, bits):
            if not (0 <= ring_idx < n_rings): continue
            try: coords1 = ring_coords1[ring_idx]; coords2 = ring_coords2[ring_idx]
            except IndexError: continue
            if coords1 is None or coords2 is None: continue
            rows1, cols1 = coords1[:, 0], coords1[:, 1]; rows2, cols2 = coords2[:, 0], coords2[:, 1]
            values1 = L1[rows1, cols1].astype(np.float32); values2 = L2[rows2, cols2].astype(np.float32)
            if values1.size == 0 or values2.size == 0: continue
            if values1.size != values2.size:
                min_size = min(values1.size, values2.size);
                if min_size == 0: continue;
                logging.warning(f"[P:{pair_index}, R:{ring_idx}] Ring size mismatch ({values1.size} vs {values2.size}), using min {min_size}.")
                values1 = values1[:min_size]; values2 = values2[:min_size]; rows1, cols1 = rows1[:min_size], cols1[:min_size]; rows2, cols2 = rows2[:min_size], cols2[:min_size]

            alpha = compute_adaptive_alpha_entropy(values1, ring_idx, frame_number)
            dct_coeffs1 = dct_1d(values1); dct_coeffs2 = dct_1d(values2)
            try: U1, S1_vec, Vt1 = svd(dct_coeffs1.reshape(-1, 1), full_matrices=False); U2, S2_vec, Vt2 = svd(dct_coeffs2.reshape(-1, 1), full_matrices=False)
            except np.linalg.LinAlgError: continue
            s1 = S1_vec[0] if S1_vec.size > 0 else 0.0; s2 = S2_vec[0] if S2_vec.size > 0 else 0.0; epsilon_svd = 1e-12
            current_ratio = s1 / (s2 + epsilon_svd); target_s1, target_s2 = s1, s2; modified = False; alpha_sq = alpha * alpha; inv_alpha = 1.0 / (alpha + epsilon_svd)
            if bit_to_embed == 0:
                if current_ratio < alpha: target_s1 = (s1*alpha_sq+alpha*s2)/(alpha_sq+1.0); target_s2 = (alpha*s1+s2)/(alpha_sq+1.0); modified = True
            else:
                if current_ratio >= inv_alpha: target_s1 = (s1+alpha*s2)/(1.0+alpha_sq); target_s2 = (alpha*s1+alpha_sq*s2)/(1.0+alpha_sq); modified = True
            log_level = logging.INFO if modified else logging.DEBUG
            logging.log(log_level, f"[P:{pair_index}, R:{ring_idx}] SVD Mod: {modified}. Bit={bit_to_embed}, Alpha={alpha:.4f}. Orig s1={s1:.4f}, s2={s2:.4f} (R={current_ratio:.4f}). New s1={target_s1:.4f}, s2={target_s2:.4f}.")

            if modified:
                modifications_count += 1
                try:
                    if Vt1.shape!=(1,1) or Vt2.shape!=(1,1): vt1_val=1.0; vt2_val=1.0; logging.warning(f"[P:{pair_index},R:{ring_idx}] Unexpected Vt shape.")
                    else: vt1_val=Vt1[0,0]; vt2_val=Vt2[0,0]
                    modified_dct_col1 = U1 * target_s1 * vt1_val; modified_dct_col2 = U2 * target_s2 * vt2_val
                    modified_dct1 = modified_dct_col1.flatten(); modified_dct2 = modified_dct_col2.flatten()
                    modified_values1 = idct_1d(modified_dct1); modified_values2 = idct_1d(modified_dct2)
                except Exception as recon_err: logging.warning(f"[P:{pair_index}, R:{ring_idx}] DCT/IDCT reconstruct error: {recon_err}"); continue
                if modified_values1.size!=values1.size or modified_values2.size!=values2.size: continue
                delta1 = modified_values1 - values1; delta2 = modified_values2 - values2
                mf1 = np.ones_like(delta1); mf2 = np.ones_like(delta2)
                if use_perceptual_masking and perceptual_mask is not None:
                    try: mv1=perceptual_mask[rows1, cols1]; mv2=perceptual_mask[rows2, cols2]; mf1*=(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mv1); mf2*=(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mv2)
                    except Exception as mask_err: logging.warning(f"[P:{pair_index},R:{ring_idx}] Mask apply error: {mask_err}")
                try: L1[rows1, cols1] += delta1*mf1; L2[rows2, cols2] += delta2*mf2
                except IndexError: continue

        pyramid1.lowpass = L1; pyramid2.lowpass = L2
        comp1_modified = dtcwt_inverse(pyramid1, frame_number); comp2_modified = dtcwt_inverse(pyramid2, frame_number + 1)
        if comp1_modified is None or comp2_modified is None: return None, None
        target_shape = (Y1.shape[0], Y1.shape[1])
        if comp1_modified.shape != target_shape: comp1_modified = cv2.resize(comp1_modified, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        if comp2_modified.shape != target_shape: comp2_modified = cv2.resize(comp2_modified, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        comp1_scaled = np.clip(comp1_modified * 255.0, 0, 255).astype(np.uint8); comp2_scaled = np.clip(comp2_modified * 255.0, 0, 255).astype(np.uint8)
        f1_ycrcb_modified = np.stack((Y1, Cr1, Cb1), axis=-1); f2_ycrcb_modified = np.stack((Y2, Cr2, Cb2), axis=-1)
        f1_ycrcb_modified[:, :, embed_component] = comp1_scaled; f2_ycrcb_modified[:, :, embed_component] = comp2_scaled
        frame1_modified_bgr = cv2.cvtColor(f1_ycrcb_modified, cv2.COLOR_YCrCb2BGR); frame2_modified_bgr = cv2.cvtColor(f2_ycrcb_modified, cv2.COLOR_YCrCb2BGR)
        logging.debug(f"--- Embed Finish P:{pair_index}, Rings:{selected_ring_indices}, Bits:{bits}, Mods:{modifications_count} ---")
        return frame1_modified_bgr, frame2_modified_bgr
    except Exception as e: logging.error(f"Critical error in embed_frame_pair (P:{pair_index}): {e}", exc_info=True); return None, None

# --- Воркер для одной пары кадров (необходим для батч-воркера) ---
# --- Эта функция почти идентична старой _embed_frame_pair_worker ---
def _embed_single_pair_task(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """Обрабатывает одну пару кадров: выбирает кольца и вызывает embed_frame_pair."""
    pair_idx = args['pair_idx']; frame_number = 2 * pair_idx; bits_arg = args['bits']
    frame1 = args['frame1']; frame2 = args['frame2']; n_rings_arg = args['n_rings']
    num_rings_to_use = args['num_rings_to_use']; candidate_pool_size = args['candidate_pool_size']
    embed_component_arg = args['embed_component']; use_perceptual_masking_arg = args['use_perceptual_masking']
    final_selected_rings: List[int] = []

    try:
        if len(bits_arg) != num_rings_to_use: raise ValueError("Bit/Ring mismatch")
        # 1. Выбор колец
        candidate_rings = get_fixed_pseudo_random_rings(pair_idx, n_rings_arg, candidate_pool_size)
        if len(candidate_rings) < num_rings_to_use: raise RuntimeError("Not enough candidate rings.")
        try:
            comp1_select = frame1[:, :, embed_component_arg].astype(np.float32) / 255.0
            pyramid1_select = dtcwt_transform(comp1_select, frame_number)
            if pyramid1_select is None or pyramid1_select.lowpass is None: raise RuntimeError("DTCWT L1 failed for selection")
            L1_select = pyramid1_select.lowpass; all_ring_coords_select = ring_division(L1_select, n_rings_arg, frame_number)
            candidate_entropies: List[Tuple[float, int]] = []
            min_pixels_for_entropy = 10
            for r_idx in candidate_rings:
                entropy = -float('inf')
                if 0 <= r_idx < n_rings_arg:
                    coords = all_ring_coords_select[r_idx]
                    if coords is not None and coords.shape[0] >= min_pixels_for_entropy:
                        try:
                            rows, cols = coords[:, 0], coords[:, 1]; ring_values = L1_select[rows, cols]
                            shannon_entropy, _ = calculate_entropies(ring_values, frame_number, r_idx)
                            if np.isfinite(shannon_entropy): entropy = shannon_entropy
                        except Exception: pass # Ignore errors during entropy calculation for selection
                candidate_entropies.append((entropy, r_idx))
            candidate_entropies.sort(key=lambda x: x[0], reverse=True)
            final_selected_rings = [idx for entr, idx in candidate_entropies if entr > -float('inf')][:num_rings_to_use]

            entropy_log_str_list = []
            candidate_map = {idx: entr for entr, idx in candidate_entropies}
            for r_idx in sorted(candidate_rings): entr = candidate_map.get(r_idx, -float('inf')); entropy_log_str_list.append(f"{r_idx}:{entr:.3f}" if entr > -float('inf') else f"{r_idx}:Err")
            entropy_log_str = ", ".join(entropy_log_str_list)
            logging.info(f"[P:{pair_idx}] Ring Selection: Candidates={candidate_rings}. Entropies=[{entropy_log_str}]. Final Selection={final_selected_rings}")

            if len(final_selected_rings) < num_rings_to_use: raise RuntimeError("Not enough valid rings selected.")
        except Exception as selection_err: logging.error(f"[P:{pair_idx}] Ring selection error: {selection_err}.", exc_info=True); return frame_number, None, None, []

        # 2. Вызов основной функции встраивания
        modified_frame1, modified_frame2 = embed_frame_pair(
            frame1_bgr=frame1, frame2_bgr=frame2, bits=bits_arg,
            selected_ring_indices=final_selected_rings, n_rings=n_rings_arg,
            frame_number=frame_number, use_perceptual_masking=use_perceptual_masking_arg,
            embed_component=embed_component_arg)
        return frame_number, modified_frame1, modified_frame2, final_selected_rings
    except Exception as e: logging.error(f"Error in _embed_single_pair_task (Pair {pair_idx}): {e}", exc_info=True); return frame_number, None, None, []


# --- Воркер для обработки БАТЧА задач (используется ThreadPoolExecutor) ---
def _embed_batch_worker(batch_args_list: List[Dict]) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]]:
    """
    Обрабатывает батч (список) задач последовательно внутри одного ПОТОКА.
    Вызывает _embed_single_pair_task для каждой задачи в батче.
    Возвращает список результатов для батча.
    """
    batch_results = []
    num_in_batch = len(batch_args_list)
    # logging.debug(f"Starting batch of {num_in_batch} tasks in thread {threading.current_thread().name}")
    for i, args in enumerate(batch_args_list):
        try:
            # Вызываем функцию обработки одной пары
            result = _embed_single_pair_task(args)
            batch_results.append(result)
        except Exception as e:
            pair_idx = args.get('pair_idx', -1)
            logging.error(f"Exception processing task {i} (pair_idx={pair_idx}) in batch: {e}", exc_info=True)
            # Возвращаем None или маркер ошибки, чтобы сохранить размерность
            # Используем формат возвращаемого значения _embed_single_pair_task
            batch_results.append((args.get('frame_number', -1), None, None, []))
    # logging.debug(f"Finished batch of {num_in_batch} tasks in thread {threading.current_thread().name}")
    return batch_results


# --- Основная функция встраивания (ThreadPool + Batches) ---
def embed_watermark_in_video(
        frames: List[np.ndarray], packet_bits: np.ndarray, n_rings: int = N_RINGS, num_rings_to_use: int = NUM_RINGS_TO_USE,
        bits_per_pair: int = BITS_PER_PAIR, candidate_pool_size: int = CANDIDATE_POOL_SIZE, max_packet_repeats: int = MAX_PACKET_REPEATS,
        fps: float = FPS, max_workers: Optional[int] = MAX_WORKERS,
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING, embed_component: int = EMBED_COMPONENT):
    """Основная функция, управляющая процессом встраивания с использованием ThreadPoolExecutor и батчинга."""
    num_frames = len(frames); total_pairs = num_frames // 2; packet_len_bits = packet_bits.size
    if total_pairs == 0 or packet_len_bits == 0:
        logging.warning("Not enough frame pairs or empty data packet. Embedding skipped.")
        return frames[:]

    pairs_needed_for_repeats = ceil(max_packet_repeats * packet_len_bits / bits_per_pair)
    pairs_to_process = min(total_pairs, pairs_needed_for_repeats)
    total_bits_to_embed = pairs_to_process * bits_per_pair
    num_actual_repeats = ceil(total_bits_to_embed / packet_len_bits) if packet_len_bits > 0 else 1
    bits_flat = np.tile(packet_bits, num_actual_repeats)[:total_bits_to_embed]
    logging.info(f"Embed Start (ThreadPool+Batches): {total_bits_to_embed} bits ({bits_per_pair}/pair) in {pairs_to_process} pairs.")
    logging.info(f"Packet Info: Length={packet_len_bits} bits. Target Repeats={max_packet_repeats}. Actual Repeats={(total_bits_to_embed / packet_len_bits) if packet_len_bits > 0 else 0:.2f}.")

    start_time = time.time(); watermarked_frames = frames[:]; rings_log: Dict[int, List[int]] = {}
    processed_pairs_count = 0; error_pairs_count = 0; updated_frames_count = 0

    # --- Подготовка списка аргументов для всех пар ---
    all_pairs_args = []
    skipped_pairs = 0
    for pair_idx in range(pairs_to_process):
        frame_idx1 = 2 * pair_idx; frame_idx2 = frame_idx1 + 1
        if frame_idx2 >= num_frames or frames[frame_idx1] is None or frames[frame_idx2] is None:
            skipped_pairs += 1; continue
        start_bit_idx = pair_idx * bits_per_pair; end_bit_idx = start_bit_idx + bits_per_pair
        current_bits = bits_flat[start_bit_idx:end_bit_idx].tolist()
        if len(current_bits) != bits_per_pair: skipped_pairs += 1; continue
        args = {
            'pair_idx': pair_idx, 'frame1': frames[frame_idx1], 'frame2': frames[frame_idx2], 'bits': current_bits,
            'n_rings': n_rings, 'num_rings_to_use': num_rings_to_use, 'candidate_pool_size': candidate_pool_size,
            'frame_number': frame_idx1, 'use_perceptual_masking': use_perceptual_masking,
            'embed_component': embed_component
        }
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0: logging.warning(f"Skipped {skipped_pairs} pairs due to invalid frames or bit slicing.")
    if num_valid_tasks == 0: logging.error("No valid tasks to process."); return watermarked_frames

    # --- Разделение на батчи ---
    # Определяем количество потоков
    num_workers = max_workers if max_workers is not None and max_workers > 0 else (os.cpu_count() or 1) # Используем CPU count как fallback
    # Определяем размер батча (можно настроить)
    # batch_size = max(1, ceil(num_valid_tasks / num_workers)) # Распределяем задачи по потокам
    batch_size = max(1, min(num_valid_tasks, 10)) # Или фиксированный размер
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = []
    for i in range(0, num_valid_tasks, batch_size):
        batch = all_pairs_args[i : i + batch_size]
        if batch: batched_args_list.append(batch)

    logging.info(f"Launching {num_batches} batches ({num_valid_tasks} total pairs) using ThreadPoolExecutor (max_workers={num_workers}, batch_size≈{batch_size})...")

    # --- Использование ThreadPoolExecutor для БАТЧЕЙ ---
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch_index = {executor.submit(_embed_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}

            for future in concurrent.futures.as_completed(future_to_batch_index):
                batch_index = future_to_batch_index[future]
                original_batch_args = batched_args_list[batch_index]
                try:
                    batch_results = future.result() # Список результатов от _embed_single_pair_task
                    if len(batch_results) != len(original_batch_args):
                         logging.error(f"Batch {batch_index} result size mismatch!"); error_pairs_count += len(original_batch_args); continue

                    for i, single_result in enumerate(batch_results):
                        if single_result is not None and isinstance(single_result, tuple) and len(single_result) == 4:
                            f_num_res, mod_f1, mod_f2, selected_rings = single_result
                            pair_idx = original_batch_args[i]['pair_idx'] # Получаем индекс исходной пары
                            frame_idx1 = 2 * pair_idx; frame_idx2 = frame_idx1 + 1
                            if selected_rings: rings_log[pair_idx] = selected_rings
                            if mod_f1 is not None and mod_f2 is not None:
                                if frame_idx1 < len(watermarked_frames): watermarked_frames[frame_idx1] = mod_f1; updated_frames_count += 1
                                if frame_idx2 < len(watermarked_frames): watermarked_frames[frame_idx2] = mod_f2; updated_frames_count += 1
                                processed_pairs_count += 1
                            else: error_pairs_count += 1 # _embed_single_pair_task вернул None для кадров
                        else: error_pairs_count += 1 # Результат некорректен или None (ошибка в батч-воркере)
                except Exception as exc:
                     logging.error(f'Batch {batch_index} raised exception: {exc}', exc_info=True)
                     error_pairs_count += len(original_batch_args)

    except Exception as e:
        logging.critical(f"Critical error during ThreadPoolExecutor (Batched) execution: {e}", exc_info=True)
        return frames[:]

    logging.info(f"Batch processing finished. Successful pairs: {processed_pairs_count}, Failed/Skipped pairs: {error_pairs_count + skipped_pairs}.")
    logging.info(f"Results applied to {updated_frames_count} frames.")

    # Сохраняем лог колец
    if rings_log:
        try:
            serializable_rings_log = {str(k): v for k, v in rings_log.items()}
            with open(SELECTED_RINGS_FILE, 'w') as f: json.dump(serializable_rings_log, f, indent=4)
            logging.info(f"Ring log saved to: {SELECTED_RINGS_FILE}")
        except Exception as e: logging.error(f"Could not save ring log: {e}", exc_info=True)
    else: logging.warning("Ring log is empty.")

    end_time = time.time(); logging.info(f"Embedding process finished. Time elapsed: {end_time - start_time:.2f} seconds.")
    return watermarked_frames


# --- Основная Функция (main) ---
def main():
    """Главная функция, управляющая всем процессом встраивания."""
    global BCH_CODE_OBJECT
    start_time_main = time.time()
    input_video = "input.mp4"
    base_output_filename = "watermarked_galois_t4_threaded_batched" # Новое имя
    output_video = base_output_filename + OUTPUT_EXTENSION
    logging.info("--- Starting Embedding Main Process (ThreadPool + Batches) ---")

    # 1. Чтение видео
    frames, fps_read = read_video(input_video)
    if not frames: logging.critical("Video read failed."); return
    fps_to_use = float(FPS) if fps_read <= 0 else fps_read
    if len(frames) < 2: logging.critical("Not enough frames."); return

    # 2. Генерация ID
    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES); original_id_hex = original_id_bytes.hex()
    logging.info(f"Generated Payload ID ({PAYLOAD_LEN_BYTES * 8} bit, Hex): {original_id_hex}")

    # 3. Подготовка пакета
    packet_bits_to_embed: Optional[np.ndarray] = None
    effective_ecc_enabled = USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is not None
    if effective_ecc_enabled:
        try:
            bch_n = BCH_CODE_OBJECT.n; bch_k = BCH_CODE_OBJECT.k; bch_t = BCH_CODE_OBJECT.t
            logging.info(f"Using Galois BCH: n={bch_n}, k={bch_k}, t={bch_t}")
            payload_len_bits = PAYLOAD_LEN_BYTES * 8
            if payload_len_bits > bch_k: logging.error(f"Payload size > k! ECC disabled."); effective_ecc_enabled = False
            else:
                payload_bits = np.unpackbits(np.frombuffer(original_id_bytes, dtype=np.uint8))
                packet_bits_to_embed = add_ecc(payload_bits, BCH_CODE_OBJECT)
                if packet_bits_to_embed is None: raise RuntimeError("add_ecc failed.")
        except Exception as e: logging.error(f"ECC packet prep failed: {e}. ECC disabled."); effective_ecc_enabled = False
    else: logging.info("ECC disabled or unavailable.")
    if packet_bits_to_embed is None:
        packet_bits_to_embed = np.unpackbits(np.frombuffer(original_id_bytes, dtype=np.uint8))
        logging.info(f"Using raw payload ({packet_bits_to_embed.size} bits).")
    else: logging.info(f"Using ECC packet ({packet_bits_to_embed.size} bits).")

    # 4. Сохранение ID
    try:
        with open(ORIGINAL_WATERMARK_FILE, "w") as f: f.write(original_id_hex)
        logging.info(f"Original ID saved: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e: logging.error(f"Save ID failed: {e}")

    # 5. Вызов функции встраивания (ThreadPool + Batches)
    watermarked_frames = embed_watermark_in_video(
        frames=frames, packet_bits=packet_bits_to_embed, n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE,
        bits_per_pair=BITS_PER_PAIR, candidate_pool_size=CANDIDATE_POOL_SIZE, max_packet_repeats=MAX_PACKET_REPEATS,
        fps=fps_to_use, max_workers=MAX_WORKERS, # Передаем max_workers для потоков
        use_perceptual_masking=USE_PERCEPTUAL_MASKING, embed_component=EMBED_COMPONENT)

    # 6. Запись результата
    if watermarked_frames and len(watermarked_frames) == len(frames):
        write_video(frames=watermarked_frames, out_path=output_video, fps=fps_to_use, codec=OUTPUT_CODEC)
        logging.info(f"Watermarked video saved: {output_video}")
        try:
            if os.path.exists(output_video): logging.info(f"Output size: {os.path.getsize(output_video)/(1024*1024):.2f} MB")
            else: logging.error(f"Output file missing after write!")
        except OSError as e: logging.error(f"Get output size failed: {e}")
    else: logging.error("Embedding failed or returned invalid frames. No output.")

    logging.info("--- Embedding Main Process Finished (ThreadPool + Batches) ---")
    total_time_main = time.time() - start_time_main
    logging.info(f"--- Total Main Function Time: {total_time_main:.2f} seconds ---")
    print(f"\nEmbedding process (ThreadPool + Batches) finished.")
    print(f"Output: {output_video}")
    print(f"Log: {LOG_FILENAME}")
    print(f"ID file: {ORIGINAL_WATERMARK_FILE}")
    print(f"Rings file: {SELECTED_RINGS_FILE}")
    print("\nRun extractor.py to extract and verify.")


# --- Точка Входа ---
if __name__ == "__main__":
    if USE_ECC and not GALOIS_AVAILABLE:
        print("\nERROR: ECC required but galois failed/unavailable. Cannot proceed safely.")
    else:
        prof = cProfile.Profile()
        prof.enable()
        try:
            main()
        except FileNotFoundError as e: print(f"\nERROR: Input file not found: {e}"); logging.error(f"Input file not found: {e}", exc_info=True)
        except ValueError as e: print(f"\nERROR: Value error: {e}."); logging.error(f"Value error: {e}", exc_info=True)
        except RuntimeError as e: print(f"\nERROR: Runtime error: {e}."); logging.error(f"Runtime error: {e}", exc_info=True)
        except Exception as e: logging.critical(f"Unhandled exception in main (ThreadPool + Batches): {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. Check log: {LOG_FILENAME}")
        finally:
            prof.disable(); stats = pstats.Stats(prof)
            print("\n--- Profiling Stats (Top 30 Cumulative Time) ---")
            stats.strip_dirs().sort_stats("cumulative").print_stats(30)
            print("-------------------------------------------------")
            pfile = f"profile_embed_threaded_batched_galois_t{BCH_T}.txt" # Новое имя файла профиля
            try:
                with open(pfile, "w") as f: sf = pstats.Stats(prof, stream=f); sf.strip_dirs().sort_stats("cumulative").print_stats()
                logging.info(f"Profiling stats saved: {pfile}"); print(f"Profiling stats saved: {pfile}")
            except IOError as e: logging.error(f"Save profile failed: {e}"); print(f"Warning: Could not save profile stats.")