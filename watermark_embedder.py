# Файл: embedder.py (Версия: Вариант 5 + NumPy DTCWT + CuPy GPU + MP4/H264 + Fixes v2)

import cv2
import numpy as np
from scipy.fftpack import dct as dct_cpu, idct as idct_cpu # Для DCT/IDCT моста
import random
import logging
import time
import concurrent.futures
import json
import os
import imagehash
import hashlib
from PIL import Image
import dtcwt # Используем NumPy бэкенд по умолчанию
from typing import List, Tuple, Optional, Dict, Any
import functools
import uuid
from math import ceil, pi # Добавлен pi
import imageio # Для записи видео

# --- Настройка Логирования (в самом начале) ---
LOG_FILENAME: str = 'watermarking_embed_numpy_dtcwt_cupy_gpu.log' # Новое имя лога
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME, filemode='w', level=logging.DEBUG, # Уровень DEBUG
    format='[%(asctime)s] %(levelname).1s %(threadName)s %(module)s:%(lineno)d - %(message)s'
)
logging.info("--- Инициализация Скрипта Embedder ---")


# --- GPU / CuPy ---
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu
    from cupy.linalg import svd as svd_gpu

    try:
        num_devices = cp.cuda.runtime.getDeviceCount()
        if num_devices == 0: raise RuntimeError("No CUDA devices found")
        cp.cuda.Device(0).use(); CUPY_AVAILABLE = True
        logging.info("CuPy доступен и GPU инициализирован.")
        props=cp.cuda.runtime.getDeviceProperties(0); gpu_name=props['name'].decode(); total_mem_gb=props['totalGlobalMem']/(1024**3)
        logging.info(f"Используется GPU 0: {gpu_name}, Память: {total_mem_gb:.2f} GB")
    except Exception as e: logging.warning(f"CuPy/GPU init error: {e}. GPU НЕ будет использоваться."); CUPY_AVAILABLE = False; cp = np
except ImportError:
    logging.warning("CuPy не найден. GPU ускорение НЕДОСТУПНО."); CUPY_AVAILABLE = False; cp = np

USE_GPU_ACCELERATION = CUPY_AVAILABLE
logging.info(f"GPU Acceleration Enabled: {USE_GPU_ACCELERATION}")
xp = cp if USE_GPU_ACCELERATION else np

# --- DTCWT Бэкенд (Принудительно NumPy) ---
DTCWT_BACKEND_SET = 'unavailable'
dtcwt_transformer = None
try:
    import dtcwt
    try:
        dtcwt_transformer = dtcwt.Transform2d() # Использует NumPy по умолчанию
        logging.info("Создан экземпляр dtcwt.Transform2d (с бэкендом NumPy).")
        DTCWT_BACKEND_SET = 'numpy'
    except Exception as e_trans_init:
        logging.error(f"Не удалось создать dtcwt.Transform2d: {e_trans_init}")
except ImportError:
     logging.error("Библиотека dtcwt не установлена!")

logging.info(f"Итоговый DTCWT Backend для использования: {DTCWT_BACKEND_SET}")
if DTCWT_BACKEND_SET == 'unavailable': print("\nFATAL: dtcwt init failed."); exit()

# --- GPU OpenCV (Проверка) ---
CV2_CUDA_AVAILABLE = False # Явно отключаем использование cv2.cuda для маски, т.к. оно не работало
# try:
#     gpus_cv = cv2.cuda.getCudaEnabledDeviceCount();
#     if gpus_cv > 0: cv2.cuda.setDevice(0); CV2_CUDA_AVAILABLE = True; logging.info(f"OpenCV CUDA доступен ({gpus_cv} GPU).")
#     else: logging.warning("OpenCV CUDA: GPU не обнаружены.")
# except AttributeError: logging.warning("OpenCV CUDA модуль недоступен.")
# except Exception as e: logging.warning(f"OpenCV CUDA init error: {e}.")
logging.info(f"OpenCV CUDA Enabled: {CV2_CUDA_AVAILABLE} (Принудительно False)")


# --- BCH ---
try:
    import bchlib; BCHLIB_AVAILABLE = True
except ImportError: BCHLIB_AVAILABLE = False

import cProfile
import pstats

# --- Параметры Алгоритма ---
LAMBDA_PARAM: float = 0.04; ALPHA_MIN: float = 1.005; ALPHA_MAX: float = 1.1
N_RINGS: int = 8; MAX_THEORETICAL_ENTROPY = 8.0; BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR; CANDIDATE_POOL_SIZE: int = 4
EMBED_COMPONENT: int = 1; USE_PERCEPTUAL_MASKING: bool = True

# --- Параметры Встраивания и ECC ---
PAYLOAD_LEN_BYTES: int = 8; USE_ECC: bool = True; BCH_M: int = 8; BCH_T: int = 5
MAX_PACKET_REPEATS: int = 5

# --- Параметры Ввода/Вывода ---
FPS: int = 30; MAX_WORKERS: Optional[int] = 4
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
OUTPUT_EXTENSION: str = '.mp4'
OUTPUT_CODEC: str = 'libx264' # Кодек для imageio

# --- Инициализация BCH ---
bch = None; payload_bits_len = PAYLOAD_LEN_BYTES * 8; packet_len_bits = payload_bits_len; ecc_bits_len = 0
effective_use_ecc = USE_ECC and BCHLIB_AVAILABLE
if effective_use_ecc:
    try:
        bch = bchlib.BCH(m=BCH_M, t=BCH_T)
        if hasattr(bch, 'ecc_bits') and bch.ecc_bits > 0:
            ecc_bits_len = bch.ecc_bits; packet_len_bits = payload_bits_len + ecc_bits_len
            logging.info(f"BCH: m={BCH_M}, t={BCH_T}. Payload={payload_bits_len}, ECC={ecc_bits_len}. Packet={packet_len_bits} bits.")
        else: effective_use_ecc = False; bch = None; packet_len_bits = payload_bits_len; logging.warning("BCH ecc_bits error.")
    except Exception as e: effective_use_ecc = False; bch = None; packet_len_bits = payload_bits_len; logging.error(f"BCH init error: {e}.")
if not effective_use_ecc: logging.warning("ECC не используется.")

# --- Проверки размеров колец ---
if NUM_RINGS_TO_USE != BITS_PER_PAIR: NUM_RINGS_TO_USE = BITS_PER_PAIR
if CANDIDATE_POOL_SIZE < NUM_RINGS_TO_USE: CANDIDATE_POOL_SIZE = NUM_RINGS_TO_USE
if CANDIDATE_POOL_SIZE > N_RINGS: CANDIDATE_POOL_SIZE = N_RINGS

# --- Основные Функции ---

def dct_1d_cupyfft(signal_1d_gpu: cp.ndarray, norm: Optional[str] = 'ortho') -> cp.ndarray:
    """1D DCT-II с использованием CuPy FFT."""
    if not USE_GPU_ACCELERATION: raise NotImplementedError("CPU DCT fallback needed")
    N = signal_1d_gpu.shape[0];
    if N == 0: return cp.array([], dtype=signal_1d_gpu.dtype)
    y = cp.empty(2 * N, dtype=signal_1d_gpu.dtype); y[:N] = signal_1d_gpu; y[N:] = signal_1d_gpu[::-1]
    Y = cp.fft.fft(y)[:N]; k = cp.arange(N, dtype=signal_1d_gpu.dtype)
    Y *= 2 * cp.exp(-1j * cp.pi * k / (2 * N))
    if norm == 'ortho': Y[0] *= cp.sqrt(1/(4*N)); Y[1:] *= cp.sqrt(1/(2*N))
    return Y.real

def idct_1d_cupyfft(coeffs_1d_gpu: cp.ndarray, norm: Optional[str] = 'ortho') -> cp.ndarray:
    """1D IDCT-II с использованием CuPy FFT."""
    if not USE_GPU_ACCELERATION: raise NotImplementedError("CPU IDCT fallback needed")
    N = coeffs_1d_gpu.shape[0];
    if N == 0: return cp.array([], dtype=coeffs_1d_gpu.dtype)
    coeffs_complex = coeffs_1d_gpu.astype(cp.complex64, copy=True)
    k = cp.arange(N, dtype=cp.float32)
    if norm == 'ortho': coeffs_complex[0] /= cp.sqrt(1/(4*N)); coeffs_complex[1:] /= cp.sqrt(1/(2*N))
    coeffs_complex *= cp.exp(1j * cp.pi * k / (2 * N)) / 2
    y = cp.empty(2 * N, dtype=cp.complex64)
    y[0] = coeffs_complex[0]; y[N] = 0.0 # Correct handling for N point? Check formula source
    if N > 1: y[1:N] = coeffs_complex[1:]; y[N+1:] = -cp.conj(coeffs_complex[1:][::-1]) # Complex conjugate for symmetry
    result = cp.fft.ifft(y); return result[:N].real

def dct_1d(signal_1d_gpu: cp.ndarray) -> cp.ndarray:
    if not USE_GPU_ACCELERATION: return np.array([]) # Fallback not implemented
    return dct_1d_cupyfft(signal_1d_gpu, norm='ortho')

def idct_1d(coeffs_1d_gpu: cp.ndarray) -> cp.ndarray:
    if not USE_GPU_ACCELERATION: return np.array([])
    return idct_1d_cupyfft(coeffs_1d_gpu, norm='ortho')

def dtcwt_transform_backend(y_plane_cpu: np.ndarray, frame_number: int = -1) -> Optional[Tuple[Any, np.ndarray]]:
    func_start_time = time.time()
    if DTCWT_BACKEND_SET == 'unavailable' or dtcwt_transformer is None: return None
    if np.any(np.isnan(y_plane_cpu)): y_plane_cpu = np.nan_to_num(y_plane_cpu)
    if y_plane_cpu.ndim != 2 or y_plane_cpu.size == 0: return None
    try:
        pyramid_result = dtcwt_transformer.forward(y_plane_cpu.astype(np.float32), nlevels=1)
        if hasattr(pyramid_result, 'lowpass') and pyramid_result.lowpass is not None:
            ll_raw = pyramid_result.lowpass; ll_module = cp.get_array_module(ll_raw)
            ll_subband_cpu = ll_module.asnumpy(ll_raw).copy() if ll_module == cp else np.copy(ll_raw)
            return pyramid_result, ll_subband_cpu
        else: logging.error(f"DTCWT no lowpass."); return None
    except Exception as e: logging.error(f"DTCWT forward error: {e}"); return None

def dtcwt_inverse_backend(pyramid: Any, ll_modified_cpu: np.ndarray, frame_number: int = -1) -> Optional[np.ndarray]:
    func_start_time = time.time()
    if DTCWT_BACKEND_SET == 'unavailable' or dtcwt_transformer is None: return None
    if not hasattr(pyramid, 'lowpass'): return None
    try:
        pyramid.lowpass = ll_modified_cpu.astype(np.float32) # NumPy бэкенд ожидает NumPy
        reconstructed_y_raw = dtcwt_transformer.inverse(pyramid)
        res_module = cp.get_array_module(reconstructed_y_raw)
        reconstructed_y_cpu = res_module.asnumpy(reconstructed_y_raw) if res_module == cp else np.copy(reconstructed_y_raw)
        if np.any(np.isnan(reconstructed_y_cpu)): reconstructed_y_cpu = np.nan_to_num(reconstructed_y_cpu)
        return reconstructed_y_cpu.astype(np.float32)
    except Exception as e: logging.error(f"DTCWT inverse error: {e}"); return None

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
    return rings_coords_np
@functools.lru_cache(maxsize=8)
def get_ring_coords_cached_cpu(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    return _ring_division_internal_cpu(subband_shape, n_rings)
def ring_division_coords_cpu(subband_shape: Tuple[int, int], n_rings: int = N_RINGS, frame_number: int = -1) -> List[Optional[np.ndarray]]:
    try:
        coords_list_np = get_ring_coords_cached_cpu(subband_shape, n_rings)
        if not isinstance(coords_list_np, list): raise TypeError("Invalid cache type")
        return [arr.copy() if arr is not None else None for arr in coords_list_np]
    except Exception as e: logging.error(f"Ring division CPU error: {e}"); return [None] * n_rings

def get_ring_values(ll_subband_gpu: cp.ndarray, ring_coords_cpu: np.ndarray) -> cp.ndarray:
    if ring_coords_cpu is None or ring_coords_cpu.size == 0: return cp.array([], dtype=cp.float32)
    try: rows_cpu, cols_cpu = ring_coords_cpu[:, 0], ring_coords_cpu[:, 1]; return ll_subband_gpu[rows_cpu, cols_cpu].astype(cp.float32)
    except Exception as e: logging.error(f"Get ring values GPU error: {e}"); return cp.array([], dtype=cp.float32)

def calculate_entropies_gpu(ring_vals_gpu: cp.ndarray, frame_number: int = -1, ring_index: int = -1) -> Tuple[float, float]:
    eps=1e-12; visual=0.0; edge=0.0
    if ring_vals_gpu.size==0: return visual, edge
    try:
        min_v, max_v = cp.min(ring_vals_gpu), cp.max(ring_vals_gpu)
        if float(min_v.get()) < 0.0 or float(max_v.get()) > 1.0: vals_clip = cp.clip(ring_vals_gpu, 0.0, 1.0)
        else: vals_clip = ring_vals_gpu
        hist, _ = cp.histogram(vals_clip, bins=256, range=(0.0, 1.0)); total = vals_clip.size
        if total == 0: return 0.0, 0.0
        prob = hist / total; prob = prob[prob > eps]
        if prob.size == 0: return 0.0, 0.0
        vg = -cp.sum(prob*cp.log2(prob)); eg = -cp.sum(prob*cp.exp(1.0-prob))
        visual=float(cp.clip(vg,0.0,MAX_THEORETICAL_ENTROPY).get()); edge=float(eg.get())
    except Exception as e: logging.error(f"GPU entropy error: {e}")
    return visual, edge

def compute_adaptive_alpha_entropy_gpu(ring_vals_gpu: cp.ndarray, ring_index: int, frame_number: int) -> float:
    min_pix=10; final_a=ALPHA_MIN
    if ring_vals_gpu.size<min_pix: return final_a
    try:
        ve, _ = calculate_entropies_gpu(ring_vals_gpu, frame_number, ring_index); lv = cp.var(ring_vals_gpu)
        en=np.clip(ve/MAX_THEORETICAL_ENTROPY,0.0,1.0); vm=0.005; vs=500; lvf=float(lv.get())
        tn=1.0/(1.0+np.exp(-vs*(lvf-vm))); we=0.6; wt=0.4; mf=np.clip((we*en+wt*tn),0.0,1.0)
        final_a=ALPHA_MIN+(ALPHA_MAX-ALPHA_MIN)*mf; final_a=np.clip(final_a,ALPHA_MIN,ALPHA_MAX)
    except Exception as e: logging.error(f"GPU alpha error: {e}"); final_a=ALPHA_MIN
    return final_a

def get_fixed_pseudo_random_rings(pair_idx: int, n_rings: int, num_to_select: int) -> List[int]:
    if num_to_select > n_rings: num_to_select = n_rings
    if num_to_select <= 0: return []
    pidxb=str(pair_idx).encode('utf-8'); ho=hashlib.sha256(pidxb); hd=ho.digest()
    seed=int.from_bytes(hd,'big'); prng=random.Random(seed); aidx=list(range(n_rings))
    sel=prng.sample(aidx, num_to_select); return sorted(sel)

def _sobel_gpu(img_gpu: cp.ndarray, dx: int, dy: int, ksize: int = 3) -> cp.ndarray:
    """Простая реализация Собеля через свертку CuPy."""
    dtype = img_gpu.dtype
    if ksize == 3:
        if dx == 1 and dy == 0: kernel = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype)
        elif dx == 0 and dy == 1: kernel = cp.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=dtype)
        else: raise ValueError("Sobel: dx/dy invalid")
    else: raise ValueError("Sobel: ksize != 3 not implemented")
    from cupyx.scipy.ndimage import convolve as convolve_gpu # Импорт внутри функции
    return convolve_gpu(img_gpu, kernel, mode='reflect')

def calculate_perceptual_mask_cupy(input_plane_gpu: cp.ndarray, frame_number: int = -1) -> Optional[cp.ndarray]:
    """Вычисляет маску на GPU (CuPy)."""
    if not USE_GPU_ACCELERATION: return cp.ones_like(input_plane_gpu)
    xp = cp; logging.debug(f"[F:{frame_number}] [GPU-CuPy] Mask Calc...")
    if input_plane_gpu.ndim != 2: return None
    try:
        plane_32f = input_plane_gpu.astype(xp.float32)
        gx = _sobel_gpu(plane_32f, 1, 0, ksize=3); gy = _sobel_gpu(plane_32f, 0, 1, ksize=3)
        grad_mag = xp.sqrt(gx**2 + gy**2)
        ksize_gauss = 11; sigma_gauss = 5
        local_mean = gaussian_filter_gpu(plane_32f, sigma=sigma_gauss, order=0, mode='reflect', truncate=4.0)
        mean_sq = gaussian_filter_gpu(plane_32f**2, sigma=sigma_gauss, order=0, mode='reflect', truncate=4.0)
        sq_mean = local_mean**2
        local_variance = xp.maximum(mean_sq - sq_mean, 0)
        local_stddev = xp.sqrt(local_variance)
        mask = xp.maximum(grad_mag, local_stddev)
        eps = 1e-9; max_mask = xp.max(mask)
        max_mask_float = float(max_mask.get()) # Перенос на CPU для if
        mask_norm = mask / (max_mask + eps) if max_mask_float > eps else xp.zeros_like(mask)
        mask_norm = xp.clip(mask_norm, 0.0, 1.0)
        return mask_norm.astype(xp.float32)
    except ImportError: logging.error("CuPyX/SciPy/ndimage required for GPU mask. Fallback CPU.")
    except Exception as e: logging.error(f"CuPy mask error: {e}. Fallback CPU.")
    # Fallback на CPU
    input_plane_cpu = cp.asnumpy(input_plane_gpu)
    mask_cpu = calculate_perceptual_mask_cpu(input_plane_cpu, frame_number) # Используем CPU-версию
    return cp.asarray(mask_cpu) if mask_cpu is not None else cp.ones_like(input_plane_gpu)

def calculate_perceptual_mask_gpu(input_plane_gpu: cp.ndarray, frame_number: int = -1) -> Optional[cp.ndarray]:
    """Обертка для вызова расчета маски на GPU (CuPy)."""
    if not USE_GPU_ACCELERATION:
        input_plane_cpu = cp.asnumpy(input_plane_gpu)
        mask_cpu = calculate_perceptual_mask_cpu(input_plane_cpu, frame_number)
        return cp.asarray(mask_cpu) if mask_cpu is not None else cp.ones_like(input_plane_gpu)
    else:
        return calculate_perceptual_mask_cupy(input_plane_gpu, frame_number)

def calculate_perceptual_mask_cpu(input_plane_cpu: np.ndarray, frame_number: int = -1) -> Optional[np.ndarray]:
    """Вычисляет маску перцептуальной заметности на CPU (Fallback)."""
    logging.debug(f"[F:{frame_number}] [CPU] Fallback Mask Calc...") # Лог
    if not isinstance(input_plane_cpu, np.ndarray) or input_plane_cpu.ndim != 2: return None
    try:
        plane_32f = input_plane_cpu.astype(np.float32)
        gx = cv2.Sobel(plane_32f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(plane_32f, cv2.CV_32F, 0, 1, ksize=3)
        gm=np.sqrt(gx**2+gy**2); ks=(11,11); sg=5;
        lm=cv2.GaussianBlur(plane_32f, ks, sg)
        ms=cv2.GaussianBlur(plane_32f**2, ks, sg)
        sm=lm**2; lv=np.maximum(ms-sm,0); ls=np.sqrt(lv); mask=np.maximum(gm,ls); eps=1e-9; mm=np.max(mask)
        mn=mask/(mm+eps) if mm>eps else np.zeros_like(mask); mn=np.clip(mn,0.0,1.0); return mn.astype(np.float32)
    except Exception as e: logging.error(f"CPU mask error: {e}"); return np.ones_like(input_plane_cpu, dtype=np.float32)

def add_ecc(data_bits: np.ndarray, bch_obj: 'bchlib.BCH') -> Optional[np.ndarray]:
    # (Код без изменений)
    if not BCHLIB_AVAILABLE or bch_obj is None: return data_bits
    orig_len = data_bits.size;
    if orig_len == 0: return None
    try:
        ecc_len = bch_obj.ecc_bits;
        if ecc_len <= 0: raise ValueError("BCH ecc_bits <= 0")
        dbytes = np.packbits(data_bits).tobytes(); ecc_bytes = bch_obj.encode(dbytes)
        ecc_bits = np.unpackbits(np.frombuffer(ecc_bytes,dtype=np.uint8))
        if len(ecc_bits) < ecc_len: ecc_bits = np.pad(ecc_bits,(0,ecc_len-len(ecc_bits)))
        elif len(ecc_bits) > ecc_len: ecc_bits = ecc_bits[:ecc_len]
        pkt = np.concatenate((data_bits, ecc_bits)).astype(np.uint8)
        exp_len = orig_len + ecc_len
        if pkt.size != exp_len:
             logging.error(f"add_ecc concat error! {pkt.size}!={exp_len}.")
             if pkt.size > exp_len: pkt = pkt[:exp_len]
             else: return None
        return pkt
    except Exception as e: logging.error(f"ECC encode error: {e}"); return data_bits

def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    # (Код без изменений)
    func_start_time = time.time(); logging.info(f"Reading video from: {video_path}")
    frames: List[np.ndarray] = []; fps = float(FPS); cap = None; expected_height, expected_width = -1, -1
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed open video: {video_path}"); return frames, fps
        fps_read = cap.get(cv2.CAP_PROP_FPS); fps = float(fps_read) if fps_read > 0 else fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Video props: {width}x{height} @ {fps:.2f} FPS, ~{frame_count_prop} frames")
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
    return frames, fps

def write_video(frames: List[np.ndarray], out_path: str, fps: float, codec: str = 'libx264'):
    # (Код без изменений)
    func_start_time = time.time();
    if not frames: logging.error("No frames to write."); return
    base, _ = os.path.splitext(out_path); out_path_corrected = base + ".mp4"
    if out_path_corrected != out_path: logging.info(f"Correcting ext to '.mp4'. Path: {out_path_corrected}"); out_path = out_path_corrected
    logging.info(f"Writing with imageio: {out_path} (FPS: {fps:.2f}, Codec: {codec})")
    writer = None
    try:
        writer = imageio.get_writer(out_path, fps=fps, codec=codec, quality=7, pixelformat='yuv420p', macro_block_size=16)
        written_count = 0; skipped_count = 0; h, w = -1, -1; black_frame = None
        for i, frame in enumerate(frames):
            if frame is None or frame.ndim != 3 or frame.dtype != np.uint8: skipped_count += 1; continue
            if h == -1: h, w, _ = frame.shape; logging.info(f"Output res: {w}x{h}"); black_frame = np.zeros((h, w, 3), dtype=np.uint8)
            if frame.shape[0] != h or frame.shape[1] != w:
                 if black_frame is not None: writer.append_data(black_frame); skipped_count += 1; continue
                 else: skipped_count += 1; continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); writer.append_data(frame_rgb); written_count += 1
        logging.info(f"Imageio write done. Written: {written_count}, Skipped: {skipped_count}")
    except ImportError: logging.error("imageio library not found. Install: pip install imageio[ffmpeg]")
    except Exception as e: logging.error(f"Imageio write exception: {e}", exc_info=True)
    finally:
        if writer is not None:
             try: writer.close()
             except Exception as e_close: logging.error(f"Error closing imageio writer: {e_close}")

# --- Функция Встраивания (NumPy DTCWT + CuPy остальное) ---
def embed_frame_pair_numpy_dtcwt_cupy_gpu( # Новое имя для ясности
    frame1_bgr_cpu: np.ndarray, frame2_bgr_cpu: np.ndarray, bits: List[int],
    selected_ring_indices: List[int],
    n_rings: int = N_RINGS, frame_number: int = 0,
    use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
    embed_component: int = EMBED_COMPONENT, gpu_id: int = 0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Встраивает N бит: NumPy DTCWT + CuPy (Маска, DCT/SVD/IDCT, Entropy, Alpha)."""
    if not USE_GPU_ACCELERATION: return None, None
    xp = cp
    try: cp.cuda.Device(gpu_id).use()
    except Exception: return None, None

    # Инициализация переменных ДО try
    func_start_time = time.time(); pair_num_log = frame_number // 2
    result_frame1_cpu, result_frame2_cpu = None, None
    comp1_gpu, comp2_gpu, L1_gpu, L2_gpu, perceptual_mask_gpu = None, None, None, None, None
    dtcwt_pyramid1, dtcwt_pyramid2 = None, None
    mod_count = 0
    logging.debug(f"[GPU-{gpu_id} P:{pair_num_log}] Embed NP+GPU Start...")

    try:
        # --- CPU Подготовка ---
        frame1_ycrcb_cpu = cv2.cvtColor(frame1_bgr_cpu, cv2.COLOR_BGR2YCrCb)
        frame2_ycrcb_cpu = cv2.cvtColor(frame2_bgr_cpu, cv2.COLOR_BGR2YCrCb)
        comp1_cpu=(frame1_ycrcb_cpu[:,:,embed_component].astype(np.float32)/255.0)
        comp2_cpu=(frame2_ycrcb_cpu[:,:,embed_component].astype(np.float32)/255.0)
        Y1_cpu=frame1_ycrcb_cpu[:,:,0];Cr1_cpu=frame1_ycrcb_cpu[:,:,1];Cb1_cpu=frame1_ycrcb_cpu[:,:,2]
        Y2_cpu=frame2_ycrcb_cpu[:,:,0];Cr2_cpu=frame2_ycrcb_cpu[:,:,1];Cb2_cpu=frame2_ycrcb_cpu[:,:,2]
        target_shape_cpu = comp1_cpu.shape

        # --- GPU ---
        comp1_gpu = xp.asarray(comp1_cpu) # Для маски

        # DTCWT (NumPy CPU)
        dtcwt_res1 = dtcwt_transform_backend(comp1_cpu, frame_number)
        dtcwt_res2 = dtcwt_transform_backend(comp2_cpu, frame_number + 1)
        if dtcwt_res1 is None or dtcwt_res2 is None: raise RuntimeError("DTCWT(NumPy) transform failed")
        dtcwt_pyramid1, ll1_cpu = dtcwt_res1; dtcwt_pyramid2, ll2_cpu = dtcwt_res2
        L1_gpu = xp.asarray(ll1_cpu); L2_gpu = xp.asarray(ll2_cpu); ll_shape = L1_gpu.shape # LL -> GPU

        # Кольца (CPU)
        rings_coords_cpu_list = ring_division_coords_cpu(ll_shape, n_rings, frame_number)

        # Маска (GPU)
        if use_perceptual_masking: perceptual_mask_gpu = calculate_perceptual_mask_gpu(comp1_gpu, frame_number)

        # Цикл встраивания (GPU)
        for ring_idx, bit in zip(selected_ring_indices, bits):
            coords_cpu = rings_coords_cpu_list[ring_idx];
            if coords_cpu is None: continue
            ring_vals_1 = get_ring_values(L1_gpu, coords_cpu)
            ring_vals_2 = get_ring_values(L2_gpu, coords_cpu)
            if ring_vals_1.size < 10: continue
            alpha = compute_adaptive_alpha_entropy_gpu(ring_vals_1, ring_idx, frame_number)
            alpha_sq = alpha * alpha; eps = 1e-12
            dct1_gpu = dct_1d(ring_vals_1); dct2_gpu = dct_1d(ring_vals_2)
            try: U1,S1,Vt1=svd_gpu(dct1_gpu.reshape(-1,1),full_matrices=False); U2,S2,Vt2=svd_gpu(dct2_gpu.reshape(-1,1),full_matrices=False)
            except cp.linalg.LinAlgError: continue
            s1_g=S1[0] if S1.size>0 else xp.array(0.0); s2_g=S2[0] if S2.size>0 else xp.array(0.0)
            s1=float(s1_g.get()); s2=float(s2_g.get()); modified=False; ns1=s1; ns2=s2
            if bit==0:
                if s1/(s2+eps)<alpha: modified=True; ns1=(s1*alpha_sq+alpha*s2)/(alpha_sq+1.0); ns2=(alpha*s1+s2)/(alpha_sq+1.0)
            else:
                if s1/(s2+eps)>=1.0/(alpha+eps): modified=True; ns1=(s1+alpha*s2)/(1.0+alpha_sq); ns2=(alpha*s1+alpha_sq*s2)/(1.0+alpha_sq)
            if modified:
                mod_count+=1; ns1g=xp.array(ns1); ns2g=xp.array(ns2)
                S1m=xp.zeros((Vt1.shape[0],U1.shape[1]),dtype=xp.float32); S1m[0,0]=ns1g
                S2m=xp.zeros((Vt2.shape[0],U2.shape[1]),dtype=xp.float32); S2m[0,0]=ns2g
                dct1m=(U1@S1m@Vt1).flatten(); dct2m=(U2@S2m@Vt2).flatten()
                rmod1=idct_1d(dct1m); rmod2=idct_1d(dct2m)
                d1=rmod1-ring_vals_1; d2=rmod2-ring_vals_2
                mf1=xp.ones_like(d1); mf2=xp.ones_like(d2)
                if perceptual_mask_gpu is not None:
                    try:
                        rc,cc=coords_cpu[:,0],coords_cpu[:,1]; mask_vals=perceptual_mask_gpu[rc,cc].astype(mf1.dtype,copy=False)
                        mf1=(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mask_vals); mf2=(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mask_vals)
                    except Exception as e_m: logging.warning(f"Apply mask error: {e_m}") # Изменено на warning
                try:  # Применение дельты
                    rc, cc = coords_cpu[:, 0], coords_cpu[:, 1]
                    if d1.size == len(rc): L1_gpu[rc, cc] += d1 * mf1
                    if d2.size == len(rc): L2_gpu[rc, cc] += d2 * mf2
                except IndexError:  # <-- ПРАВИЛЬНЫЙ ОТСТУП
                    logging.error(f"Index error apply delta GPU P:{pair_num_log} R:{ring_idx}")

        if mod_count == 0: return frame1_bgr_cpu, frame2_bgr_cpu

        # --- Обратный DTCWT (NumPy CPU) и завершение на CPU ---
        ll1_mod_cpu = cp.asnumpy(L1_gpu); ll2_mod_cpu = cp.asnumpy(L2_gpu)
        comp1_mod_cpu = dtcwt_inverse_backend(dtcwt_pyramid1, ll1_mod_cpu, frame_number)
        comp2_mod_cpu = dtcwt_inverse_backend(dtcwt_pyramid2, ll2_mod_cpu, frame_number + 1)
        if comp1_mod_cpu is None or comp2_mod_cpu is None: raise RuntimeError("Inverse DTCWT failed")
        comp1_scl=np.clip(comp1_mod_cpu*255.0,0,255).astype(np.uint8); comp2_scl=np.clip(comp2_mod_cpu*255.0,0,255).astype(np.uint8)
        if comp1_scl.shape!=target_shape_cpu: comp1_scl=cv2.resize(comp1_scl,(target_shape_cpu[1],target_shape_cpu[0]))
        if comp2_scl.shape!=target_shape_cpu: comp2_scl=cv2.resize(comp2_scl,(target_shape_cpu[1],target_shape_cpu[0]))
        new_ycrcb1=np.stack((Y1_cpu,Cr1_cpu,Cb1_cpu),axis=-1); new_ycrcb1[:,:,embed_component]=comp1_scl
        new_ycrcb2=np.stack((Y2_cpu,Cr2_cpu,Cb2_cpu),axis=-1); new_ycrcb2[:,:,embed_component]=comp2_scl
        result_frame1_cpu=cv2.cvtColor(new_ycrcb1,cv2.COLOR_YCrCb2BGR); result_frame2_cpu=cv2.cvtColor(new_ycrcb2,cv2.COLOR_YCrCb2BGR)

    except Exception as e:
        # Правильный отступ для except
        logging.error(f"!!! EXCEPTION embed_np_gpu (P:{pair_num_log}): {e}", exc_info=True)
        result_frame1_cpu, result_frame2_cpu = None, None
    finally:
        # Правильный отступ для finally
        # logging.debug(f"[GPU-{gpu_id} P:{pair_num_log}] Cleaning up GPU resources...")
        del comp1_gpu, comp2_gpu, L1_gpu, L2_gpu, perceptual_mask_gpu # Безопасно удалять None
        # Опциональная очистка пула
        # if USE_GPU_ACCELERATION: cp.get_default_memory_pool().free_all_blocks()

    # Логирование и возврат ПОСЛЕ finally, но ВНУТРИ функции
    total_time=time.time()-func_start_time
    logging.info(f"[GPU-{gpu_id} P:{pair_num_log}] Embed NP+GPU Finish. Mods: {mod_count}. Result valid: {result_frame1_cpu is not None}. Time: {total_time:.4f}s")
    return result_frame1_cpu, result_frame2_cpu


# --- Воркер (вызывает embed_frame_pair_numpy_dtcwt_cupy_gpu) ---
# (Код без изменений)
def _embed_frame_pair_worker_numpy_dtcwt_cupy_gpu(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    pair_idx=args['pair_idx']; frame_number=2*pair_idx; frame1_cpu=args['frame1']; frame2_cpu=args['frame2']
    bits_to_embed=args['bits']; gpu_id=args['gpu_id']; n_rings=args['n_rings']; num_rings_to_use=args['num_rings_to_use']
    candidate_pool_size=args['candidate_pool_size']; embed_component=args['embed_component']; use_perceptual_masking=args['use_perceptual_masking']
    if not USE_GPU_ACCELERATION: return frame_number, None, None
    xp=cp; f1_mod, f2_mod = None, None; L1_gpu = None
    try: cp.cuda.Device(gpu_id).use()
    except Exception as e_dev: logging.error(f"Worker GPU-{gpu_id} set dev error: {e_dev}"); return frame_number, None, None
    try:
        if len(bits_to_embed)!=num_rings_to_use: raise ValueError("Bit count mismatch")
        candidate_ring_indices = get_fixed_pseudo_random_rings(pair_idx, n_rings, candidate_pool_size)
        if not candidate_ring_indices or len(candidate_ring_indices) < num_rings_to_use: raise ValueError(f"Not enough candidates: {len(candidate_ring_indices)}")
        target_ring_indices = []
        try:
            comp1_cpu=(frame1_cpu[:,:,embed_component].astype(np.float32)/255.0)
            dtcwt_res=dtcwt_transform_backend(comp1_cpu, frame_number) # NumPy DTCWT
            if dtcwt_res is None: raise RuntimeError("DTCWT(NumPy) failed for L1 calc")
            _, L1_cpu = dtcwt_res; L1_gpu=xp.asarray(L1_cpu); ll_shape=L1_gpu.shape # -> GPU
            rings_coords_cpu_list=ring_division_coords_cpu(ll_shape, n_rings, frame_number)
            entropy_values: List[Tuple[float,int]]=[]; min_pixels=10
            for r_idx in candidate_ring_indices:
                current_entropy=-1.0
                if 0<=r_idx<len(rings_coords_cpu_list):
                    coords_cpu=rings_coords_cpu_list[r_idx]
                    if coords_cpu is not None and coords_cpu.size>=min_pixels*2:
                        ring_vals_gpu=get_ring_values(L1_gpu,coords_cpu)
                        if ring_vals_gpu.size>=min_pixels:
                            v_entropy,_=calculate_entropies_gpu(ring_vals_gpu,frame_number,r_idx)
                            if np.isfinite(v_entropy): current_entropy=v_entropy
                entropy_values.append((current_entropy, r_idx))
            entropy_values.sort(key=lambda x:x[0], reverse=True)
            valid_candidates=[(e,i) for e,i in entropy_values if e>=0.0]
            if len(valid_candidates)<num_rings_to_use: raise ValueError(f"Only {len(valid_candidates)}/{num_rings_to_use} valid rings")
            target_ring_indices=[idx for _,idx in valid_candidates[:num_rings_to_use]]
            logging.info(f"[Worker GPU-{gpu_id} P:{pair_idx}] Selected rings: {target_ring_indices} from {candidate_ring_indices}")
        except Exception as e_select: logging.error(f"Worker GPU-{gpu_id} P:{pair_idx} select error: {e_select}"); raise
        finally: del L1_gpu
        if not target_ring_indices: raise ValueError("No final rings selected")
        f1_mod, f2_mod = embed_frame_pair_numpy_dtcwt_cupy_gpu( # Вызов функции этого варианта
            frame1_cpu, frame2_cpu, bits=bits_to_embed, selected_ring_indices=target_ring_indices,
            n_rings=n_rings, frame_number=frame_number, use_perceptual_masking=use_perceptual_masking,
            embed_component=embed_component, gpu_id=gpu_id
        )
        return frame_number, f1_mod, f2_mod
    except Exception as e: logging.error(f"WORKER EXCEPTION GPU-{gpu_id} P:{pair_idx}: {e}", exc_info=True); return frame_number, None, None

# --- Основная Функция Встраивания ---
# (Код embed_watermark_in_video без изменений, вызывает нужный воркер)
def embed_watermark_in_video(
        frames_cpu: List[np.ndarray], packet_bits_cpu: np.ndarray,
        n_rings: int = N_RINGS, num_rings_to_use: int = NUM_RINGS_TO_USE,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE, bits_per_pair: int = BITS_PER_PAIR,
        max_packet_repeats: int = MAX_PACKET_REPEATS, fps: float = FPS,
        max_workers: Optional[int] = MAX_WORKERS, use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT
):
    num_frames=len(frames_cpu); total_pairs=num_frames//2; packet_len=packet_bits_cpu.size
    if total_pairs==0 or packet_len==0: return frames_cpu[:]
    pairs_needed=ceil(max_packet_repeats*packet_len/bits_per_pair); pairs_proc=min(total_pairs,pairs_needed)
    bits_total=pairs_proc*bits_per_pair; repeats=ceil(bits_total/packet_len) if packet_len>0 else 1
    bits_flat_cpu=np.tile(packet_bits_cpu,repeats)[:bits_total]
    logging.info(f"Starting NumPy DTCWT + CuPy GPU embedding: {bits_total} bits across {pairs_proc} pairs.")
    start_time = time.time(); watermarked_frames_cpu = frames_cpu[:]; tasks_args = []
    num_gpus = cp.cuda.runtime.getDeviceCount() if USE_GPU_ACCELERATION else 0; gpu_cnt = 0
    for pair_idx in range(pairs_proc):
        i1=2*pair_idx; i2=i1+1;
        if i2>=num_frames or frames_cpu[i1] is None or frames_cpu[i2] is None: continue
        bits_start=pair_idx*bits_per_pair; bits_end=bits_start+bits_per_pair
        current_bits=bits_flat_cpu[bits_start:bits_end].tolist()
        if len(current_bits)!=bits_per_pair: continue
        gpu_id = gpu_cnt % num_gpus if num_gpus > 0 else 0; gpu_cnt += 1
        args={'pair_idx':pair_idx, 'frame1':frames_cpu[i1], 'frame2':frames_cpu[i2], 'bits':current_bits, 'n_rings':n_rings, 'num_rings_to_use':num_rings_to_use, 'candidate_pool_size':candidate_pool_size, 'use_perceptual_masking':use_perceptual_masking, 'embed_component':embed_component, 'gpu_id':gpu_id}
        tasks_args.append(args)
    if not tasks_args: return watermarked_frames_cpu
    results: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    proc_cnt=0; err_cnt=0; task_cnt=len(tasks_args)
    worker_func = _embed_frame_pair_worker_numpy_dtcwt_cupy_gpu if USE_GPU_ACCELERATION else None
    if worker_func is None: logging.error("GPU ускорение недоступно/отключено."); return frames_cpu[:]
    try:
        logging.info(f"Submitting {task_cnt} tasks (NumPy DTCWT + CuPy GPU worker)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            f2idx1 = {executor.submit(worker_func, arg): arg['pair_idx']*2 for arg in tasks_args}
            for i, future in enumerate(concurrent.futures.as_completed(f2idx1)):
                idx1=f2idx1[future]; pidx=idx1//2
                try: fnum_check, f1m, f2m = future.result()
                except Exception as exc_future: logging.error(f'Pair {pidx} Future Exception: {exc_future}'); err_cnt+=1; continue
                if fnum_check != idx1: logging.error("Worker index mismatch!")
                if f1m is not None and f2m is not None: results[idx1]=(f1m, f2m); proc_cnt+=1
                else: err_cnt+=1; logging.error(f"Pair {pidx} failed (worker None).")
    except Exception as e: logging.critical(f"Executor error: {e}", exc_info=True); return frames_cpu[:]
    logging.info(f"Executor finished. Success: {proc_cnt}, Failed: {err_cnt}.")
    upd_cnt=0
    for idx1, (f1m, f2m) in results.items():
        idx2 = idx1+1
        if idx1<len(watermarked_frames_cpu): watermarked_frames_cpu[idx1]=f1m; upd_cnt+=1
        if idx2<len(watermarked_frames_cpu): watermarked_frames_cpu[idx2]=f2m; upd_cnt+=1
    logging.info(f"Applied results: {len(results)} pairs ({upd_cnt} frames updated).")
    end_time = time.time(); logging.info(f"Embedding finished. Time: {end_time - start_time:.2f} sec.")
    return watermarked_frames_cpu

# --- Main ---
def main():
    # (Код main без изменений)
    main_start_time = time.time()
    input_video = "input.mp4"
    base_output_name = f"watermarked_v5_npDTCWT_cupyGPU_{OUTPUT_CODEC}" # Новое имя
    output_video = base_output_name + OUTPUT_EXTENSION
    logging.info(f"--- Starting Embedding Main Process (NumPy DTCWT + CuPy GPU, Output: {output_video}) ---")
    frames_cpu, input_fps = read_video(input_video)
    if not frames_cpu or len(frames_cpu) < 2: print("Read failed or video too short."); return
    fps_to_use = float(FPS) if input_fps <= 0 else input_fps
    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES)
    original_id_hex = original_id_bytes.hex()
    logging.info(f"Generated Payload ID: {original_id_hex}")
    try: open(ORIGINAL_WATERMARK_FILE, "w").write(original_id_hex); logging.info("Original ID saved.")
    except IOError as e: logging.error(f"Could not save ID: {e}")
    payload_bits_np_cpu = np.unpackbits(np.frombuffer(original_id_bytes, dtype=np.uint8))
    packet_to_embed_cpu: np.ndarray = payload_bits_np_cpu
    if effective_use_ecc and bch is not None:
        packet_ecc = add_ecc(payload_bits_np_cpu, bch)
        if packet_ecc is not None and packet_ecc.size > payload_bits_np_cpu.size:
            packet_to_embed_cpu = packet_ecc; logging.info(f"ECC added. Packet size: {packet_to_embed_cpu.size} bits.")
        else: logging.warning("Using raw payload (ECC add failed).")
    else: logging.info(f"Using raw payload (ECC disabled/failed). Packet size: {packet_to_embed_cpu.size} bits.")
    watermarked_frames_cpu = embed_watermark_in_video( frames_cpu=frames_cpu, packet_bits_cpu=packet_to_embed_cpu, n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE, candidate_pool_size=CANDIDATE_POOL_SIZE, bits_per_pair=BITS_PER_PAIR, max_packet_repeats=MAX_PACKET_REPEATS, fps=fps_to_use, max_workers=MAX_WORKERS, use_perceptual_masking=USE_PERCEPTUAL_MASKING, embed_component=EMBED_COMPONENT )
    if watermarked_frames_cpu and len(watermarked_frames_cpu) == len(frames_cpu):
        write_video(watermarked_frames_cpu, output_video, fps=fps_to_use, codec=OUTPUT_CODEC)
        logging.info(f"Watermarked video saved to: {output_video}")
        try:
            if os.path.exists(output_video): logging.info(f"Output size: {os.path.getsize(output_video)/(1024*1024):.2f} MB")
            else: logging.error("Output file not created.")
        except OSError as e: logging.error(f"Could not get file size: {e}")
    else: logging.error("Embedding failed. Output not saved."); print("ERROR: Embedding failed.")
    logging.info("--- Embedding Main Process Finished ---")
    total_time = time.time() - main_start_time
    logging.info(f"--- Total Embedder Script Time: {total_time:.2f} sec ---")
    print(f"\nEmbedding finished."); print(f"Output: {output_video}"); print(f"Logs: {LOG_FILENAME}"); print(f"Original ID: {ORIGINAL_WATERMARK_FILE}")
    print("\nRun extractor (updated for NumPy DTCWT + CuPy GPU) to verify.")


# --- Запуск ---
if __name__ == "__main__":
    if not USE_GPU_ACCELERATION: print("\nFATAL: CuPy/GPU not available. This script requires GPU."); exit()
    if DTCWT_BACKEND_SET == 'unavailable': print("\nFATAL: dtcwt init failed."); exit()
    if USE_ECC and not BCHLIB_AVAILABLE: print("\nWARNING: bchlib not installed. ECC disabled.")
    try: import imageio.v3 as iio
    except ImportError: print("\nFATAL: imageio library missing. Install: pip install imageio[ffmpeg]"); exit()

    profiler = cProfile.Profile(); profiler.enable()
    try: main()
    except Exception as e: logging.critical(f"Unhandled main exception: {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. Logs: {LOG_FILENAME}")
    finally:
        profiler.disable(); stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        profile_file = "profile_stats_embed_npDTCWT_cupyGPU.txt"
        try:
            with open(profile_file, "w") as f: stats.stream=f; stats.print_stats()
            logging.info(f"Profiling stats saved to {profile_file}"); print(f"Profiling stats saved to {profile_file}")
        except IOError as e: logging.error(f"Could not save profile stats: {e}")