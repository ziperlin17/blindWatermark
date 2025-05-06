# Файл: embedder_pytorch_wavelets.py (ПОСЛЕ РЕФАКТОРИНГА)
import gc
import math
import shutil
import subprocess
import tempfile

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
import hashlib

from av.video.reformatter import VideoReformatter
from scipy.fftpack import dct as scipy_dct, idct as scipy_idct
from scipy.linalg import svd as scipy_svd

from galois import BCH
# from PIL import Image # Не используется
from line_profiler import line_profiler, profile # Раскомментировать для профилирования
# --- SciPy убираем (или оставляем только SVD, если PyTorch SVD не подойдет) ---
# from scipy.fftpack import dct as scipy_dct, idct as scipy_idct
# from scipy.linalg import svd as scipy_svd
# --- PyTorch импорты ---
import torch
import torch.nn.functional as F
try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    class DTCWTForward: pass
    class DTCWTInverse: pass
    logging.error("Библиотека pytorch_wavelets не найдена!")
try:
    # Используем нашу новую библиотеку для DCT/IDCT
    import torch_dct as dct_torch # Импортируем под псевдонимом
    TORCH_DCT_AVAILABLE = True
except ImportError:
    TORCH_DCT_AVAILABLE = False
    logging.error("Библиотека torch-dct не найдена! Невозможно использовать PyTorch DCT.")
# --------------------------
from typing import List, Tuple, Optional, Dict, Any, Set
import uuid
from math import ceil
import cProfile
import pstats
from fractions import Fraction

# --- Импорт PyAV ---
try:
    import av
    # Импортируем базовый класс ошибок FFmpeg
    from av import FFmpegError, VideoFrame  # Базовый класс доступен напрямую из av
    # Импортируем специфичные ошибки, которые хотим ловить отдельно (если нужно)
    # Например:
    from av import EOFError as FFmpegEOFError # Переименуем, чтобы не конфликтовать со встроенным
    from av import ValueError as FFmpegValueError # Переименуем
    # Или можно просто ловить встроенные ValueError/EOFError, так как ошибки PyAV от них наследуются

    PYAV_AVAILABLE = True
    logging.info("PyAV library imported successfully.")

except ImportError:
    PYAV_AVAILABLE = False
    logging.error("PyAV library not found! Install it: pip install av")
    # --- Определим классы-пустышки ---
    class av_dummy: # Используем другое имя
        class VideoFrame: pass
        class AudioFrame: pass
        class Packet: pass
        class TimeBase: pass
        class container:
            class Container: pass
        # Добавим заглушки для классов ошибок, которые мы импортировали выше
        FFmpegError = Exception # Базовый тип - Exception
        EOFError = EOFError     # Используем встроенный для заглушки
        ValueError = ValueError # Используем встроенный для заглушки
        NotFoundError = Exception
    # Переопределяем сам модуль 'av' и классы ошибок глобально ТОЛЬКО при ошибке импорта
    av = av_dummy
    FFmpegError = Exception
    FFmpegEOFError = EOFError # Используем встроенный
    FFmpegValueError = ValueError # Используем встроенный

try:
    import galois
    BCH_TYPE = galois.BCH; GALOIS_IMPORTED = True; logging.info("galois library imported.")
except ImportError:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.info("galois library not found.")
except Exception as import_err:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.error(f"Galois import error: {import_err}", exc_info=True)

CODEC_CONTAINER_COMPATIBILITY: Dict[str, Set[Tuple[str, str]]] = {
    ".mp4": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('audio', 'aac'), ('audio', 'mp3'), ('audio', 'alac')},
    ".mov": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('video', 'prores'), ('audio', 'aac'), ('audio', 'mp3'), ('audio', 'alac'), ('audio', 'pcm_s16le')},
    ".mkv": {('video', 'h264'), ('video', 'hevc'), ('video', 'vp9'), ('video', 'av1'), ('video', 'mpeg4'), ('video', 'mpeg2video'), ('video', 'theora'), ('video', 'prores'), ('audio', 'aac'), ('audio', 'opus'), ('audio', 'vorbis'), ('audio', 'flac'), ('audio', 'ac3'), ('audio', 'dts'), ('audio', 'mp3'), ('audio', 'pcm_s16le'), ('audio', 'alac')},
    ".webm": {('video', 'vp8'), ('video', 'vp9'), ('video', 'av1'), ('audio', 'opus'), ('audio', 'vorbis')},
}
DEFAULT_OUTPUT_CONTAINER_EXT_FINAL: str = ".mp4"  # Для финального файла
DEFAULT_VIDEO_ENCODER_LIB_FOR_HEAD: str = "libx264" # Для временного файла головы
DEFAULT_VIDEO_CODEC_NAME_FOR_HEAD: str = "h264"    # Для временного файла головы
DEFAULT_AUDIO_CODEC_FOR_FFMPEG: str = "aac"        # Если FFmpeg нужно перекодировать аудио хвоста
FALLBACK_CONTAINER_EXT_FINAL: str = ".mkv"       # Для финального файла


# --- Глобальные Параметры ---
# (Остаются те же, что и в вашей последней версии embedder)
LAMBDA_PARAM: float = 0.05 # Вернул ваше значение
ALPHA_MIN: float = 1.13
ALPHA_MAX: float = 1.28 # Ваше новое значение
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
BCH_T: int = 9
FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed_pytorch.log'
OUTPUT_CODEC: str = 'mp4v'
OUTPUT_EXTENSION: str = '.mp4'
SELECTED_RINGS_FILE: str = 'selected_rings_embed_pytorch.json'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS: Optional[int] = 8
MAX_TOTAL_PACKETS = 15
SAFE_MAX_WORKERS = 8

# --- Инициализация Галуа (с t=9, k=187) ---
BCH_CODE_OBJECT: Optional['galois.BCH'] = None  # Аннотация в кавычках для отложенного импорта
GALOIS_AVAILABLE = False
try:
    import galois

    logging.info("galois: импортирован.")
    _test_bch_ok = False;
    _test_decode_ok = False
    try:
        _test_m = BCH_M
        _test_t = BCH_T
        _test_n = (1 << _test_m) - 1
        _test_d = 2 * _test_t + 1

        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)

        # Определяем ожидаемое k на основе t
        if _test_t == 5:
            expected_k = 215
        elif _test_t == 7:
            expected_k = 201
        elif _test_t == 9:
            expected_k = 187  # <-- Для t=9
        elif _test_t == 11:
            expected_k = 173
        elif _test_t == 15:
            expected_k = 131
        else:
            logging.error(f"Неизвестное ожидаемое k для t={_test_t}")
            expected_k = -1

        if expected_k != -1 and _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
            logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
            _test_bch_ok = True;
            BCH_CODE_OBJECT = _test_bch_galois
        else:
            logging.error(
                f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t={_test_bch_galois.t}, k={_test_bch_galois.k}.")
            _test_bch_ok = False;
            BCH_CODE_OBJECT = None

        # Тест декодирования (если инициализация прошла)
        if _test_bch_ok and BCH_CODE_OBJECT is not None:
            _n_bits = BCH_CODE_OBJECT.n
            _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)
            GF2 = galois.GF(2);
            _dummy_cw_vec = GF2(_dummy_cw_bits)
            # Подавляем вывод ошибок декодера в тесте
            _msg, _flips = BCH_CODE_OBJECT.decode(_dummy_cw_vec, errors=True)
            if _flips is not None:  # Проверяем, что декодирование не вызвало исключение
                logging.info(f"galois: decode() test OK (flips={_flips}).")
                _test_decode_ok = True
            else:
                # decode() может вернуть None для _flips если ошибок 0, проверим и так
                _test_decode_ok = True  # Считаем тест пройденным, если исключения не было
                logging.info("galois: decode() test potentially OK (flips is None/0).")
    except ValueError as ve:
        logging.error(f"galois: ОШИБКА ValueError при инициализации BCH: {ve}")
        BCH_CODE_OBJECT = None;
        _test_bch_ok = False
    except Exception as test_err:
        logging.error(f"galois: ОШИБКА теста инициализации/декодирования: {test_err}", exc_info=True)
        BCH_CODE_OBJECT = None;
        _test_bch_ok = False

    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if GALOIS_AVAILABLE:
        logging.info("galois: Тесты пройдены, объект BCH доступен.")
    else:
        logging.warning("galois: Тесты НЕ ПРОЙДЕНЫ или объект BCH не создан.")

except ImportError:
    GALOIS_AVAILABLE = False;
    BCH_CODE_OBJECT = None
    logging.info("galois library not found.")
except Exception as import_err:
    GALOIS_AVAILABLE = False;
    BCH_CODE_OBJECT = None
    logging.error(f"galois: Ошибка импорта: {import_err}", exc_info=True)

# --- Настройка логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG) # Раскомментировать для детального лога

# --- Логирование конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Встраивания (PyTorch Wavelets) ---")
logging.info(f"PyTorch Wavelets Доступно: {PYTORCH_WAVELETS_AVAILABLE}")
logging.info(f"Torch DCT Доступно: {TORCH_DCT_AVAILABLE}")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Payload: {PAYLOAD_LEN_BYTES * 8}bit, ECC for 1st: {effective_use_ecc} (Galois BCH m={BCH_M}, t={BCH_T})")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(
    f"Маскировка: {USE_PERCEPTUAL_MASKING} (Lambda={LAMBDA_PARAM}), Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS or 'default'}) с батчингом.")
if USE_ECC and not GALOIS_AVAILABLE:
    logging.warning("ECC вкл, но galois недоступна/не работает! Первый пакет будет Raw.")
elif not USE_ECC:
    logging.info("ECC выкл для первого пакета.")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG', 'DIVX']: logging.warning(f"Используется кодек с потерями '{OUTPUT_CODEC}'.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE:
    logging.error(f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) > CANDIDATE_POOL_SIZE ({CANDIDATE_POOL_SIZE})! Исправлено.")
    NUM_RINGS_TO_USE = CANDIDATE_POOL_SIZE
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(
    f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) != BITS_PER_PAIR ({BITS_PER_PAIR}).")


# --- Базовые Функции (с изменениями для PyTorch) ---

def dct1d_torch(s_tensor: torch.Tensor) -> torch.Tensor:
    """1D DCT-II используя torch-dct."""
    if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct не доступен")
    # dct ожидает тензор и применяет преобразование к последнему измерению
    return dct_torch.dct(s_tensor, norm='ortho')

def idct1d_torch(c_tensor: torch.Tensor) -> torch.Tensor:
    """1D IDCT-III используя torch-dct."""
    if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct не доступен")
    return dct_torch.idct(c_tensor, norm='ortho')

def svd_torch(tensor_1d: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Применяет SVD к 1D тензору (рассматривая как столбец) используя torch.linalg.svd."""
    try:
        # Добавляем измерение, чтобы сделать его 2D (N, 1)
        tensor_2d = tensor_1d.unsqueeze(-1)
        # full_matrices=False не так важно для Nx1 матрицы
        U, S, Vh = torch.linalg.svd(tensor_2d, full_matrices=False)
        # Vh для матрицы Nx1 будет тензором 1x1. Нам нужен V, который тоже 1x1.
        # S - вектор сингулярных чисел (в нашем случае одно)
        # U - матрица Nx1
        return U, S, Vh.T # Возвращаем V = Vh.T
    except Exception as e:
        logging.error(f"PyTorch SVD error: {e}", exc_info=True)
        return None, None, None

# --- НОВАЯ функция обертка для PyTorch DTCWT Forward ---
def dtcwt_pytorch_forward(yp_tensor: torch.Tensor, xfm: DTCWTForward, device: torch.device, fn: int = -1) -> Tuple[
    Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Применяет прямое DTCWT PyTorch к одному каналу (2D тензору)."""
    if not PYTORCH_WAVELETS_AVAILABLE:  # Проверка доступности библиотеки
        logging.error("PyTorch Wavelets не доступна для dtcwt_pytorch_forward.")
        return None, None
    if not isinstance(yp_tensor, torch.Tensor):
        logging.error(f"[Frame:{fn}] Input is not a torch Tensor.")
        return None, None
    if yp_tensor.ndim != 2:
        logging.error(f"[Frame:{fn}] Input tensor must be 2D (H, W), got {yp_tensor.shape}")
        return None, None
    try:
        # Добавляем Batch и Channel измерения, перемещаем на device
        # Убедимся, что тензор имеет тип float32
        yp_tensor = yp_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)  # -> (1, 1, H, W)
        xfm = xfm.to(device)

        with torch.no_grad():  # Отключаем расчет градиентов для экономии памяти/скорости
            Yl, Yh = xfm(yp_tensor)  # Yh - список

        # Проверка результата
        if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh:
            logging.error(f"[Frame:{fn}] DTCWTForward вернула некорректный результат (None или пустой Yh).")
            return None, None

        # logging.debug(f"[Frame:{fn}] PyTorch DTCWT FWD done. Yl shape: {Yl.shape}, Yh[0] shape: {Yh[0].shape}")
        # Возвращаем тензоры как есть (на device)
        return Yl, Yh
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error(f"[Frame:{fn}] CUDA out of memory during PyTorch DTCWT forward!", exc_info=True)
        else:
            logging.error(f"[Frame:{fn}] PyTorch DTCWT forward runtime error: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logging.error(f"[Frame:{fn}] PyTorch DTCWT forward unexpected error: {e}", exc_info=True)
        return None, None


# --- НОВАЯ функция обертка для PyTorch DTCWT Inverse ---
def dtcwt_pytorch_inverse(Yl: torch.Tensor, Yh: List[torch.Tensor], ifm: DTCWTInverse, device: torch.device,
                          target_shape: Tuple[int, int], fn: int = -1) -> Optional[np.ndarray]:
    """Применяет обратное DTCWT PyTorch и возвращает NumPy массив float32."""
    if not PYTORCH_WAVELETS_AVAILABLE:
        logging.error("PyTorch Wavelets не доступна для dtcwt_pytorch_inverse.")
        return None
    if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh:
        logging.error(f"[Frame:{fn}] Invalid input for inverse (Yl or Yh is None/empty list).")
        return None
    try:
        # Перемещаем все на device
        Yl = Yl.to(device)
        Yh = [h.to(device) for h in Yh if h is not None and h.numel() > 0]  # Фильтруем пустые/None
        ifm = ifm.to(device)

        # Убедимся, что Yh не стал пустым после фильтрации
        if not Yh:
            logging.error(f"[Frame:{fn}] Yh list is empty after filtering Nones/empty tensors.")
            return None

        with torch.no_grad():  # Отключаем градиенты
            reconstructed_X_tensor = ifm((Yl, Yh))

        # Убираем batch и channel измерения
        if reconstructed_X_tensor.dim() == 4 and reconstructed_X_tensor.shape[0] == 1 and reconstructed_X_tensor.shape[
            1] == 1:
            reconstructed_X_tensor = reconstructed_X_tensor.squeeze(0).squeeze(0)  # (H, W)
        elif reconstructed_X_tensor.dim() != 2:
            logging.error(f"[Frame:{fn}] Unexpected output dimension from inverse: {reconstructed_X_tensor.dim()}")
            return None

        # logging.debug(f"[Frame:{fn}] PyTorch DTCWT INV done. Output shape: {reconstructed_X_tensor.shape}")

        # Обрезаем до нужного размера
        current_h, current_w = reconstructed_X_tensor.shape
        target_h, target_w = target_shape
        if current_h > target_h or current_w > target_w:
            logging.warning(
                f"[Frame:{fn}] Inverse result shape {reconstructed_X_tensor.shape} > target {target_shape}. Cropping.")
            # Убедимся, что не обрезаем до нуля или отрицательного размера
            if target_h > 0 and target_w > 0:
                reconstructed_X_tensor = reconstructed_X_tensor[:target_h, :target_w]
            else:
                logging.error(f"[Frame:{fn}] Invalid target shape for cropping: {target_shape}")
                return None
        elif current_h < target_h or current_w < target_w:
            logging.warning(
                f"[Frame:{fn}] Inverse result shape {reconstructed_X_tensor.shape} < target {target_shape}. Padding might be needed if this causes issues.")
            # Паддинг обычно не требуется, т.к. inverse восстанавливает размер

        # Перемещаем на CPU и конвертируем в NumPy float32
        reconstructed_np = reconstructed_X_tensor.cpu().numpy().astype(np.float32)

        if np.any(np.isnan(reconstructed_np)):
            logging.warning(f"[Frame:{fn}] NaN found after PyTorch inverse!")

        return reconstructed_np
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error(f"[Frame:{fn}] CUDA out of memory during PyTorch DTCWT inverse!", exc_info=True)
        else:
            logging.error(f"[Frame:{fn}] PyTorch DTCWT inverse runtime error: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"[Frame:{fn}] PyTorch DTCWT inverse unexpected error: {e}", exc_info=True)
        return None


# --- ПЕРЕПИСАННАЯ ring_division для PyTorch ---
def ring_division(lp_tensor: torch.Tensor, nr: int = N_RINGS, fn: int = -1) -> List[Optional[torch.Tensor]]:
    """Разбивает 2D PyTorch тензор на N концентрических колец. Возвращает список тензоров координат."""
    if not isinstance(lp_tensor, torch.Tensor) or lp_tensor.ndim != 2:
        logging.error(
            f"[Frame:{fn}] Invalid input for ring_division (expected 2D torch.Tensor). Got {type(lp_tensor)} with ndim {lp_tensor.ndim if hasattr(lp_tensor, 'ndim') else 'N/A'}")
        return [None] * nr

    H, W = lp_tensor.shape
    if H < 2 or W < 2:
        logging.warning(f"[Frame:{fn}] Tensor too small for ring division ({H}x{W})")
        return [None] * nr
    device = lp_tensor.device

    try:
        # Создаем сетку координат
        rr, cc = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32),
                                indexing='ij')  # indexing='ij' важно для H, W порядка

        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
        distances = torch.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)

        min_dist, max_dist = torch.tensor(0.0, device=device), torch.max(distances)

        # Границы колец
        if max_dist < 1e-9:  # Избегаем деления на ноль или проблем с linspace
            logging.warning(f"[Frame:{fn}] Max distance in ring division is near zero ({max_dist}).")
            # Все пиксели попадут в первое кольцо
            ring_bins = torch.tensor([0.0, max_dist + 1e-6] + [max_dist + 1e-6] * (nr - 1), device=device)
        else:
            ring_bins = torch.linspace(min_dist.item(), (max_dist + 1e-6).item(), nr + 1, device=device)

        # Назначение индексов кольца
        # Используем маски для большей ясности и избежания проблем с bucketize
        ring_indices = torch.zeros_like(distances, dtype=torch.long) - 1  # Инициализируем -1
        for i in range(nr):
            lower_bound = ring_bins[i]
            upper_bound = ring_bins[i + 1]
            # Маска для текущего кольца
            # Включаем нижнюю границу, исключаем верхнюю (кроме последнего кольца)
            if i < nr - 1:
                mask = (distances >= lower_bound) & (distances < upper_bound)
            else:  # Последнее кольцо включает верхнюю границу
                mask = (distances >= lower_bound) & (distances <= upper_bound)  # <= для max_dist
            ring_indices[mask] = i

        # Убедимся, что центр в первом кольце (если вдруг не попал из-за точности float)
        ring_indices[distances < ring_bins[1]] = 0

        rings: List[Optional[torch.Tensor]] = [None] * nr
        for rdx in range(nr):
            # Находим координаты (индексы) пикселей для кольца rdx
            coords_tensor = torch.nonzero(ring_indices == rdx, as_tuple=False)  # -> shape (N_pixels, 2)
            if coords_tensor.shape[0] > 0:
                rings[rdx] = coords_tensor.long()  # Сохраняем как LongTensor
            else:  # Кольцо пустое
                logging.debug(f"[Frame:{fn}] Ring {rdx} is empty.")

        return rings
    except Exception as e:
        logging.error(f"Ring division PyTorch error Frame {fn}: {e}", exc_info=True)
        return [None] * nr


# --- calculate_entropies, compute_adaptive_alpha_entropy - остаются на NumPy ---
# Они принимают 1D NumPy массив значений кольца
def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    eps = 1e-12;
    shannon_entropy = 0.;
    collision_entropy = 0.  # Ваша старая формула давала ee

    if rv.size > 0:
        rv_processed = np.clip(rv.copy(), 0.0, 1.0)  # Работаем с копией и клиппингом
        if np.all(rv_processed == rv_processed[0]): return 0.0, 0.0  # Энтропия константы 0

        hist, _ = np.histogram(rv_processed, bins=256, range=(0., 1.), density=False)
        total_count = rv_processed.size
        if total_count > 0:
            probabilities = hist / total_count
            p = probabilities[probabilities > eps]  # Убираем нулевые вероятности
            if p.size > 0:
                shannon_entropy = -np.sum(p * np.log2(p))
                # Используем Реньи 2-го порядка как collision entropy
                # collision_entropy = -np.log2(np.sum(p**2)) if np.sum(p**2) > eps else 0.0
                # Возвращаем вашу старую ee для совместимости, если она нужна где-то еще
                ee = -np.sum(p * np.exp(1. - p))
                collision_entropy = ee  # Присваиваем ee, чтобы сигнатура не менялась
    return shannon_entropy, collision_entropy  # Возвращаем (ve, ee)


def compute_adaptive_alpha_entropy(rv: np.ndarray, ri: int, fn: int) -> float:
    if rv.size < 10: return ALPHA_MIN  # Мало данных для статистики
    # Используем shannon_entropy (первый элемент кортежа)
    ve, _ = calculate_entropies(rv, fn, ri)
    lv = np.var(rv)
    # Проверка на NaN/inf перед использованием
    if not np.isfinite(ve) or not np.isfinite(lv):
        logging.warning(f"[F:{fn}, R:{ri}] Non-finite entropy ({ve}) or variance ({lv}). Using ALPHA_MIN.")
        return ALPHA_MIN

    en = np.clip(ve / MAX_THEORETICAL_ENTROPY, 0., 1.)
    vmp = 0.005;
    vsc = 500
    # Использование np.exp может дать overflow если lv очень велико, добавим clip или обработку
    try:
        exp_term = np.exp(-vsc * (lv - vmp))
    except OverflowError:
        exp_term = 0.0  # Если экспонента уходит в -inf, результат 0
    tn = 1. / (1. + exp_term) if (1. + exp_term) != 0 else 1.0  # Избегаем деления на ноль

    we = .6;
    wt = .4
    mf = np.clip((we * en + wt * tn), 0., 1.)
    fa = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * mf
    logging.debug(f"[F:{fn}, R:{ri}] Alpha={fa:.4f} (E={ve:.3f},V={lv:.6f})")
    return np.clip(fa, ALPHA_MIN, ALPHA_MAX)


# --- get_fixed_pseudo_random_rings - остается без изменений ---
def get_fixed_pseudo_random_rings(pi: int, nr: int, ps: int) -> List[int]:
    """
    Генерирует детерминированный псевдослучайный набор индексов колец.
    pi: Индекс пары (или другой уникальный идентификатор).
    nr: Общее количество колец (N_RINGS).
    ps: Размер пула кандидатов (CANDIDATE_POOL_SIZE).
    """
    if ps <= 0:
        logging.warning(f"Pool size {ps} <= 0, returning empty list.")
        return []
    if ps > nr:
        logging.warning(f"Pool size {ps} > N_RINGS {nr}. Clamping pool size to {nr}.")
        ps = nr

    # Используем индекс пары как сид для PRNG
    seed_str = str(pi).encode('utf-8')
    hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big')
    prng = random.Random(seed_int)

    # Генерируем выборку без повторений
    try:
        candidate_indices = prng.sample(range(nr), ps)
    except ValueError:
        # Это может случиться, если nr < ps, но мы уже обработали это выше.
        # На всякий случай, если nr == 0:
        if nr == 0:
            candidate_indices = []
        else:
            # Альтернативный метод, если sample по какой-то причине не сработал
            logging.warning(f"prng.sample failed for P:{pi}, nr:{nr}, ps:{ps}. Using shuffle fallback.")
            candidate_indices = list(range(nr))
            prng.shuffle(candidate_indices)
            candidate_indices = candidate_indices[:ps]

    # logging.debug(f"[P:{pi}] Candidate rings: {candidate_indices}") # Можно раскомментировать
    return candidate_indices

# --- calculate_perceptual_mask - АДАПТИРОВАНА под тензоры ---
def calculate_perceptual_mask(ip_tensor: torch.Tensor, device: torch.device, fn: int = -1) -> Optional[torch.Tensor]:
    """Вычисляет перцептуальную маску для 2D тензора."""
    if not isinstance(ip_tensor, torch.Tensor) or ip_tensor.ndim != 2:
        logging.error(f"Mask error F{fn}: Input is not a 2D tensor.")
        return torch.ones_like(ip_tensor, device=device)  # Возвращаем единичную маску
    try:
        # Конвертация в NumPy для OpenCV
        pf = ip_tensor.cpu().numpy().astype(np.float32)
        # Проверка на NaN/inf после конвертации
        if not np.all(np.isfinite(pf)):
            logging.warning(f"Mask error F{fn}: Input tensor contains NaN/inf.")
            return torch.ones_like(ip_tensor, device=device)

        # Операции OpenCV
        gx = cv2.Sobel(pf, cv2.CV_32F, 1, 0, ksize=3);
        gy = cv2.Sobel(pf, cv2.CV_32F, 0, 1, ksize=3)
        # Проверка на NaN/inf после Sobel
        if not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
            logging.warning(f"Mask error F{fn}: Sobel result contains NaN/inf.")
            return torch.ones_like(ip_tensor, device=device)

        gm = np.sqrt(gx ** 2 + gy ** 2)
        ks = (11, 11);
        s = 5
        lm = cv2.GaussianBlur(pf, ks, s);
        lms = cv2.GaussianBlur(pf ** 2, ks, s)
        # Проверка на NaN/inf после GaussianBlur
        if not np.all(np.isfinite(lm)) or not np.all(np.isfinite(lms)):
            logging.warning(f"Mask error F{fn}: GaussianBlur result contains NaN/inf.")
            return torch.ones_like(ip_tensor, device=device)

        # np.maximum(lms-lm**2,0) - защита от отрицательных под корнем
        lv = np.sqrt(np.maximum(lms - lm ** 2, 0))
        if not np.all(np.isfinite(lv)):  # Проверка после sqrt
            logging.warning(f"Mask error F{fn}: Local variance result contains NaN/inf.")
            # Попробуем заменить NaN/inf нулями перед np.maximum(gm, lv)
            lv = np.nan_to_num(lv, nan=0.0, posinf=0.0, neginf=0.0)
            # return torch.ones_like(ip_tensor, device=device) # Раньше возвращали единицы

        cm = np.maximum(gm, lv)
        eps = 1e-9;
        mc = np.max(cm)

        # Проверка mc
        if not np.isfinite(mc):
            logging.warning(f"Mask error F{fn}: Max complexity (mc) is not finite.")
            return torch.ones_like(ip_tensor, device=device)

        mn = cm / (mc + eps) if mc > eps else np.zeros_like(cm)
        mask_np = np.clip(mn, 0., 1.).astype(np.float32)

        # Конвертация обратно в Tensor
        mask_tensor = torch.from_numpy(mask_np).to(device)
        # logging.debug(f"Mask F{fn}: range {torch.min(mask_tensor):.2f}-{torch.max(mask_tensor):.2f}")
        return mask_tensor
    except cv2.error as cv_err:
        logging.error(f"Mask OpenCV error F{fn}: {cv_err}", exc_info=True)
        return torch.ones_like(ip_tensor, device=device)
    except Exception as e:
        logging.error(f"Mask general error F{fn}: {e}", exc_info=True)
        return torch.ones_like(ip_tensor, device=device)


# --- add_ecc - остается без изменений (работает с NumPy) ---
def add_ecc(data_bits: np.ndarray, bch_code: Optional[galois.BCH]) -> Optional[np.ndarray]:
    if not GALOIS_AVAILABLE or bch_code is None:
        logging.warning("ECC не доступен или не предоставлен, возвращаем исходные биты.")
        return data_bits  # Возвращаем как есть, если ECC нет
    try:
        k = bch_code.k;
        n = bch_code.n
        if data_bits.size > k:
            logging.error(f"ECC Error: Data size ({data_bits.size}) > k ({k})")
            return None
        pad_len = k - data_bits.size
        # Убедимся, что data_bits - это 1D массив uint8
        msg_bits = data_bits.astype(np.uint8).flatten()
        if pad_len > 0:
            msg_bits = np.pad(msg_bits, (0, pad_len), 'constant')

        GF = bch_code.field;
        msg_vec = GF(msg_bits);
        cw_vec = bch_code.encode(msg_vec)
        pkt_bits = cw_vec.view(np.ndarray).astype(np.uint8)

        if pkt_bits.size != n:  # Используем if вместо assert для мягкой ошибки
            logging.error(f"ECC Error: Output packet size ({pkt_bits.size}) != n ({n})")
            return None
        logging.info(f"Galois ECC: Data({data_bits.size}b->{k}b) -> Packet({pkt_bits.size}b).")
        return pkt_bits
    except Exception as e:
        logging.error(f"Galois encode error: {e}", exc_info=True)
        return None


CODEC_CONTAINER_COMPATIBILITY: Dict[str, Set[Tuple[str, str]]] = {
    ".mp4": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('audio', 'aac'), ('audio', 'mp3'),
             ('audio', 'alac')},
    ".mov": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('video', 'prores'), ('audio', 'aac'),
             ('audio', 'mp3'), ('audio', 'alac'), ('audio', 'pcm_s16le')},
    ".mkv": {('video', 'h264'), ('video', 'hevc'), ('video', 'vp9'), ('video', 'av1'), ('video', 'mpeg4'),
             ('video', 'mpeg2video'), ('video', 'theora'), ('video', 'prores'), ('audio', 'aac'), ('audio', 'opus'),
             ('audio', 'vorbis'), ('audio', 'flac'), ('audio', 'ac3'), ('audio', 'dts'), ('audio', 'mp3'),
             ('audio', 'pcm_s16le'), ('audio', 'alac')},
    ".webm": {('video', 'vp8'), ('video', 'vp9'), ('video', 'av1'), ('audio', 'opus'), ('audio', 'vorbis')},
}
DEFAULT_OUTPUT_CONTAINER_EXT_FINAL: str = ".mp4"  # Для финального файла
DEFAULT_VIDEO_ENCODER_LIB_FOR_HEAD: str = "libx264"  # Для временного файла головы
DEFAULT_VIDEO_CODEC_NAME_FOR_HEAD: str = "h264"  # Имя кодека для головы
DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL: str = "aac"  # Если FFmpeg нужно перекодировать аудио хвоста
FALLBACK_CONTAINER_EXT_FINAL: str = ".mkv"  # Для финального файла


# --- ПОЛНАЯ ФУНКЦИЯ: check_compatibility_and_choose_output ---
def check_compatibility_and_choose_output(
        input_metadata: Dict[str, Any]
) -> Tuple[str, str, str]:
    """
    Анализирует метаданные входа, выбирает:
    1. Рекомендуемую библиотеку CPU-кодера для "головы" (для записи в temp_head).
    2. Расширение для ФИНАЛЬНОГО выходного файла.
    3. Рекомендуемое действие для аудиопотока "хвоста" при склейке FFmpeg
       ('copy', 'none', или имя кодека для перекодирования, e.g., 'aac').

    Returns:
        Tuple[str, str, str]:
            - final_output_extension (e.g., ".mp4")
            - recommended_head_video_encoder_lib (e.g., "libx264")
            - ffmpeg_tail_audio_action (e.g., "copy", "none", "aac")
    """
    in_video_codec = input_metadata.get('video_codec')  # e.g., 'h264', 'hevc'
    in_audio_codec = input_metadata.get('audio_codec') if input_metadata.get('has_audio') else None
    has_audio_original = input_metadata.get('has_audio', False)

    logging.info(f"Проверка совместимости и выбор параметров выхода:")
    logging.info(f"  Вход: Видео='{in_video_codec}', Аудио='{in_audio_codec}' (есть: {has_audio_original})")

    # --- 1. Выбор рекомендуемого CPU-кодера для "головы" (temp_head.mp4) ---
    # Цель: создать "голову" с высокой совместимостью для последующей склейки.
    # libx264 (H.264) - хороший, широко совместимый выбор.
    # Аудио для головы всегда будет 'aac' (это решается в write_head_only).

    recommended_head_video_encoder_lib = DEFAULT_VIDEO_ENCODER_LIB_FOR_HEAD
    # Можно было бы сделать более сложный выбор, например:
    # if in_video_codec == 'hevc':
    #     recommended_head_video_encoder_lib = 'libx265'
    # elif in_video_codec == 'vp9':
    #     recommended_head_video_encoder_lib = 'libvpx-vp9'
    # Но для простоты и надежности стыковки, кодирование головы в H.264 (libx264)
    # обычно является безопасным вариантом, даже если оригинал был другим.

    logging.info(f"  Рекомендуемый видеокодер для 'головы' (temp_file): {recommended_head_video_encoder_lib}")
    # Аудио для головы будет 'aac', это задается в write_head_only.
    head_audio_codec_name_assumed = 'aac'

    # --- 2. Определение расширения для ФИНАЛЬНОГО файла и действия для аудио "хвоста" (в FFmpeg) ---
    # Это зависит от кодеков ОРИГИНАЛЬНОГО видео (которое станет хвостом)
    # и аудио (голова будет AAC, хвост - оригинальный аудиокодек или перекодированный).

    final_output_extension = DEFAULT_OUTPUT_CONTAINER_EXT_FINAL  # По умолчанию .mp4

    # Определяем, можно ли копировать аудио хвоста, или его нужно будет перекодировать FFmpeg
    ffmpeg_tail_audio_action = DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL  # По умолчанию перекодировать
    if not has_audio_original:
        ffmpeg_tail_audio_action = 'none'  # Нет аудио в оригинале, значит и в хвосте
    elif in_audio_codec:
        # Пытаемся подобрать контейнер, который поддерживает копирование оригинального видео И аудио
        # Если аудио хвоста AAC, MP3, Opus, Vorbis - хорошие кандидаты для копирования в совместимые контейнеры
        if in_audio_codec == 'aac' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.mp4', '.mov', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
        elif in_audio_codec == 'mp3' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.mp4', '.mov', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
        elif in_audio_codec == 'opus' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.webm', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
            final_output_extension = ".webm" if in_video_codec in ['vp8', 'vp9', 'av1'] else ".mkv"
        elif in_audio_codec == 'vorbis' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.webm', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
            final_output_extension = ".webm" if in_video_codec in ['vp8', 'vp9', 'av1'] else ".mkv"
        # Если не один из распространенных "копируемых" кодеков, оставляем ffmpeg_tail_audio_action
        # на DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL ('aac'), что означает перекодирование хвоста.

    # Корректируем final_output_extension на основе видеокодека хвоста, если нужно
    if in_video_codec in ['vp8', 'vp9', 'av1']:
        # Для этих кодеков WebM или MKV предпочтительнее MP4
        if final_output_extension == ".mp4":  # Если не было изменено выше аудио логикой
            final_output_extension = ".mkv"  # MKV более гибкий
            if ffmpeg_tail_audio_action not in ['opus', 'vorbis', 'aac', 'none', 'copy']:
                # Если аудио хвоста будет перекодироваться, и это не Opus/Vorbis, то AAC для MKV
                ffmpeg_tail_audio_action = DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL

    # Финальная проверка совместимости выбранного final_output_extension
    # с видеокодеком хвоста (in_video_codec) и аудиокодеками
    # (голова = 'aac', хвост = in_audio_codec если 'copy', иначе ffmpeg_tail_audio_action)

    final_check_audio_codec = in_audio_codec if ffmpeg_tail_audio_action == 'copy' else ffmpeg_tail_audio_action
    if final_check_audio_codec == 'none': final_check_audio_codec = None  # Для проверки

    allowed_codecs_in_final = CODEC_CONTAINER_COMPATIBILITY.get(final_output_extension, set())

    video_ok_final = (in_video_codec is None) or (('video', in_video_codec) in allowed_codecs_in_final)
    # Аудио головы (всегда 'aac') должно быть совместимо
    audio_head_ok_final = ('audio', head_audio_codec_name_assumed) in allowed_codecs_in_final
    # Аудио хвоста (либо скопированное, либо перекодированное) должно быть совместимо
    audio_tail_ok_final = (final_check_audio_codec is None) or \
                          (('audio', final_check_audio_codec) in allowed_codecs_in_final)

    if not (video_ok_final and audio_head_ok_final and audio_tail_ok_final):
        logging.warning(
            f"  Выбранный финальный контейнер '{final_output_extension}' может быть несовместим с кодеками:")
        logging.warning(f"    Видео хвоста ({in_video_codec or 'N/A'}): {video_ok_final}")
        logging.warning(f"    Аудио головы ({head_audio_codec_name_assumed}): {audio_head_ok_final}")
        logging.warning(f"    Аудио хвоста ({final_check_audio_codec or 'N/A'}): {audio_tail_ok_final}")

        current_problematic_ext = final_output_extension
        final_output_extension = FALLBACK_CONTAINER_EXT_FINAL  # Переключаемся на .mkv
        logging.warning(f"  Переключение на fallback финальный контейнер: '{final_output_extension}'")

        # Перепроверяем совместимость для MKV
        allowed_codecs_in_mkv = CODEC_CONTAINER_COMPATIBILITY.get(".mkv", set())
        video_ok_mkv = (in_video_codec is None) or (('video', in_video_codec) in allowed_codecs_in_mkv)
        audio_head_ok_mkv = ('audio', head_audio_codec_name_assumed) in allowed_codecs_in_mkv
        audio_tail_ok_mkv = (final_check_audio_codec is None) or \
                            (('audio', final_check_audio_codec) in allowed_codecs_in_mkv)

        if not (video_ok_mkv and audio_head_ok_mkv and audio_tail_ok_mkv):
            logging.error(f"  Даже fallback контейнер '{final_output_extension}' несовместим! "
                          f"Возврат к '{current_problematic_ext}'. FFmpeg может потребовать перекодирование или выдать ошибку.")
            final_output_extension = current_problematic_ext
            # В этом случае, если ffmpeg_tail_audio_action был 'copy', лучше принудительно перекодировать
            if ffmpeg_tail_audio_action == 'copy' and has_audio_original:
                logging.warning(
                    f"   Аудио хвоста будет принудительно перекодировано в {DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL} из-за несовместимости контейнера.")
                ffmpeg_tail_audio_action = DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL

    logging.info(f"  Итоговое решение для финального файла: Расширение='{final_output_extension}', "
                 f"Видеокодер 'головы' (для temp файла)='{recommended_head_video_encoder_lib}', "
                 f"Действие для аудио 'хвоста' (FFmpeg)='{ffmpeg_tail_audio_action}'")

    return final_output_extension, recommended_head_video_encoder_lib, ffmpeg_tail_audio_action


# --- Функция чтения видео (остается без изменений) ---
def get_input_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Читает метаданные из видеофайла с использованием PyAV, включая битрейт аудио
    и индексы потоков. Предоставляет fallback на OpenCV для базовых параметров.

    Args:
        video_path: Путь к видеофайлу.

    Returns:
        Словарь с метаданными или None при критической ошибке.
        Ключи словаря включают:
        'input_path', 'format_name', 'duration' (в микросекундах, если есть),
        'video_codec', 'width', 'height', 'fps' (Fraction или float),
        'video_bitrate', 'video_time_base' (Fraction), 'pix_fmt',
        'color_space_tag', 'color_primaries_tag', 'color_transfer_tag', 'color_range_tag',
        'video_stream_index' (int),
        'has_audio' (bool), 'audio_codec', 'audio_rate' (int), 'audio_layout' (str),
        'audio_time_base' (Fraction), 'audio_bitrate' (int, bps, или None),
        'audio_stream_index' (int, или -1),
        'audio_codec_context_params' (dict),
        'total_frames' (int, оценка, может быть неточной).
    """
    if not PYAV_AVAILABLE:  # Глобальный флаг
        logging.error("PyAV недоступен для get_input_metadata.")
        return None

    # Инициализация словаря метаданных
    metadata: Dict[str, Any] = {
        'input_path': video_path,
        'format_name': None,
        'duration': None,  # В микросекундах (AV_TIME_BASE)
        'video_codec': None,
        'width': 0,
        'height': 0,
        'fps': None,  # Будет Fraction или float
        'video_bitrate': None,
        'video_time_base': None,
        'pix_fmt': None,
        'color_space_tag': None,
        'color_primaries_tag': None,
        'color_transfer_tag': None,
        'color_range_tag': None,
        'video_stream_index': -1,
        'has_audio': False,
        'audio_codec': None,
        'audio_rate': None,
        'audio_layout': None,
        'audio_time_base': None,
        'audio_bitrate': None,  # Важное поле
        'audio_stream_index': -1,
        'audio_codec_context_params': None,
        'total_frames': 0  # Оценка
    }
    input_container: Optional[av.container.Container] = None
    opencv_fallback_used = False

    try:
        # --- Попытка открыть с PyAV ---
        logging.info(f"Attempting to open '{video_path}' with PyAV to read metadata...")
        try:
            input_container = av.open(video_path, mode='r')
            if input_container is None:
                raise av.FFmpegError(f"av.open вернул None для '{video_path}'")
            logging.info("Opened with PyAV successfully.")
        except (av.FFmpegError, FileNotFoundError, Exception) as e_open:
            logging.error(f"PyAV не смог открыть '{video_path}' для метаданных: {e_open}", exc_info=True)
            logging.warning("Попытка fallback на OpenCV для базовых метаданных (W, H, FPS)...")
            opencv_fallback_used = True
            cap_cv2 = None
            try:
                cap_cv2 = cv2.VideoCapture(video_path)
                if cap_cv2.isOpened():
                    w_cv2 = int(cap_cv2.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h_cv2 = int(cap_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps_cv2 = cap_cv2.get(cv2.CAP_PROP_FPS)  # Может быть 0
                    if w_cv2 > 0 and h_cv2 > 0:
                        metadata['width'] = w_cv2
                        metadata['height'] = h_cv2
                        logging.info(f"  OpenCV Fallback: Получены размеры {w_cv2}x{h_cv2}")
                    if fps_cv2 and fps_cv2 > 0:
                        metadata['fps'] = float(fps_cv2)  # Сохраняем как float
                        logging.info(f"  OpenCV Fallback: Получен FPS {fps_cv2:.2f}")
                else:
                    logging.error("  OpenCV fallback не удался: Не удалось открыть видео.")
            except Exception as e_cv2_meta:
                logging.error(f"  Ошибка во время OpenCV fallback: {e_cv2_meta}")
            finally:
                if cap_cv2: cap_cv2.release()
            # Если PyAV не открылся, дальше нет смысла, возвращаем что есть (или None)
            if not (metadata['width'] > 0 and metadata['height'] > 0):
                logging.critical("Не удалось получить даже базовые размеры кадра.")
                return None
            return metadata  # Возвращаем то, что удалось получить через OpenCV

        # --- Чтение метаданных контейнера ---
        if input_container.format:
            metadata['format_name'] = input_container.format.name
        if input_container.duration:  # Длительность в микросекундах
            metadata['duration'] = input_container.duration
        if input_container.bit_rate:  # Общий битрейт файла
            # Можно сохранить, но видео/аудио битрейты потоков важнее
            # metadata['container_bitrate'] = input_container.bit_rate
            pass

        # --- Чтение метаданных видеопотока ---
        if not input_container.streams.video:
            logging.warning("Видеопотоки не найдены PyAV.")
            # Проверим, есть ли размеры из OpenCV fallback
            if not (metadata['width'] > 0 and metadata['height'] > 0):
                logging.error("Нет видеопотоков и не удалось получить размеры через OpenCV.")
                return None  # Не можем работать без видео
        else:
            try:
                # Берем первый видеопоток
                video_stream = input_container.streams.video[0]
                metadata['video_stream_index'] = video_stream.index
                ctx = video_stream.codec_context

                metadata['video_codec'] = video_stream.codec.name
                # Используем размеры из контекста, если они валидны, иначе оставляем из OpenCV fallback
                if ctx.width and ctx.width > 0: metadata['width'] = ctx.width
                if ctx.height and ctx.height > 0: metadata['height'] = ctx.height

                # Получение FPS: приоритет average_rate, затем r_frame_rate, затем из OpenCV
                fps_val = None
                if video_stream.average_rate and float(video_stream.average_rate) > 0:
                    fps_val = video_stream.average_rate  # Это Fraction
                elif video_stream.r_frame_rate and float(video_stream.r_frame_rate) > 0:
                    fps_val = video_stream.r_frame_rate  # Это Fraction
                if fps_val: metadata['fps'] = fps_val
                # Если fps все еще None, оставляем значение из OpenCV fallback (если было)

                metadata['pix_fmt'] = ctx.pix_fmt
                if ctx.bit_rate and ctx.bit_rate > 0:  # Битрейт видеопотока
                    metadata['video_bitrate'] = ctx.bit_rate
                elif input_container.bit_rate and input_container.bit_rate > 0 and not input_container.streams.audio:
                    # Если аудио нет, общий битрейт = видео битрейт
                    metadata['video_bitrate'] = input_container.bit_rate

                if video_stream.time_base: metadata['video_time_base'] = video_stream.time_base

                # Цветовые теги
                metadata['color_space_tag'] = video_stream.metadata.get('color_space')
                metadata['color_primaries_tag'] = video_stream.metadata.get('color_primaries')
                metadata['color_transfer_tag'] = video_stream.metadata.get('color_transfer')
                metadata['color_range_tag'] = video_stream.metadata.get('color_range')

                # Оценка общего числа кадров (может быть неточной)
                if video_stream.frames and video_stream.frames > 0:
                    metadata['total_frames'] = video_stream.frames
                elif metadata['duration'] and metadata['fps'] and float(metadata['fps']) > 0:
                    # Рассчитываем из длительности и FPS, если stream.frames нет
                    try:
                        metadata['total_frames'] = int(
                            round((float(metadata['duration']) / 1_000_000.0) * float(metadata['fps'])))
                    except Exception:
                        pass  # Ошибка расчета

                logging.info(
                    f"  PyAV Video Stream Meta: Codec={metadata['video_codec']}, Res={metadata['width']}x{metadata['height']}, "
                    f"FPS={float(metadata['fps']):.2f if metadata['fps'] else 'N/A'}, Frames={metadata['total_frames'] or 'N/A'}, "
                    f"Index={metadata['video_stream_index']}")

            except (AttributeError, ValueError, TypeError, av.FFmpegError) as e_video:
                logging.error(f"Ошибка при доступе к свойствам видеопотока: {e_video}")
                # Проверяем критичные параметры еще раз
                if not (metadata['width'] > 0 and metadata['height'] > 0 and metadata['fps']):
                    logging.critical("Критичные видео метаданные отсутствуют после всех проверок.")
                    return None  # Не можем продолжать без W, H, FPS

        # --- Чтение метаданных аудиопотока ---
        if not input_container.streams.audio:
            logging.info("Аудиопотоки не найдены PyAV.")
            metadata['has_audio'] = False
            metadata['audio_stream_index'] = -1
            metadata['audio_bitrate'] = None
        else:
            try:
                # Берем первый аудиопоток
                audio_stream = input_container.streams.audio[0]
                metadata['audio_stream_index'] = audio_stream.index
                ctx = audio_stream.codec_context

                metadata['has_audio'] = True
                metadata['audio_codec'] = audio_stream.codec.name
                metadata['audio_rate'] = ctx.rate
                metadata['audio_layout'] = ctx.layout.name if ctx.layout else None
                if audio_stream.time_base: metadata['audio_time_base'] = audio_stream.time_base

                # --- Получение аудио битрейта ---
                audio_bitrate = ctx.bit_rate  # int или None
                if audio_bitrate and audio_bitrate > 0:
                    metadata['audio_bitrate'] = audio_bitrate
                else:
                    metadata['audio_bitrate'] = None  # Явно None
                    # Можно попытаться рассчитать, если знаем размер потока и длительность, но это ненадежно

                # Сохраняем другие параметры контекста, если они могут понадобиться
                # (например, для настройки декодера в write_head_only)
                metadata['audio_codec_context_params'] = {
                    'format': ctx.format.name if ctx.format else None,
                    'layout': metadata['audio_layout'],
                    'rate': metadata['audio_rate'],
                    'bit_rate': metadata['audio_bitrate'],
                    'codec_tag': ctx.codec_tag,
                    'extradata': bytes(ctx.extradata) if ctx.extradata else None,
                }
                logging.info(
                    f"  PyAV Audio Stream Meta: Codec={metadata['audio_codec']}, Rate={metadata['audio_rate']}, "
                    f"Layout={metadata['audio_layout']}, Bitrate={metadata['audio_bitrate'] or 'N/A'}, "
                    f"Index={metadata['audio_stream_index']}")

            except (AttributeError, ValueError, TypeError, av.FFmpegError) as e_audio:
                logging.warning(f"Не удалось получить/обработать метаданные аудиопотока: {e_audio}")
                metadata['has_audio'] = False
                metadata['audio_stream_index'] = -1
                metadata['audio_bitrate'] = None  # Сбрасываем при ошибке

        # Финальная проверка критичных видео параметров
        if not (metadata['width'] > 0 and metadata['height'] > 0 and metadata['fps']):
            logging.critical("Критичные видео метаданные (W, H, FPS) отсутствуют.")
            return None

        return metadata

    except Exception as e_general:  # Ловим другие неожиданные ошибки
        logging.error(f"Неожиданная ошибка при получении метаданных для '{video_path}': {e_general}", exc_info=True)
        return None  # Возвращаем None при серьезной ошибке
    finally:
        if input_container:
            try:
                input_container.close()
                logging.debug("Metadata reading: Input container closed.")
            except av.FFmpegError as e_close:
                logging.error(f"Error closing input container after metadata read: {e_close}")


# --- Новая функция для чтения ТОЛЬКО нужных кадров и ВСЕХ аудиопакетов ---
@profile  # Если используете line_profiler
def read_processing_head(
        video_path: str,
        frames_to_read: int,
        video_stream_index: int,  # Индекс видеопотока для чтения
        audio_stream_index: int  # Индекс аудиопотока для чтения (-1 если нет аудио)
) -> Tuple[Optional[List[np.ndarray]], Optional[List[av.Packet]]]:
    """
    Читает и декодирует ТОЛЬКО первые `frames_to_read` видеокадров (в BGR NumPy)
    и собирает ВСЕ аудиопакеты из указанных потоков.

    Args:
        video_path: Путь к входному видеофайлу.
        frames_to_read: Количество видеокадров, которые нужно прочитать и декодировать.
        video_stream_index: Индекс видеопотока в контейнере.
        audio_stream_index: Индекс аудиопотока в контейнере (-1, если аудио не нужно/нет).

    Returns:
        Кортеж (list_of_bgr_frames, list_of_audio_packets).
        list_of_bgr_frames: Список NumPy массивов (кадры в BGR).
        list_of_audio_packets: Список всех av.Packet аудиопотока.
        Возвращает (None, None) при критической ошибке.
    """
    if not PYAV_AVAILABLE:  # PYAV_AVAILABLE должно быть определено глобально
        logging.error("PyAV недоступен для read_processing_head.")
        return None, None

    if frames_to_read <= 0:
        logging.warning(
            f"read_processing_head: Количество кадров для чтения ({frames_to_read}) <= 0. Возвращаем пустые списки.")
        return [], []  # Возвращаем пустые списки, это не ошибка, просто нечего читать

    head_frames_bgr: List[np.ndarray] = []
    all_audio_packets: List[av.Packet] = []
    input_container: Optional[av.container.Container] = None
    frames_decoded_count = 0
    reformatter_yuv_to_bgr: Optional[VideoReformatter] = None

    logging.info(f"Чтение 'головы': {frames_to_read} видеокадров и все аудиопакеты из '{video_path}'...")
    logging.debug(f"  Целевой видеопоток: индекс {video_stream_index}, аудиопоток: индекс {audio_stream_index}")

    try:
        input_container = av.open(video_path, mode='r')
        if input_container is None:
            # Это может случиться, если av.open вернул None вместо исключения
            raise av.FFmpegError(f"av.open вернул None для файла: {video_path}")

        # Проверяем наличие нужных потоков
        has_target_video_stream = any(
            s.index == video_stream_index and s.type == 'video' for s in input_container.streams)
        has_target_audio_stream = audio_stream_index != -1 and any(
            s.index == audio_stream_index and s.type == 'audio' for s in input_container.streams)

        if not has_target_video_stream and frames_to_read > 0:
            logging.error(f"  Видеопоток с индексом {video_stream_index} не найден в '{video_path}'.")
            # Если кадры нужны, но потока нет - это проблема
            return None, None  # или ([], all_audio_packets) если аудио важнее
        if audio_stream_index != -1 and not has_target_audio_stream:
            logging.warning(f"  Аудиопоток с индексом {audio_stream_index} не найден. Аудиопакеты не будут собраны.")
            # Не прерываем, если видео еще можно прочитать

        logging.debug("Начало демультиплексирования пакетов...")
        packet_count = 0
        for packet in input_container.demux():
            packet_count += 1
            if packet.dts is None:  # Пропускаем flush-пакеты или пакеты без DTS
                logging.debug(f"  Пакет {packet_count}: пропущен (нет DTS). Stream index: {packet.stream.index}")
                continue

            # Собираем ВСЕ аудиопакеты нужного потока
            if has_target_audio_stream and packet.stream.index == audio_stream_index:
                try:
                    packet_data = bytes(packet)
                    new_packet = av.Packet(packet_data)

                    # Копируем важные атрибуты вручную
                    new_packet.pts = packet.pts
                    new_packet.dts = packet.dts
                    new_packet.duration = packet.duration
                    # --- УДАЛЕНО: new_packet.stream_index = packet.stream_index ---
                    # new_packet.is_keyframe = packet.is_keyframe # Если нужно

                    # Сохраняем оригинальный stream_index, если он понадобится позже для идентификации
                    # (например, если у вас несколько аудиопотоков и вы хотите их различать)
                    # Но для самого new_packet это поле не пишется напрямую.
                    # Можно сохранить его в отдельной структуре или как временный атрибут, если PyAV это позволяет
                    # setattr(new_packet, '_original_stream_index', packet.stream.index) # Нестандартно, может не работать

                    all_audio_packets.append(new_packet)
                except Exception as e_packet_create:
                    logging.error(f"  Ошибка при создании копии аудиопакета: {e_packet_create}", exc_info=True)

                if len(all_audio_packets) % 200 == 0:
                    logging.debug(f"  Собрано {len(all_audio_packets)} аудиопакетов...")

            # Декодируем видеопакеты, пока не наберем нужное количество кадров
            elif has_target_video_stream and packet.stream.index == video_stream_index:
                if frames_decoded_count >= frames_to_read:
                    # Видеокадры головы уже набраны, но продолжаем собирать аудио, если нужно
                    continue

                try:
                    for frame in packet.decode():  # packet.decode() может вернуть несколько кадров
                        if frame and isinstance(frame, av.VideoFrame):
                            # Конвертируем в BGR NumPy
                            try:
                                # Попытка прямого преобразования в BGR24
                                np_frame_bgr = frame.to_ndarray(format='bgr24')
                            except (av.FFmpegError, ValueError, TypeError) as e_to_ndarray:
                                logging.warning(
                                    f"    Не удалось напрямую конвертировать видеокадр в bgr24: {e_to_ndarray}. Попытка через YUV420p.")
                                # Попытка реформатирования в YUV420P, затем в NumPy, затем OpenCV
                                try:
                                    # Инициализируем реформаттер один раз
                                    if reformatter_yuv_to_bgr is None:
                                        # Убедимся, что frame.width и frame.height валидны
                                        if frame.width <= 0 or frame.height <= 0:
                                            logging.error(
                                                f"    Невалидные размеры кадра для реформаттера: {frame.width}x{frame.height}")
                                            continue  # Пропускаем этот кадр
                                        reformatter_yuv_to_bgr = VideoReformatter(frame.width, frame.height, 'yuv420p')

                                    frame_yuv = reformatter_yuv_to_bgr.reformat(frame)
                                    np_frame_yuv = frame_yuv.to_ndarray()  # Это будет массив YUV (planes)

                                    # Конвертация YUV (скорее всего I420/YUV420P) в BGR
                                    if np_frame_yuv.shape[
                                        0] * 2 // 3 == frame_yuv.height:  # Проверка типичной структуры YUV420P
                                        np_frame_bgr = cv2.cvtColor(np_frame_yuv, cv2.COLOR_YUV2BGR_I420)
                                    else:  # Фоллбек на другой формат YUV, если известен, или ошибка
                                        logging.error(
                                            f"    Неизвестный формат NumPy массива после YUV реформатирования: {np_frame_yuv.shape}")
                                        continue  # Пропускаем этот кадр
                                except Exception as e_reformat_cv:
                                    logging.error(
                                        f"    Ошибка при реформатировании в YUV или конвертации OpenCV: {e_reformat_cv}",
                                        exc_info=True)
                                    continue  # Пропускаем этот кадр

                            head_frames_bgr.append(np_frame_bgr)
                            frames_decoded_count += 1

                            if frames_decoded_count % 50 == 0:  # Логируем каждые 50 кадров
                                logging.debug(
                                    f"  Декодировано {frames_decoded_count}/{frames_to_read} видеокадров 'головы'...")

                            if frames_decoded_count >= frames_to_read:
                                break  # Выходим из внутреннего цикла по кадрам в пакете

                    if frames_decoded_count >= frames_to_read:
                        # Видеокадры головы набраны, но продолжаем собирать аудио, если оно еще не все
                        pass  # Цикл по пакетам продолжится

                except (av.FFmpegError, ValueError) as e_decode_video:
                    # Некоторые ошибки декодирования могут быть некритичны (например, для поврежденных кадров)
                    logging.warning(
                        f"  Ошибка декодирования видеопакета (stream {packet.stream.index}): {e_decode_video} - пакет пропущен.")
                except Exception as e_unexpected_decode:
                    logging.error(f"  Неожиданная ошибка при декодировании видеопакета: {e_unexpected_decode}",
                                  exc_info=True)
                    # Решаем, стоит ли прерывать при неожиданной ошибке
                    # return None, None # Вариант: прервать

            if frames_to_read > 0 and has_target_video_stream and frames_decoded_count >= frames_to_read and \
                    (audio_stream_index == -1 or not has_target_audio_stream):
                # Если видео набрано и аудио не нужно/нет, можно выйти раньше
                logging.info(
                    "  Все необходимые видеокадры 'головы' прочитаны, аудио не требуется/нет. Завершение демультиплексирования.")
                break

        # Цикл по пакетам завершен
        if frames_to_read > 0 and frames_decoded_count < frames_to_read:
            logging.warning(
                f"  Прочитано только {frames_decoded_count} видеокадров 'головы', запрашивалось {frames_to_read} (возможно, конец файла).")

    except FFmpegEOFError:
        logging.info("  Достигнут конец файла (EOF) при чтении 'головы'.")
    except av.FFmpegError as e_av:
        logging.error(f"Ошибка PyAV/FFmpeg при чтении 'головы' из '{video_path}': {e_av}", exc_info=True)
        return None, None  # Критическая ошибка
    except FileNotFoundError:
        logging.error(f"Файл не найден при попытке открыть для чтения 'головы': '{video_path}'")
        return None, None
    except Exception as e_main:
        logging.error(f"Неожиданная ошибка при чтении 'головы' из '{video_path}': {e_main}", exc_info=True)
        return None, None
    finally:
        if input_container:
            try:
                input_container.close()
                logging.debug("  Контейнер входного файла для 'головы' закрыт.")
            except av.FFmpegError as e_close:
                logging.error(f"  Ошибка при закрытии контейнера входного файла 'головы': {e_close}")

    logging.info(
        f"Чтение 'головы' завершено. Декодировано {len(head_frames_bgr)} видеокадров. Собрано {len(all_audio_packets)} аудиопакетов.")
    return head_frames_bgr, all_audio_packets


# --- Функция записи видео (остается без изменений) ---
# @profile # Добавьте, если нужно профилировать


def rescale_time(value: Optional[int], old_tb: Optional[Fraction], new_tb: Optional[Fraction], label: str = "") -> Optional[int]:
    """
    Пересчитывает значение времени (PTS, DTS, Duration) из одной time_base в другую.
    Возвращает None при ошибке или если входные данные некорректны.
    """
    if value is None or old_tb is None or new_tb is None \
       or not isinstance(old_tb, Fraction) or not isinstance(new_tb, Fraction) \
       or old_tb.denominator == 0 or new_tb.denominator == 0:
         return None
    try:
         scaled_value = Fraction(value * old_tb.numerator * new_tb.denominator, old_tb.denominator * new_tb.numerator)
         if scaled_value >= 0: result = int(scaled_value + Fraction(1, 2))
         else: result = int(scaled_value - Fraction(1, 2))
         return result
    except (ZeroDivisionError, OverflowError, TypeError) as e:
         logging.warning(f"Rescale warning ({label}): value={value}, old={old_tb}, new={new_tb}. Error: {e}")
         return None

# --- Основная функция записи "Голова + Хвост" ---
def get_assumed_color_properties(width: int, height: int,
                                 original_tags: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Определяет предполагаемые/стандартные цветовые свойства на основе разрешения
    и имеющихся оригинальных тегов.
    Возвращает словарь с ключами 'colorspace', 'primaries', 'trc', 'range'.
    Значения - строки, понятные FFmpeg/PyAV.
    """
    cs_tag = original_tags.get('color_space')  # Пример ключей, используйте ваши из get_input_metadata
    cp_tag = original_tags.get('color_primaries')
    ct_tag = original_tags.get('color_transfer')
    cr_tag = original_tags.get('color_range')

    # Значения по умолчанию (для HD)
    assumed_cs = 'bt709'
    assumed_cp = 'bt709'
    assumed_ct = 'bt709'
    assumed_cr = 'tv'  # Limited range ('mpeg' в некоторых API)

    # Логика на основе разрешения (упрощенный пример, нужно адаптировать)
    if height <= 576:  # Пример для SD
        assumed_cs = 'bt470bg' if cs_tag and '601' not in cs_tag else 'bt470bg'  # или smpte170m
        assumed_cp = 'bt470bg' if cp_tag and '601' not in cp_tag else 'bt470bg'  # или smpte170m
        assumed_ct = 'gamma28' if ct_tag and '601' not in ct_tag else 'gamma28'  # или smpte170m
        assumed_cr = cr_tag if cr_tag in ['tv', 'pc', 'mpeg', 'jpeg'] else 'tv'
    elif height > 1080:  # Пример для UHD
        # Оставим BT.709 для совместимости, если не указано иное
        if cs_tag == 'bt2020ncl': assumed_cs = 'bt2020ncl'
        if cp_tag == 'bt2020': assumed_cp = 'bt2020'
        # Для HDR нужны специфичные transfer functions (e.g., smpte2084/pq, arib-std-b67/hlg)
        if ct_tag and ('bt2020' in ct_tag or 'pq' in ct_tag or 'hlg' in ct_tag):
            assumed_ct = ct_tag
        # Для SDR UHD часто используют тот же bt709/gamma2.4 или iec61966-2-1/srgb
        # elif ct_tag and ('iec61966' in ct_tag or 'srgb' in ct_tag) : assumed_ct = ct_tag
        else:
            assumed_ct = 'bt709'  # Fallback к bt709 для SDR UHD

        if cr_tag == 'pc': assumed_cr = 'pc'  # Если диапазон полный, сохраняем
    else:  # HD
        # Проверяем, если исходные теги похожи на bt709, используем их
        if cs_tag and '709' in cs_tag: assumed_cs = cs_tag
        if cp_tag and '709' in cp_tag: assumed_cp = cp_tag
        if ct_tag and '709' in ct_tag: assumed_ct = ct_tag
        if cr_tag == 'pc': assumed_cr = 'pc'

    # Финальный словарь свойств
    final_props = {
        'colorspace': assumed_cs,
        'primaries': assumed_cp,
        'trc': assumed_ct,
        'range': assumed_cr
    }
    logging.debug(f"Определены цветовые свойства для {width}x{height}: {final_props} (исходные: {original_tags})")
    return final_props


# --- НОВАЯ ФУНКЦИЯ: Запись "Головы" во Временный Файл ---
# @profile # Если используете line_profiler
def write_head_only(
        watermarked_head_frames: List[np.ndarray],
        all_audio_packets: Optional[List[av.Packet]],
        input_metadata: Dict[str, Any],
        temp_head_path: str,
        target_video_encoder_lib: str,
        audio_action_for_head: str,
        video_encoder_options: Optional[Dict[str, str]] = None,
        audio_encoder_options: Optional[Dict[str, str]] = None
) -> Optional[float]:
    if not PYAV_AVAILABLE:
        logging.error("PyAV недоступен для write_head_only.")
        return None
    if not watermarked_head_frames:
        logging.warning("write_head_only: Нет кадров для записи в 'голову'.")
        return 0.0

    num_head_frames = len(watermarked_head_frames)
    logging.info(f"Запись 'головы' в '{temp_head_path}' ({num_head_frames} кадров)...")
    logging.info(f"  Видеокодер: {target_video_encoder_lib}, Аудио для головы: {audio_action_for_head}")

    output_container: Optional['av.container.Container'] = None
    video_stream: Optional['av.stream.Stream'] = None
    audio_stream: Optional['av.stream.Stream'] = None
    input_audio_decoder: Optional['av.codec.context.CodecContext'] = None
    reformatter: Optional[VideoReformatter] = None

    calculated_video_duration_sec: Optional[float] = None
    last_encoded_video_pts = -1
    video_frame_duration_tb = 0  # Длительность видеокадра в единицах time_base

    # Глобальная константа FPS (пример, должна быть определена в вашем коде)
    FPS = 30.0

    try:
        width = input_metadata.get('width')
        height = input_metadata.get('height')
        fps_meta = input_metadata.get('fps')

        if not (width and height and width > 0 and height > 0):
            logging.error(f"Некорректные размеры кадра: {width}x{height}");
            return None

        fps_to_use = float(FPS)
        if fps_meta and isinstance(fps_meta, (float, int, Fraction)) and float(fps_meta) > 0:
            fps_to_use = fps_meta
        else:
            logging.warning(f"Некорректный FPS ({fps_meta}). Используется fallback FPS={fps_to_use}.")
        fps_fraction = Fraction(fps_to_use).limit_denominator()

        has_audio_original = input_metadata.get('has_audio', False)
        process_audio = has_audio_original and audio_action_for_head == 'aac' and all_audio_packets is not None

        output_container = av.open(temp_head_path, mode='w')
        if output_container is None: raise av.FFmpegError("Не удалось открыть выходной контейнер")

        original_color_tags = {key: input_metadata.get(f'{key}_tag') for key in
                               ['color_space', 'color_primaries', 'color_transfer', 'color_range']}
        color_props_for_output = get_assumed_color_properties(width, height, original_color_tags)

        output_pix_fmt = 'yuv420p'
        video_stream = output_container.add_stream(target_video_encoder_lib, rate=fps_fraction)
        video_stream.width = width;
        video_stream.height = height;
        video_stream.pix_fmt = output_pix_fmt

        codec_options = video_encoder_options.copy() if video_encoder_options else {}
        if color_props_for_output.get('colorspace'): codec_options['colorspace'] = color_props_for_output['colorspace']
        if color_props_for_output.get('primaries'): codec_options['color_primaries'] = color_props_for_output[
            'primaries']
        if color_props_for_output.get('trc'): codec_options['color_trc'] = color_props_for_output['trc']
        if color_props_for_output.get('range'): codec_options['color_range'] = color_props_for_output['range']
        if codec_options: video_stream.codec_context.options = codec_options; logging.debug(
            f"Опции видеокодека: {codec_options}")

        input_video_time_base = input_metadata.get('video_time_base')
        if input_video_time_base and isinstance(input_video_time_base, Fraction):
            video_stream.time_base = input_video_time_base
            logging.info(f"Видеопоток time_base из оригинала: {video_stream.time_base}")
        else:
            if video_stream.time_base is None: video_stream.time_base = Fraction(1, 90000)
            logging.info(f"Видеопоток time_base (auto/fallback): {video_stream.time_base}")

        if video_stream.time_base and float(fps_fraction) > 0:
            try:
                video_frame_duration_tb = int(round((1.0 / float(fps_fraction)) / float(video_stream.time_base)))
            except:
                pass
        if video_frame_duration_tb <= 0: video_frame_duration_tb = 1; logging.warning(
            "Не удалось рассчитать video_frame_duration_tb, используется 1.")
        logging.debug(f"Расчетная длительность видео кадра (в time_base): {video_frame_duration_tb}")

        decoded_audio_frames_for_head: List[av.AudioFrame] = []
        head_duration_sec_estimated = num_head_frames / float(fps_fraction) if float(fps_fraction) > 0 else 0.0

        if process_audio:
            input_audio_codec_name = input_metadata.get('audio_codec')
            input_audio_rate = input_metadata.get('audio_rate')
            input_audio_layout_str = input_metadata.get('audio_layout')
            input_audio_time_base = input_metadata.get('audio_time_base')

            if not (input_audio_codec_name and input_audio_rate and input_audio_layout_str and input_audio_time_base):
                logging.warning("Аудио: Недостаточно метаданных для обработки.");
                process_audio = False
            else:
                logging.info(f"Аудио: Перекодирование в AAC, Rate={input_audio_rate}, Layout={input_audio_layout_str}")
                try:
                    audio_stream = output_container.add_stream('aac', rate=input_audio_rate)
                    audio_stream.codec_context.layout = input_audio_layout_str
                    default_audio_opts = {'b:a': '128k'};
                    final_audio_opts = default_audio_opts.copy()
                    if audio_encoder_options: final_audio_opts.update(audio_encoder_options)
                    audio_stream.codec_context.options = final_audio_opts;
                    logging.debug(f"Опции AAC: {final_audio_opts}")
                    if audio_stream.time_base is None: audio_stream.time_base = Fraction(1, input_audio_rate)
                    logging.info(f"Аудиопоток time_base: {audio_stream.time_base}")

                    # --- НАЧАЛО ПОЛНОЙ ЛОГИКИ ДЕКОДИРОВАНИЯ АУДИО ---
                    input_audio_decoder = av.Codec(input_audio_codec_name, 'r').create()  # type: ignore
                    if input_audio_decoder is None: raise av.FFmpegError("Не удалось создать аудио декодер.")

                    in_audio_ctx_params = input_metadata.get('audio_codec_context_params')
                    if in_audio_ctx_params:
                        if in_audio_ctx_params.get('format'): input_audio_decoder.format = in_audio_ctx_params['format']
                        if in_audio_ctx_params.get('layout'): input_audio_decoder.layout = in_audio_ctx_params['layout']
                        if in_audio_ctx_params.get('rate'): input_audio_decoder.sample_rate = in_audio_ctx_params[
                            'rate']
                        if in_audio_ctx_params.get('extradata'): input_audio_decoder.extradata = in_audio_ctx_params[
                            'extradata']
                    # Если какие-то параметры не установились из ctx_params, берем из основных метаданных
                    if input_audio_decoder.layout is None and input_audio_layout_str: input_audio_decoder.layout = input_audio_layout_str
                    if input_audio_decoder.sample_rate is None and input_audio_rate: input_audio_decoder.sample_rate = input_audio_rate

                    logging.debug(
                        f"Декодирование аудиопакетов (оценка длины головы ~{head_duration_sec_estimated:.3f}s)")
                    processed_packet_count = 0
                    for audio_packet_from_list in all_audio_packets or []:
                        processed_packet_count += 1
                        if audio_packet_from_list.dts is None: continue  # Пропускаем flush пакеты

                        packet_time_sec = 0.0
                        if audio_packet_from_list.pts is not None and input_audio_time_base:
                            try:
                                packet_time_sec = float(audio_packet_from_list.pts * input_audio_time_base)
                            except Exception:
                                pass  # Ошибка при расчете времени

                        # Прерываем декодирование, если время пакета вышло за ОЦЕНОЧНУЮ длит. головы + буфер
                        if packet_time_sec > head_duration_sec_estimated + 1.0:  # +1 секунда буфера
                            logging.debug(
                                f"Аудиопакет {processed_packet_count} PTS {audio_packet_from_list.pts} ({packet_time_sec:.3f}s) за пределами ОЦЕНОЧНОЙ длины головы. Прерывание декодирования.")
                            break
                        try:
                            frames = input_audio_decoder.decode(audio_packet_from_list)
                            if frames: decoded_audio_frames_for_head.extend(frames)
                        except av.FFmpegError as e_decode:
                            if e_decode.errno == -11 or 'again' in str(e_decode).lower():  # EAGAIN
                                logging.debug(
                                    f"Ошибка декодирования аудиопакета {processed_packet_count} (EAGAIN), требуется больше данных.")
                                continue
                            logging.warning(f"Ошибка декодирования аудиопакета {processed_packet_count}: {e_decode}")
                    try:  # Flush декодера
                        frames = input_audio_decoder.decode(None)
                        if frames: decoded_audio_frames_for_head.extend(frames)
                    except av.FFmpegError:
                        pass  # Ошибки при flush часто нормальны
                    logging.info(f"Декодировано {len(decoded_audio_frames_for_head)} аудиокадров.")
                    # --- КОНЕЦ ПОЛНОЙ ЛОГИКИ ДЕКОДИРОВАНИЯ АУДИО ---
                except Exception as e_setup_audio:
                    logging.error(f"Ошибка при настройке/декодировании аудио: {e_setup_audio}", exc_info=True)
                    process_audio = False;
                    audio_stream = None

        logging.info(f"Кодирование и мультиплексирование {num_head_frames} видеокадров...")
        encoded_video_frame_count = 0
        current_video_pts = 0

        for frame_idx, bgr_frame_np in enumerate(watermarked_head_frames):
            try:
                if not isinstance(bgr_frame_np, np.ndarray) or bgr_frame_np.shape[:2] != (height, width):
                    logging.warning(f"Пропуск некорректного видеокадра {frame_idx}");
                    continue
                video_frame_in = VideoFrame.from_ndarray(bgr_frame_np, format='bgr24')
                if reformatter is None:
                    reformatter = VideoReformatter(
                        video_frame_in.width, video_frame_in.height, output_pix_fmt,
                        src_format='bgr24', src_colorspace='srgb',
                        dst_colorspace=color_props_for_output.get('colorspace'),
                        dst_primaries=color_props_for_output.get('primaries'),
                        dst_trc=color_props_for_output.get('trc'),
                        dst_color_range=color_props_for_output.get('range'))
                video_frame_out = reformatter.reformat(video_frame_in)
                video_frame_out.pts = current_video_pts

                encoded_video_packets = video_stream.encode(video_frame_out)  # type: ignore
                if encoded_video_packets is not None:  # encode может вернуть пустой список
                    last_encoded_video_pts = current_video_pts
                    current_video_pts += video_frame_duration_tb
                    encoded_video_frame_count += 1
                    for packet in encoded_video_packets: output_container.mux(packet)  # type: ignore
            except Exception as e_vf:
                logging.error(f"Ошибка при обработке/кодировании видеокадра {frame_idx}: {e_vf}", exc_info=True);
                raise

        logging.debug("Flush видеокодера...");
        encoded_video_packets = video_stream.encode(None)  # type: ignore
        if encoded_video_packets: output_container.mux(encoded_video_packets)  # type: ignore
        logging.info(f"Закодировано и записано {encoded_video_frame_count} видеокадров.")

        if video_stream and last_encoded_video_pts >= 0 and video_frame_duration_tb > 0 and video_stream.time_base:
            try:
                last_effective_pts = last_encoded_video_pts + video_frame_duration_tb
                duration_val = float(last_effective_pts * video_stream.time_base)
                if duration_val >= 0:
                    calculated_video_duration_sec = duration_val
                else:
                    logging.error("Рассчитана отрицательная видео длительность!")
            except Exception as e_calc_vid:
                logging.error(f"Ошибка расчета видео длительности: {e_calc_vid}")
        elif encoded_video_frame_count == 0 and num_head_frames > 0:
            calculated_video_duration_sec = None
        else:
            calculated_video_duration_sec = 0.0

        if calculated_video_duration_sec is not None:
            logging.info(f"Финальная расчетная ВИДЕО длительность: {calculated_video_duration_sec:.6f}s")
        else:
            logging.error("Не удалось рассчитать финальную видео длительность.")

        if process_audio and audio_stream and decoded_audio_frames_for_head:
            logging.info(
                f"Кодирование и мультиплексирование аудиокадров (до ~{calculated_video_duration_sec or head_duration_sec_estimated:.3f}s)...")
            encoded_audio_frame_count = 0
            duration_limit_for_audio = calculated_video_duration_sec if (
                        calculated_video_duration_sec is not None and calculated_video_duration_sec > 0) else head_duration_sec_estimated
            for audio_frame_idx, audio_frame in enumerate(decoded_audio_frames_for_head):
                try:
                    audio_pts_sec = 0.0
                    if audio_frame.pts is not None and audio_stream.time_base:
                        try:
                            audio_pts_sec = float(audio_frame.pts * audio_stream.time_base)
                        except:
                            pass
                    if audio_pts_sec > duration_limit_for_audio + 0.01:  # Небольшой буфер
                        logging.debug(f"Пропуск аудиокадра {audio_frame_idx} (PTS={audio_pts_sec:.3f}s > {duration_limit_for_audio:.3f}s)")
                        continue
                    encoded_audio_packets = audio_stream.encode(audio_frame)
                    if encoded_audio_packets is not None:
                        encoded_audio_frame_count += 1
                        for packet in encoded_audio_packets: output_container.mux(packet)  # type: ignore
                except Exception as e_af:
                    logging.warning(f"Ошибка кодирования/мультиплексирования аудиокадра {audio_frame_idx}: {e_af}")
            logging.debug("Flush аудиокодера...");
            encoded_audio_packets = audio_stream.encode(None)  # type: ignore
            if encoded_audio_packets: output_container.mux(encoded_audio_packets)  # type: ignore
            logging.info(f"Закодировано и записано {encoded_audio_frame_count} аудиокадров.")

        logging.info(f"Запись 'головы' в '{temp_head_path}' завершена.")
        return calculated_video_duration_sec

    except av.FFmpegError as e:
        logging.error(f"Ошибка PyAV/FFmpeg при записи 'головы': {e}", exc_info=True); return None
    except Exception as e:
        logging.error(f"Неожиданная ошибка при записи 'головы': {e}", exc_info=True); return None
    finally:
        if output_container:  # type: ignore
            try:
                output_container.close()  # type: ignore
            except av.FFmpegError as e_close:
                logging.error(f"Ошибка при закрытии контейнера 'головы': {e_close}")

# --- НОВАЯ ФУНКЦИЯ: Склейка с "Хвостом" через FFmpeg ---
def concatenate_with_ffmpeg(
        original_input_path: str,
        temp_head_path: str,
        final_output_path: str,
        head_duration_sec: float,
        input_metadata: Dict[str, Any],
        original_audio_bitrate: Optional[int] = None
) -> bool:
    logging.info(
        f"Склейка FFmpeg (2-этапный, хвост trim/re-encode, concat с видео-copy): '{temp_head_path}' + хвост из '{original_input_path}' (с {head_duration_sec:.3f}s) -> '{final_output_path}'")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logging.error("Ошибка: FFmpeg не найден.")
        return False

    pid = os.getpid()
    output_extension = os.path.splitext(final_output_path)[1]
    temp_tail_path = temp_head_path.replace("_head" + output_extension, f"_tail_pid{pid}" + output_extension)
    list_file_path = None  # Инициализируем

    # Расчет длительности оригинала
    original_duration_av = input_metadata.get('duration')
    total_original_duration_sec = 0.0
    if original_duration_av and isinstance(original_duration_av, (int, float)) and original_duration_av > 0:
        total_original_duration_sec = float(original_duration_av) / 1_000_000.0
    elif input_metadata.get('total_frames_estimated', -1) > 0 and isinstance(input_metadata.get('fps'),
                                                                             (int, float)) and input_metadata[
        'fps'] > 0:
        total_original_duration_sec = float(input_metadata['total_frames_estimated']) / float(input_metadata['fps'])

    # Этап 0: Проверка, нужен ли хвост
    if total_original_duration_sec > 0 and head_duration_sec >= total_original_duration_sec - 0.01:
        logging.warning(f"Длительность головы ({head_duration_sec:.3f}s) покрывает оригинал. Хвост не нужен.")
        try:
            if os.path.exists(final_output_path): os.remove(final_output_path)
            try:
                os.rename(temp_head_path, final_output_path)
            except OSError:
                shutil.copy2(temp_head_path, final_output_path)
            return True
        except Exception as e_rn_cp:
            logging.error(f"Ошибка переименования/копирования файла головы: {e_rn_cp}")
            return False

    # --- Этап 1: Создание временного файла хвоста с ПЕРЕКОДИРОВАНИЕМ через фильтры ---
    logging.info(f"Этап 1: Создание '{temp_tail_path}' (с фильтрами trim/setpts и перекодированием)...")

    target_audio_bitrate_str_for_tail = str(
        original_audio_bitrate) if original_audio_bitrate and original_audio_bitrate >= 32000 else '128k'

    # Логика для -ss и trim
    # Используем -ss перед -i для быстрого поиска до точки, немного предшествующей trim_start.
    # trim_start_target_abs - это head_duration_sec
    # seek_to_time_abs - немного раньше, чтобы у trim был "запас" для точного старта
    seek_to_time_abs = max(0.0, head_duration_sec - 10.0)  # Например, за 10 секунд до
    trim_start_relative_to_seek = head_duration_sec - seek_to_time_abs  # Начало для trim относительно seek_to_time_abs

    cmd_create_tail_prefix = [ffmpeg_path, '-y']
    # Добавляем -ss только если есть смысл искать (не с самого начала)
    if seek_to_time_abs > 0.01:  # Небольшой порог, чтобы не ставить -ss 0
        cmd_create_tail_prefix.extend(['-ss', f"{seek_to_time_abs:.6f}"])
    cmd_create_tail_prefix.extend(['-i', original_input_path])

    cmd_create_tail = cmd_create_tail_prefix + [
        '-vf', f"trim=start={trim_start_relative_to_seek:.6f},setpts=PTS-STARTPTS",
        '-af', f"atrim=start={trim_start_relative_to_seek:.6f},asetpts=PTS-STARTPTS",
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p',
        '-force_key_frames', "expr:eq(n,0)",
        '-c:a', 'aac', '-b:a', target_audio_bitrate_str_for_tail,
        '-map_metadata', '-1',
        temp_tail_path
    ]
    logging.debug(f"Команда создания хвоста (фильтры, перекодирование): {' '.join(cmd_create_tail)}")

    tail_created_successfully = False
    try:
        result_tail = subprocess.run(cmd_create_tail, check=False, capture_output=True, text=True, encoding='utf-8')
        if result_tail.returncode != 0:
            logging.error(
                f"Ошибка FFmpeg при создании хвоста (код {result_tail.returncode}).\nStderr: {result_tail.stderr}")
        elif not os.path.exists(temp_tail_path) or os.path.getsize(temp_tail_path) < 100:
            logging.error(f"Файл хвоста '{temp_tail_path}' не создан или пуст.\nStderr: {result_tail.stderr}")
        else:
            logging.info(f"Этап 1: Временный файл хвоста '{temp_tail_path}' успешно создан (перекодирован).")
            tail_created_successfully = True
    except Exception as e_tail:
        logging.error(f"Неожиданная ошибка при создании хвоста: {e_tail}", exc_info=True)

    if not tail_created_successfully:
        if os.path.exists(temp_tail_path):
            try:
                os.remove(temp_tail_path)
            except OSError:
                pass
        return False

    # --- Этап 2: Склейка головы и хвоста через concat демультиплексор ---
    # Видео копируется (т.к. хвост уже "чистый"), аудио перекодируется для идеального стыка.
    logging.info("Этап 2: Склейка головы и хвоста (concat, видео-copy, аудио-re-encode)...")
    success_concat = False
    try:
        abs_temp_head_path = os.path.abspath(temp_head_path).replace('\\', '/')
        abs_temp_tail_path = os.path.abspath(temp_tail_path).replace('\\', '/')
        list_file_content = (
            f"file '{abs_temp_head_path}'\n"
            f"file '{abs_temp_tail_path}'\n"  # Хвост теперь "чистый" и начинается с 0
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as list_file:
            list_file_path = list_file.name
            list_file.write(list_file_content)
        logging.debug(f"Создан временный файл списка для concat: '{list_file_path}'")

        # Определяем целевой аудио битрейт для финальной склейки
        final_target_audio_bitrate_str = str(
            original_audio_bitrate) if original_audio_bitrate and original_audio_bitrate >= 32000 else '192k'
        logging.info(f"Аудио битрейт для финальной склейки (concat): {final_target_audio_bitrate_str}")

        cmd_concat = [
            ffmpeg_path,
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file_path,
            '-c:v', 'copy',  # Копировать видео (оба сегмента теперь "чистые")
            '-c:a', 'aac',  # Перекодировать аудио для идеального слияния
            '-b:a', final_target_audio_bitrate_str,
            '-avoid_negative_ts', 'make_zero',  # На всякий случай, для финального потока
            '-movflags', '+faststart',
            final_output_path
        ]
        logging.debug(f"Команда склейки (concat, видео-copy, аудио-re-encode): {' '.join(cmd_concat)}")

        result_concat = subprocess.run(cmd_concat, check=False, capture_output=True, text=True, encoding='utf-8')

        if result_concat.returncode == 0:
            logging.info("FFmpeg склейка (concat) успешно завершена.")
            if result_concat.stderr: logging.debug(f"FFmpeg stderr (concat):\n{result_concat.stderr}")
            if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 100:
                success_concat = True
            else:
                logging.error(
                    f"FFmpeg (concat) завершился успешно, но выходной файл '{final_output_path}' не создан или пуст.\nStderr: {result_concat.stderr}")
        else:
            logging.error(
                f"Ошибка выполнения FFmpeg (concat) (код {result_concat.returncode}).\nStderr: {result_concat.stderr}")

    except Exception as e_concat:
        logging.error(f"Неожиданная ошибка при склейке concat: {e_concat}", exc_info=True)
    finally:
        if list_file_path and os.path.exists(list_file_path):
            try:
                os.remove(list_file_path)
            except OSError:
                pass  # Ошибку удаления временного файла можно проигнорировать

        if os.path.exists(temp_tail_path):
            try:
                os.remove(temp_tail_path)
            except OSError:
                pass

    return success_concat


@profile
def embed_frame_pair(
        frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bits: List[int],
        selected_ring_indices: List[int], n_rings: int, frame_number: int,
        use_perceptual_masking: bool, embed_component: int,
        # --- Аргументы PyTorch ---
        device: torch.device,
        dtcwt_fwd: 'DTCWTForward',
        dtcwt_inv: 'DTCWTInverse'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Встраивает биты (PyTorch DCT/SVD, улучшенная модификация, ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ).
    Полная версия с ИСПРАВЛЕННЫМ применением дельты.
    """
    pair_index = frame_number // 2
    prefix_base = f"[P:{pair_index}]"

    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
         logging.error(f"{prefix_base} Отсутствуют PyTorch Wavelets или DCT!")
         return None, None

    min_len_bits_rings = min(len(bits), len(selected_ring_indices))
    if len(bits) != len(selected_ring_indices):
        logging.warning(f"{prefix_base} Mismatch bits/rings: {len(bits)} vs {len(selected_ring_indices)}. Using min len {min_len_bits_rings}.")
    if min_len_bits_rings == 0:
        logging.debug(f"{prefix_base} No bits/rings to process.")
        return frame1_bgr, frame2_bgr
    bits_to_embed = bits[:min_len_bits_rings]
    rings_to_process = selected_ring_indices[:min_len_bits_rings]

    logging.debug(f"{prefix_base} --- Starting Embedding Pair (PyTorch v2 Fixed Delta Apply) for {len(bits_to_embed)} bits ---")
    try:
        # 1. Преобразование в Тензоры
        f1_ycrcb_np = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb)
        f2_ycrcb_np = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2YCrCb)
        comp1_np = f1_ycrcb_np[:, :, embed_component].copy().astype(np.float32) / 255.0
        comp2_np = f2_ycrcb_np[:, :, embed_component].copy().astype(np.float32) / 255.0
        comp1_tensor = torch.from_numpy(comp1_np).to(device=device)
        comp2_tensor = torch.from_numpy(comp2_np).to(device=device)
        target_shape_hw = (frame1_bgr.shape[0], frame1_bgr.shape[1])

        # 2. Прямое DTCWT
        Yl_t, Yh_t = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, frame_number)
        Yl_t1, Yh_t1 = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, frame_number + 1)
        if Yl_t is None or Yh_t is None or Yl_t1 is None or Yh_t1 is None: return None, None
        if Yl_t.dim() > 2: Yl_t = Yl_t.squeeze(0).squeeze(0)
        if Yl_t1.dim() > 2: Yl_t1 = Yl_t1.squeeze(0).squeeze(0)

        # 3. Подготовка к модификации
        ring_coords_t = ring_division(Yl_t, n_rings, frame_number)
        ring_coords_t1 = ring_division(Yl_t1, n_rings, frame_number + 1)
        if ring_coords_t is None or ring_coords_t1 is None: return None, None
        perceptual_mask_tensor = calculate_perceptual_mask(comp1_tensor, device, frame_number)
        if perceptual_mask_tensor is None: perceptual_mask_tensor = torch.ones_like(Yl_t, device=device)
        if perceptual_mask_tensor.shape != Yl_t.shape:
             try: perceptual_mask_tensor = F.interpolate(...) # ваш код интерполяции
             except Exception as e_interp: logging.error(...); perceptual_mask_tensor = torch.ones_like(Yl_t, device=device)

        modifications_count = 0
        Yl_t_mod = Yl_t.clone(); Yl_t1_mod = Yl_t1.clone()

        # --- Цикл по кольцам ---
        logging.debug(f"{prefix_base} --- Start Ring Loop (Embedding {len(bits_to_embed)} bits) ---")
        for ring_idx, bit_to_embed in zip(rings_to_process, bits_to_embed):
            prefix = f"[P:{pair_index} R:{ring_idx}]"
            logging.debug(f"{prefix} ------- Processing bit {bit_to_embed} -------")

            if not (0 <= ring_idx < n_rings and ring_idx < len(ring_coords_t) and ring_idx < len(ring_coords_t1)): continue
            coords1_tensor = ring_coords_t[ring_idx]; coords2_tensor = ring_coords_t1[ring_idx]
            if coords1_tensor is None or coords2_tensor is None or coords1_tensor.shape[0] < 10 or coords2_tensor.shape[0] < 10: continue

            try:
                rows1, cols1 = coords1_tensor[:, 0], coords1_tensor[:, 1]
                rows2, cols2 = coords2_tensor[:, 0], coords2_tensor[:, 1]
                v1_tensor = Yl_t_mod[rows1, cols1].float()
                v2_tensor = Yl_t1_mod[rows2, cols2].float()
                min_s = min(v1_tensor.numel(), v2_tensor.numel())
                if min_s == 0: continue
                if v1_tensor.numel() != v2_tensor.numel():
                    v1_tensor = v1_tensor[:min_s]; v2_tensor = v2_tensor[:min_s]
                    rows1, cols1 = rows1[:min_s], cols1[:min_s]; rows2, cols2 = rows2[:min_s], cols2[:min_s]

                # logging.debug(f"{prefix} v1 stats...") # Логи можно оставить
                # logging.debug(f"{prefix} v2 stats...")

                alpha_float = compute_adaptive_alpha_entropy(v1_tensor.cpu().numpy(), ring_idx, frame_number)
                alpha_t = torch.tensor(alpha_float, device=device, dtype=v1_tensor.dtype)
                inv_a = 1.0 / (alpha_t + 1e-12)
                # logging.debug(f"{prefix} Adaptive alpha...")

                d1_tensor = dct1d_torch(v1_tensor)
                d2_tensor = dct1d_torch(v2_tensor)
                if not torch.isfinite(d1_tensor).all() or not torch.isfinite(d2_tensor).all(): continue
                # logging.debug(f"{prefix} DCT done...")

                U1, S1_vec, Vh1 = torch.linalg.svd(d1_tensor.unsqueeze(-1), full_matrices=False)
                U2, S2_vec, Vh2 = torch.linalg.svd(d2_tensor.unsqueeze(-1), full_matrices=False)
                if U1 is None or S1_vec is None or Vh1 is None or U2 is None or S2_vec is None or Vh2 is None: continue
                if S1_vec.numel() == 0 or S2_vec.numel() == 0: continue
                s1 = S1_vec[0]; s2 = S2_vec[0]
                if not torch.isfinite(s1) or not torch.isfinite(s2): continue
                # logging.debug(f"{prefix} SVD done...")

                eps = torch.tensor(1e-12, device=device, dtype=s1.dtype)
                s2_safe = s2 + eps if torch.abs(s2) < eps else s2
                original_ratio = s1 / s2_safe
                # logging.debug(f"{prefix} Original Ratio...")

                ns1, ns2 = s1.clone(), s2.clone()
                modified = False
                action = "No change needed"
                current_bit = 0 if original_ratio >= 1.0 else 1
                modify_needed = (current_bit != bit_to_embed)
                strengthen_needed = False
                target_ratio = original_ratio

                if not modify_needed:
                    if bit_to_embed == 0 and original_ratio < alpha_t: strengthen_needed = True; action = "Strengthening bit 0"
                    elif bit_to_embed == 1 and original_ratio >= inv_a: strengthen_needed = True; action = "Strengthening bit 1"
                else: action = f"Modifying {current_bit}->{bit_to_embed}"

                if modify_needed or strengthen_needed:
                    modified = True
                    energy = torch.sqrt(s1**2 + s2**2); energy = energy + eps if energy < eps else energy
                    if bit_to_embed == 0: target_ratio = alpha_t
                    else: target_ratio = inv_a
                    denominator = torch.sqrt(target_ratio**2 + 1.0 + eps)
                    ns1 = energy * target_ratio / denominator
                    ns2 = energy / denominator
                    if not torch.isfinite(ns1) or not torch.isfinite(ns2): modified = False

                logging.info(f"{prefix} ACTION: {action}.")

                if modified:
                    modifications_count += 1
                    # logging.debug(f"{prefix}    Target Ratio...") # Логи можно оставить
                    # logging.debug(f"{prefix}    New Ratio Check...")
                    # logging.debug(f"{prefix}    New s1..., s2...")

                    ns1_diag = torch.diag(ns1.unsqueeze(0))
                    ns2_diag = torch.diag(ns2.unsqueeze(0))
                    d1m_tensor = torch.matmul(U1, torch.matmul(ns1_diag, Vh1)).squeeze(-1)
                    d2m_tensor = torch.matmul(U2, torch.matmul(ns2_diag, Vh2)).squeeze(-1)
                    v1m_tensor = idct1d_torch(d1m_tensor)
                    v2m_tensor = idct1d_torch(d2m_tensor)

                    if not torch.isfinite(v1m_tensor).all() or not torch.isfinite(v2m_tensor).all(): continue
                    if v1m_tensor.shape != v1_tensor.shape or v2m_tensor.shape != v2_tensor.shape: continue

                    delta1_pt = v1m_tensor - v1_tensor # Используем _pt для ясности
                    delta2_pt = v2m_tensor - v2_tensor
                    logging.debug(f"{prefix} PT Delta1 Stats: mean={delta1_pt.mean():.6e}, std={delta1_pt.std():.6e}")
                    logging.debug(f"{prefix} PT Delta2 Stats: mean={delta2_pt.mean():.6e}, std={delta2_pt.std():.6e}")

                    # --- ЛОГ ДО ПРИМЕНЕНИЯ ДЕЛЬТЫ ---
                    yl_sub_before = Yl_t_mod[rows1, cols1]
                    logging.debug(f"{prefix} Yl_t_mod[ring] BEFORE apply: mean={yl_sub_before.mean():.6e}, std={yl_sub_before.std():.6e}")
                    yl_sub2_before = Yl_t1_mod[rows2, cols2]
                    logging.debug(f"{prefix} Yl_t1_mod[ring] BEFORE apply: mean={yl_sub2_before.mean():.6e}, std={yl_sub2_before.std():.6e}")
                    # ----------------------------------

                    mf1 = torch.ones_like(delta1_pt); mf2 = torch.ones_like(delta2_pt)
                    if use_perceptual_masking and perceptual_mask_tensor is not None:
                        try:
                            mv1 = perceptual_mask_tensor[rows1, cols1]; mv2 = perceptual_mask_tensor[rows2, cols2]
                            lambda_t = torch.tensor(LAMBDA_PARAM, device=device, dtype=mf1.dtype); one_minus_lambda_t = 1.0 - lambda_t
                            mf1.mul_(lambda_t + one_minus_lambda_t * mv1); mf2.mul_(lambda_t + one_minus_lambda_t * mv2)
                            # logging.debug(f"{prefix} Mask factors applied...")
                        except Exception as mask_err: logging.warning(f"{prefix} Mask apply error: {mask_err}")

                    # --- ИСПРАВЛЕНИЕ: Явное присваивание ---
                    Yl_t_mod[rows1, cols1] = yl_sub_before + delta1_pt * mf1 # Используем значения ДО, а не текущие
                    Yl_t1_mod[rows2, cols2] = yl_sub2_before + delta2_pt * mf2
                    # -------------------------------------

                    # --- ЛОГ ПОСЛЕ ПРИМЕНЕНИЯ ДЕЛЬТЫ ---
                    yl_sub_after = Yl_t_mod[rows1, cols1] # Читаем снова после присваивания
                    logging.debug(f"{prefix} Yl_t_mod[ring] AFTER apply: mean={yl_sub_after.mean():.6e}, std={yl_sub_after.std():.6e}")
                    yl_sub2_after = Yl_t1_mod[rows2, cols2]
                    logging.debug(f"{prefix} Yl_t1_mod[ring] AFTER apply: mean={yl_sub2_after.mean():.6e}, std={yl_sub2_after.std():.6e}")
                    # -----------------------------------
                    logging.debug(f"{prefix} Deltas applied to Yl_mod tensors using assignment.")

            except Exception as e:
                logging.error(f"{prefix} Error in ring loop: {e}", exc_info=True); continue
            logging.debug(f"{prefix} ------- Finished Processing Ring -------")
        # --- Конец цикла по кольцам ---
        logging.debug(f"{prefix_base} --- End Ring Loop ({modifications_count} modifications) ---")

        # 5. Обратное DTCWT
        if Yl_t_mod.dim() == 2: Yl_t_mod = Yl_t_mod.unsqueeze(0).unsqueeze(0)
        if Yl_t1_mod.dim() == 2: Yl_t1_mod = Yl_t1_mod.unsqueeze(0).unsqueeze(0)
        c1m_np = dtcwt_pytorch_inverse(Yl_t_mod, Yh_t, dtcwt_inv, device, target_shape_hw, frame_number)
        c2m_np = dtcwt_pytorch_inverse(Yl_t1_mod, Yh_t1, dtcwt_inv, device, target_shape_hw, frame_number + 1)
        if c1m_np is None or c2m_np is None: return None, None

        # --- ЛОГ СТАТИСТИКИ РЕКОНСТРУКЦИИ ПЕРЕД КВАНТОВАНИЕМ ---
        logging.debug(f"{prefix_base} Reconstructed c1m_np stats: mean={np.mean(c1m_np):.6e}, std={np.std(c1m_np):.6e}, min={np.min(c1m_np):.6e}, max={np.max(c1m_np):.6e}")
        logging.debug(f"{prefix_base} Reconstructed c2m_np stats: mean={np.mean(c2m_np):.6e}, std={np.std(c2m_np):.6e}, min={np.min(c2m_np):.6e}, max={np.max(c2m_np):.6e}")
        logging.debug(f"{prefix_base} Original comp1_np stats: mean={np.mean(comp1_np):.6e}, std={np.std(comp1_np):.6e}")
        # ------------------------------------------------------

        # 6. Постобработка и сборка кадра
        c1s_np = np.clip(c1m_np * 255.0, 0, 255).astype(np.uint8)
        c2s_np = np.clip(c2m_np * 255.0, 0, 255).astype(np.uint8)
        if c1s_np.shape != target_shape_hw: c1s_np = cv2.resize(c1s_np, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        if c2s_np.shape != target_shape_hw: c2s_np = cv2.resize(c2s_np, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        f1_ycrcb_out_np = f1_ycrcb_np.copy(); f2_ycrcb_out_np = f2_ycrcb_np.copy()
        f1_ycrcb_out_np[:, :, embed_component] = c1s_np; f2_ycrcb_out_np[:, :, embed_component] = c2s_np
        f1m = cv2.cvtColor(f1_ycrcb_out_np, cv2.COLOR_YCrCb2BGR); f2m = cv2.cvtColor(f2_ycrcb_out_np, cv2.COLOR_YCrCb2BGR)

        logging.debug(f"{prefix_base} Embed Pair Finished Successfully.")
        return f1m, f2m

    except Exception as e:
        logging.error(f"{prefix_base} Critical error in embed_frame_pair: {e}", exc_info=True)
        if 'device' in locals() and device.type == 'cuda':
            with torch.no_grad(): torch.cuda.empty_cache()
        return None, None
# --- ИЗМЕНЕННЫЙ _embed_single_pair_task ---
# @profile
def _embed_single_pair_task(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """
    Обрабатывает одну пару кадров: выбирает кольца, вызывает embed_frame_pair (PyTorch).
    """
    pair_idx = args.get('pair_idx', -1); f1_bgr = args.get('frame1'); f2_bgr = args.get('frame2')
    bits_for_this_pair = args.get('bits', [])
    nr = args.get('n_rings', N_RINGS); nrtu = args.get('nu  m_rings_to_use', NUM_RINGS_TO_USE)
    cps = args.get('candidate_pool_size', CANDIDATE_POOL_SIZE); ec = args.get('embed_component', EMBED_COMPONENT)
    upm = args.get('use_perceptual_masking', USE_PERCEPTUAL_MASKING)
    # Получаем объекты PyTorch из аргументов
    device = args.get('device'); dtcwt_fwd = args.get('dtcwt_fwd'); dtcwt_inv = args.get('dtcwt_inv')
    fn = 2 * pair_idx; selected_rings = []

    if pair_idx == -1 or f1_bgr is None or f2_bgr is None or not bits_for_this_pair \
       or device is None or dtcwt_fwd is None or dtcwt_inv is None:
        logging.error(f"Missing args or data for _embed_single_pair_task (P:{pair_idx})")
        return fn, None, None, []
    if not PYTORCH_WAVELETS_AVAILABLE:
        logging.error(f"PyTorch Wavelets not available in _embed_single_pair_task (P:{pair_idx})")
        return fn, None, None, []

    try:
        # --- ШАГ 1: Выбор колец ---
        # (Логика остается прежней, но использует PyTorch DTCWT и ring_division)
        candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
        if len(candidate_rings) < nrtu: # Используем фактическое nrtu
            logging.warning(f"[P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{nrtu}. Using all.")
            # Не меняем nrtu здесь, пусть используется меньше колец, если надо
            if len(candidate_rings) == 0: raise ValueError("No candidates found.")
        #else: # Лог не обязателен
        #   logging.debug(f"[P:{pair_idx}] Candidates: {candidate_rings}")

        # Конвертируем кадр для выбора колец
        f1_ycrcb_np = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
        comp1_tensor = torch.from_numpy(f1_ycrcb_np[:, :, ec].copy()).to(device=device, dtype=torch.float32) / 255.0

        # Вычисляем DTCWT только для Yl
        Yl_t_select, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
        if Yl_t_select is None: raise RuntimeError(f"DTCWT FWD failed P:{pair_idx}")
        if Yl_t_select.dim() > 2: Yl_t_select = Yl_t_select.squeeze()

        # Вычисляем координаты колец
        coords = ring_division(Yl_t_select, nr, fn) # PyTorch версия ring_division
        if coords is None or len(coords) != nr: raise RuntimeError(f"Ring division failed P:{pair_idx}")

        # Выбор по энтропии
        entropies = []; min_pixels_for_entropy = 10
        for r_idx in candidate_rings:
            entropy_val = -float('inf')
            if 0 <= r_idx < len(coords) and coords[r_idx] is not None and coords[r_idx].shape[0] >= min_pixels_for_entropy:
                 c_tensor = coords[r_idx]
                 try:
                       rows, cols = c_tensor[:, 0], c_tensor[:, 1]
                       rv_tensor = Yl_t_select[rows, cols] # Извлекаем из тензора
                       rv_np = rv_tensor.cpu().numpy() # Конвертируем в NumPy для calculate_entropies
                       shannon_entropy, _ = calculate_entropies(rv_np, fn, r_idx)
                       if np.isfinite(shannon_entropy): entropy_val = shannon_entropy
                 except Exception as e: logging.warning(f"[P:{pair_idx},R:{r_idx}] Entropy calc error: {e}")
            entropies.append((entropy_val, r_idx))

        entropies.sort(key=lambda x: x[0], reverse=True)
        selected_rings = [idx for e, idx in entropies if e > -float('inf')][:nrtu]

        if len(selected_rings) < nrtu: # Fallback
            logging.warning(f"[P:{pair_idx}] Fallback ring selection ({len(selected_rings)}<{nrtu}).")
            det_fallback = candidate_rings[:nrtu]
            for ring in det_fallback:
                if ring not in selected_rings: selected_rings.append(ring)
                if len(selected_rings) == nrtu: break
            if len(selected_rings) < nrtu: raise RuntimeError(f"Fallback failed P:{pair_idx}")
            logging.warning(f"[P:{pair_idx}] Selected rings after fallback: {selected_rings}")
        # logging.info(f"[P:{pair_idx}] Selected rings: {selected_rings}") # Можно раскомментировать для отладки

        # --- ШАГ 2: Вызов встраивания ---
        # Обрезаем биты, если колец выбрано меньше, чем nrtu (из-за fallback или ошибок)
        bits_to_embed_now = bits_for_this_pair[:len(selected_rings)]
        if not bits_to_embed_now:
             logging.warning(f"P:{pair_idx} No bits to embed after ring selection/trimming.")
             return fn, f1_bgr, f2_bgr, selected_rings # Возвращаем исходные, если нет бит

        # Передаем все нужные объекты
        mod_f1, mod_f2 = embed_frame_pair(
            f1_bgr, f2_bgr, bits_to_embed_now, selected_rings, nr, fn, upm, ec,
            device, dtcwt_fwd, dtcwt_inv # Передаем объекты PyTorch
        )

        return fn, mod_f1, mod_f2, selected_rings

    except Exception as e:
        logging.error(f"Error in _embed_single_pair_task P:{pair_idx}: {e}", exc_info=True)
        return fn, None, None, [] # Возвращаем пустой список колец при ошибке



# --- _embed_batch_worker - остается БЕЗ ИЗМЕНЕНИЙ (вызывает _embed_single_pair_task) ---
def _embed_batch_worker(batch_args_list: List[Dict]) -> List[
    Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]]:
    batch_results = []
    for args in batch_args_list:
        result = _embed_single_pair_task(args)
        batch_results.append(result)
    return batch_results


@profile
def embed_watermark_in_video(
        frames_to_process: List[np.ndarray], # <--- Принимаем только кадры "головы"
        payload_id_bytes: bytes,
        n_rings: int = N_RINGS,
        num_rings_to_use: int = NUM_RINGS_TO_USE,
        bits_per_pair: int = BITS_PER_PAIR,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        # --- Параметры гибридного режима ---
        use_hybrid_ecc: bool = True,
        max_total_packets: int = 15,
        use_ecc_for_first: bool = USE_ECC,
        bch_code: Optional[BCH_TYPE] = BCH_CODE_OBJECT,
        # --- Параметры PyTorch ---
        device: torch.device = torch.device("cpu"),
        dtcwt_fwd: Optional[DTCWTForward] = None,
        dtcwt_inv: Optional[DTCWTInverse] = None,
        # --- Другие параметры ---
        max_workers: Optional[int] = MAX_WORKERS, # <--- Принимаем безопасное значение
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT
        # Параметр fps убран, так как расчет пар идет от len(frames_to_process)
    ) -> Optional[List[np.ndarray]]: # <--- Возвращаем обработанные кадры "головы"
    """
    Основная функция встраивания, адаптированная для подхода "Голова + Хвост".
    Обрабатывает ТОЛЬКО переданные кадры (frames_to_process) параллельно
    с ограниченным числом воркеров.
    """
    # Проверка наличия PyTorch объектов
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        logging.critical("PyTorch Wavelets или Torch DCT недоступны!")
        return None # Возвращаем None при критической ошибке
    if dtcwt_fwd is None or dtcwt_inv is None:
        logging.critical("Экземпляры DTCWTForward/DTCWTInverse не переданы!")
        return None

    num_frames_head = len(frames_to_process)
    # Расчет пар идет ТОЛЬКО для переданных кадров "головы"
    total_pairs_head = num_frames_head // 2
    payload_len_bytes = len(payload_id_bytes)
    payload_len_bits = payload_len_bytes * 8

    # Проверки входных данных
    if payload_len_bits == 0:
        logging.error("Payload ID пустой! Встраивание отменено.")
        return None
    if total_pairs_head == 0:
        logging.warning("Нет пар кадров для обработки в переданном списке.")
        # Возвращаем исходный (пустой?) список или None? Лучше None.
        return None
    if max_total_packets <= 0:
         logging.warning(f"max_total_packets ({max_total_packets}) должен быть > 0. Установлено в 1.")
         max_total_packets = 1

    logging.info(f"--- Embed Head Start (Processing {num_frames_head} frames / {total_pairs_head} pairs) ---")
    logging.info(f"Using max_workers: {max_workers}")

    # --- Формирование последовательности бит для встраивания (как раньше) ---
    bits_to_embed_list = []
    raw_payload_bits: Optional[np.ndarray] = None
    try:
        raw_payload_bits = np.unpackbits(np.frombuffer(payload_id_bytes, dtype=np.uint8))
        if raw_payload_bits.size != payload_len_bits:
             raise ValueError(f"Ошибка unpackbits: {payload_len_bits} vs {raw_payload_bits.size}")
    except Exception as e:
        logging.error(f"Ошибка подготовки raw_payload_bits: {e}", exc_info=True)
        return None
    if raw_payload_bits is None: logging.error("Не создан raw_payload_bits."); return None

    first_packet_bits: Optional[np.ndarray] = None
    packet1_type_str = "N/A"; packet1_len = 0; num_raw_packets_added = 0
    can_use_ecc = use_ecc_for_first and GALOIS_AVAILABLE and bch_code is not None and payload_len_bits <= bch_code.k

    if use_hybrid_ecc and can_use_ecc:
        first_packet_bits = add_ecc(raw_payload_bits, bch_code)
        if first_packet_bits is not None:
            bits_to_embed_list.extend(first_packet_bits.tolist())
            packet1_len = len(first_packet_bits); packet1_type_str = f"ECC(n={packet1_len}, t={bch_code.t})"
            logging.info(f"Гибридный режим: Первый пакет создан как {packet1_type_str}.")
        else: logging.error("Ошибка создания ECC пакета!"); return None
    else:
        first_packet_bits = raw_payload_bits; bits_to_embed_list.extend(first_packet_bits.tolist())
        packet1_len = len(first_packet_bits); packet1_type_str = f"Raw({packet1_len})"
        if not use_hybrid_ecc: logging.info(f"Режим НЕ гибридный: Первый пакет - {packet1_type_str}.")
        elif not can_use_ecc: logging.info(f"Гибридный режим: ECC невозможен/выключен. Первый пакет - {packet1_type_str}.")
        use_hybrid_ecc = False # Отключаем добавление Raw пакетов, если первый Raw

    if use_hybrid_ecc:
        num_raw_repeats_to_add = max(0, max_total_packets - 1)
        for _ in range(num_raw_repeats_to_add):
            bits_to_embed_list.extend(raw_payload_bits.tolist())
            num_raw_packets_added += 1
        if num_raw_packets_added > 0: logging.info(f"Гибридный режим: Добавлено {num_raw_packets_added} Raw пакетов.")

    total_packets_actual = 1 + num_raw_packets_added
    total_bits_to_embed = len(bits_to_embed_list)
    if total_bits_to_embed == 0: logging.error("Нет бит для встраивания."); return None

    # --- Определяем, сколько пар нужно обработать (ИЗ ПЕРЕДАННЫХ) ---
    if bits_per_pair <= 0: logging.error(f"Invalid bits_per_pair: {bits_per_pair}"); return None
    pairs_needed = ceil(total_bits_to_embed / bits_per_pair)
    # Обрабатываем не больше пар, чем есть в переданных кадрах "головы"
    pairs_to_process = min(total_pairs_head, pairs_needed)

    # Создаем финальный массив бит нужной длины для обработки
    bits_flat_final = np.array(bits_to_embed_list[:pairs_to_process * bits_per_pair], dtype=np.uint8)
    actual_bits_embedded = len(bits_flat_final)

    logging.info(f"Head processing details:")
    logging.info(f"  Target packets: {total_packets_actual} ({packet1_type_str} + {num_raw_packets_added} Raw)")
    logging.info(f"  Total bits prepared: {total_bits_to_embed}")
    logging.info(f"  Available pairs in head: {total_pairs_head}, Pairs needed: {pairs_needed}, Pairs to process: {pairs_to_process}")
    logging.info(f"  Actual bits to embed in head: {actual_bits_embedded}")
    if pairs_to_process < pairs_needed:
         logging.warning(f"Not enough frames in the provided head ({num_frames_head}) to embed all prepared bits!")

    # --- Подготовка аргументов для батчей ---
    start_time_embed_loop = time.time()
    # Создаем копию ТОЛЬКО переданных кадров "головы"
    watermarked_frames = [frame.copy() for frame in frames_to_process] # Используем list comprehension + copy
    rings_log: Dict[int, List[int]] = {}
    pc, ec, uc = 0, 0, 0
    skipped_pairs = 0
    all_pairs_args = []

    for pair_idx in range(pairs_to_process): # Итерируем только по нужным парам
        i1 = 2 * pair_idx
        i2 = i1 + 1
        # Проверка валидности кадров ВНУТРИ frames_to_process
        # (хотя они должны быть валидны, если read_processing_head отработал)
        if i2 >= len(frames_to_process) or frames_to_process[i1] is None or frames_to_process[i2] is None:
            skipped_pairs += 1
            logging.warning(f"Skipping pair {pair_idx}: invalid frames/indices within the head list.")
            continue

        sbi = pair_idx * bits_per_pair
        ebi = sbi + bits_per_pair
        if sbi >= len(bits_flat_final): break # Выходим, если биты кончились
        if ebi > len(bits_flat_final): ebi = len(bits_flat_final)
        cb = bits_flat_final[sbi:ebi].tolist()
        if len(cb) == 0: continue

        args = {'pair_idx': pair_idx,
                'frame1': frames_to_process[i1], # Берем из переданного списка
                'frame2': frames_to_process[i2], # Берем из переданного списка
                'bits': cb,
                'n_rings': n_rings, 'num_rings_to_use': num_rings_to_use,
                'candidate_pool_size': candidate_pool_size,
                'frame_number': i1, # Индекс кадра относительно начала "головы"
                'use_perceptual_masking': use_perceptual_masking,
                'embed_component': embed_component,
                'device': device, 'dtcwt_fwd': dtcwt_fwd, 'dtcwt_inv': dtcwt_inv}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0: logging.warning(f"Skipped {skipped_pairs} pairs during task prep.")
    if num_valid_tasks == 0: logging.error("No valid tasks to process."); return None

    # --- Запуск ThreadPoolExecutor с ОГРАНИЧЕННЫМ числом воркеров ---
    # Используем переданный max_workers (который должен быть безопасным)
    num_workers_to_use = max_workers if max_workers is not None and max_workers > 0 else 1 # Минимум 1
    # Адаптируем размер батча
    batch_size = max(1, ceil(num_valid_tasks / num_workers_to_use));
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i:i + batch_size] for i in range(0, num_valid_tasks, batch_size) if all_pairs_args[i:i+batch_size]]
    actual_num_batches = len(batched_args_list)

    logging.info(f"Launching {actual_num_batches} batches ({num_valid_tasks} pairs) in ThreadPool (max_workers={num_workers_to_use}, batch≈{batch_size})...")

    try:
        with ThreadPoolExecutor(max_workers=num_workers_to_use) as executor: # Используем num_workers_to_use
            future_to_batch_idx = {executor.submit(_embed_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                original_batch = batched_args_list[batch_idx]
                try:
                    batch_results = future.result()
                    if not isinstance(batch_results, list) or len(batch_results) != len(original_batch):
                        logging.error(f"Batch {batch_idx} result size mismatch!")
                        ec += len(original_batch); continue

                    for i, single_res in enumerate(batch_results):
                        original_args = original_batch[i]
                        pair_idx = original_args.get('pair_idx', -1)
                        if pair_idx == -1: logging.error(f"No pair_idx in result?"); ec += 1; continue

                        if isinstance(single_res, tuple) and len(single_res) == 4:
                            fn_res, mod_f1, mod_f2, sel_rings = single_res
                            i1 = 2 * pair_idx; i2 = i1 + 1
                            if isinstance(sel_rings, list): rings_log[pair_idx] = sel_rings
                            if isinstance(mod_f1, np.ndarray) and isinstance(mod_f2, np.ndarray):
                                # Обновляем кадры ВНУТРИ списка watermarked_frames (который является копией "головы")
                                if i1 < len(watermarked_frames): watermarked_frames[i1] = mod_f1; uc += 1
                                else: logging.error(f"Index {i1} out of bounds for watermarked_frames (len={len(watermarked_frames)})")
                                if i2 < len(watermarked_frames): watermarked_frames[i2] = mod_f2; uc += 1
                                else: logging.error(f"Index {i2} out of bounds for watermarked_frames (len={len(watermarked_frames)})")
                                pc += 1
                            else: logging.warning(f"Embedding failed for pair {pair_idx}."); ec += 1
                        else: logging.warning(f"Incorrect result structure for pair {pair_idx}."); ec += 1
                except Exception as e:
                    failed_pairs_count = len(original_batch)
                    logging.error(f"Batch {batch_idx} execution failed: {e}", exc_info=True)
                    ec += failed_pairs_count
    except Exception as e:
        logging.critical(f"ThreadPoolExecutor critical error: {e}", exc_info=True)
        return None # Возвращаем None при ошибке пула

    # --- Завершение и запись логов колец ---
    processing_time = time.time() - start_time_embed_loop
    logging.info(f"Processing {pairs_to_process} pairs finished in {processing_time:.2f} sec.")
    logging.info(f"Result: Processed OK: {pc}, Errors/Skipped: {ec + skipped_pairs}, Frames Updated: {uc}.")

    # Запись лога колец (без изменений)
    if rings_log:
        try:
            serializable_log = {str(k): v for k, v in rings_log.items()}
            with open(SELECTED_RINGS_FILE, 'w', encoding='utf-8') as f: json.dump(serializable_log, f, indent=4)
            logging.info(f"Selected rings log saved: {SELECTED_RINGS_FILE}")
        except Exception as e: logging.error(f"Failed to save rings log: {e}", exc_info=True)
    else: logging.warning("Rings log is empty.")

    logging.info(f"Function embed_watermark_in_video (head processing) finished.")
    # Возвращаем ТОЛЬКО обработанные кадры "головы"
    return watermarked_frames

# --- ИЗМЕНЕННАЯ main ---
@profile  # Если используете line_profiler для main
def main() -> int:
    start_time_main = time.time()
    logging.info(f"--- Запуск Основного Процесса Встраивания (Голова+Хвост FFmpeg) ---")

    # --- Этап 1: Подготовка и Анализ ---
    # Проверки доступности библиотек уже в __main__

    # Настройка PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device=device)
            logging.info(f"Используется CUDA: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e_cuda:
            logging.error(f"Ошибка CUDA: {e_cuda}. Переключение на CPU.")
            device = torch.device("cpu")
    else:
        logging.info("Используется CPU.")

    # Создание экземпляров DTCWT
    dtcwt_fwd: Optional[DTCWTForward] = None
    dtcwt_inv: Optional[DTCWTInverse] = None
    if PYTORCH_WAVELETS_AVAILABLE:  # Глобальный флаг
        try:
            dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
            dtcwt_inv = DTCWTInverse(biort='near_sym_a', qshift='qshift_a').to(device)
            logging.info("Экземпляры DTCWTForward и DTCWTInverse созданы.")
        except Exception as e_dtcwt:
            logging.critical(f"Не удалось инициализировать DTCWT: {e_dtcwt}")
            return 1
    else:  # Эта проверка уже есть в __main__, но для надежности
        logging.critical("pytorch_wavelets недоступен! Невозможно создать DTCWT.")
        return 1

    # Определить Имена Файлов
    input_video_path = "test_video.mp4"  # ЗАМЕНИТЕ НА ВАШ ВХОДНОЙ ФАЙЛ
    if not os.path.exists(input_video_path):
        logging.critical(f"Входной файл не найден: {input_video_path}")
        print(f"ОШИБКА: Входной файл не найден: {input_video_path}")
        return 1

    base_output_filename = f"watermarked_ffmpeg_t{BCH_T}"  # Используем t из BCH для имени

    logging.info(f"Входное видео: '{input_video_path}'")
    logging.info(f"Базовое имя выходного файла: '{base_output_filename}'")

    # Получить Общее Число Кадров (OpenCV)
    total_original_frames = 0
    cap_check = None
    try:
        logging.debug(f"Попытка получить общее число кадров через OpenCV для '{input_video_path}'...")
        cap_check = cv2.VideoCapture(input_video_path)
        if cap_check.isOpened():
            total_original_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_original_frames <= 0:
                logging.warning("OpenCV вернул невалидное число кадров (<= 0).")
            else:
                logging.info(f"OpenCV определил общее число кадров: {total_original_frames}")
        else:
            logging.warning(f"OpenCV не смог открыть файл '{input_video_path}' для подсчета кадров.")
    except Exception as e_cv2_count:
        logging.warning(f"Ошибка при получении числа кадров через OpenCV: {e_cv2_count}")
    finally:
        if cap_check:
            cap_check.release()

    # Читать Метаданные (PyAV)
    logging.debug(f"Чтение метаданных через PyAV для '{input_video_path}'...")
    input_metadata = get_input_metadata(input_video_path)
    if input_metadata is None:
        logging.critical("Не удалось прочитать метаданные. Прерывание.")
        return 1

    # Уточнить Число Кадров (используя PyAV, если OpenCV не справился)
    if total_original_frames <= 0:
        pyav_frames_meta = input_metadata.get('total_frames')
        if pyav_frames_meta and isinstance(pyav_frames_meta, int) and pyav_frames_meta > 0:
            total_original_frames = pyav_frames_meta
            logging.info(f"Используется общее число кадров из метаданных PyAV: {total_original_frames}")
        else:
            logging.warning(
                "Не удалось определить общее число кадров ни через OpenCV, ни через PyAV. Копирование хвоста может быть неточным или невозможным.")
            total_original_frames = -1  # Флаг неизвестного числа кадров
    input_metadata['total_frames_estimated'] = total_original_frames

    # Извлечь оригинальный аудио битрейт для передачи
    original_audio_bitrate = input_metadata.get('audio_bitrate')  # Может быть int или None
    if original_audio_bitrate:
        logging.info(f"Извлечен оригинальный аудио битрейт: {original_audio_bitrate} bps")
    else:
        logging.info("Оригинальный аудио битрейт не найден в метаданных.")

    # Рассчитать "Голову"
    payload_id_for_calc_len = os.urandom(PAYLOAD_LEN_BYTES)
    bits_for_calc_len_list: List[int] = []
    raw_payload_bits_for_calc: np.ndarray = np.unpackbits(np.frombuffer(payload_id_for_calc_len, dtype=np.uint8))
    can_use_ecc_for_calc = USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT and (
                len(raw_payload_bits_for_calc) <= BCH_CODE_OBJECT.k)

    if can_use_ecc_for_calc:
        first_packet_calc = add_ecc(raw_payload_bits_for_calc, BCH_CODE_OBJECT)  # type: ignore
        if first_packet_calc is not None:
            bits_for_calc_len_list.extend(first_packet_calc.tolist())
        else:
            logging.warning("Ошибка расчета ECC для определения длины. Используется Raw для расчета.")
            bits_for_calc_len_list.extend(raw_payload_bits_for_calc.tolist())
        num_raw_to_add = max(0, MAX_TOTAL_PACKETS - 1)
        for _ in range(num_raw_to_add):
            bits_for_calc_len_list.extend(raw_payload_bits_for_calc.tolist())
    else:
        for _ in range(MAX_TOTAL_PACKETS if MAX_TOTAL_PACKETS > 0 else 1):
            bits_for_calc_len_list.extend(raw_payload_bits_for_calc.tolist())

    total_bits_to_embed_estimation = len(bits_for_calc_len_list)

    if BITS_PER_PAIR <= 0:
        logging.critical("BITS_PER_PAIR должен быть > 0. Прерывание.")
        return 1

    pairs_needed = math.ceil(
        total_bits_to_embed_estimation / BITS_PER_PAIR) if total_bits_to_embed_estimation > 0 else 0
    frames_to_process = pairs_needed * 2

    if total_original_frames > 0:  # Если известно общее число кадров
        frames_to_process = min(frames_to_process, total_original_frames)

    if frames_to_process % 2 != 0:  # Должно быть четным для парной обработки
        frames_to_process -= 1

    if frames_to_process <= 0:
        logging.warning(
            "Расчетное число кадров для обработки 'головы' равно нулю или отрицательно. Проверьте настройки полезной нагрузки, BITS_PER_PAIR и MAX_TOTAL_PACKETS.")
        print(
            "ПРЕДУПРЕЖДЕНИЕ: Нет кадров для обработки ЦВЗ. Выходной файл не будет изменен (или будет копией оригинала).")
        return 0
    logging.info(f"Расчетная 'голова' для встраивания ЦВЗ: {frames_to_process} кадров ({frames_to_process // 2} пар).")

    # Выбрать Параметры Выхода
    output_extension, target_video_encoder_lib_for_head, _ = check_compatibility_and_choose_output(input_metadata)
    audio_action_for_head_file = 'aac'  # Всегда AAC для временного файла головы
    logging.info(
        f"Параметры для временного файла 'головы': Видеокодер={target_video_encoder_lib_for_head}, Аудиокодер={audio_action_for_head_file}")
    logging.info(f"Параметры для финального файла: Расширение={output_extension}")

    # Определить Имена Файлов
    temp_head_path = base_output_filename + "_head" + output_extension
    final_output_path = base_output_filename + output_extension
    logging.info(f"Временный файл 'головы': {temp_head_path}")
    logging.info(f"Финальный выходной файл: {final_output_path}")

    # --- Этап 2: Чтение и Обработка "Головы" ---
    logging.info(f"Чтение 'головы' ({frames_to_process} кадров) и всех аудиопакетов...")
    video_idx = input_metadata.get('video_stream_index', 0)
    audio_idx = input_metadata.get('audio_stream_index', -1 if not input_metadata.get('has_audio') else 1)

    head_frames_bgr, all_audio_packets = read_processing_head(
        input_video_path, frames_to_process, video_idx, audio_idx
    )

    if head_frames_bgr is None or not head_frames_bgr:
        logging.critical(f"Не удалось прочитать 'голову' из '{input_video_path}' или список кадров пуст. Прерывание.")
        return 1

    if len(head_frames_bgr) < frames_to_process:
        logging.warning(
            f"Фактически прочитано кадров для 'головы' ({len(head_frames_bgr)}) меньше, чем рассчитано ({frames_to_process}). Используется фактическое число.")
        frames_to_process = len(head_frames_bgr)
        if frames_to_process % 2 != 0: frames_to_process -= 1
        if frames_to_process <= 0:
            logging.error("После коррекции не осталось кадров для обработки головы.")
            return 1
        head_frames_bgr = head_frames_bgr[:frames_to_process]  # Обрезаем список
    logging.info(
        f"Прочитано {len(head_frames_bgr)} видеокадров для 'головы'. Собрано {len(all_audio_packets or [])} аудиопакетов.")

    # Генерировать ID
    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES)
    original_id_hex = original_id_bytes.hex()
    logging.info(f"Сгенерирован Payload ID: {original_id_hex}")
    try:
        with open(ORIGINAL_WATERMARK_FILE, "w", encoding='utf-8') as f:
            f.write(original_id_hex)
        logging.info(f"Оригинальный ID сохранен: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e_id_save:
        logging.error(f"Не удалось сохранить ID: {e_id_save}")

    # Встроить ЦВЗ
    logging.info("Встраивание ЦВЗ в 'голову'...")
    watermarked_head_frames = embed_watermark_in_video(
        frames_to_process=head_frames_bgr, payload_id_bytes=original_id_bytes,
        n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE, bits_per_pair=BITS_PER_PAIR,
        candidate_pool_size=CANDIDATE_POOL_SIZE, use_hybrid_ecc=USE_ECC,
        max_total_packets=MAX_TOTAL_PACKETS, use_ecc_for_first=USE_ECC,
        bch_code=BCH_CODE_OBJECT, device=device, dtcwt_fwd=dtcwt_fwd, dtcwt_inv=dtcwt_inv,
        max_workers=SAFE_MAX_WORKERS, use_perceptual_masking=USE_PERCEPTUAL_MASKING,
        embed_component=EMBED_COMPONENT
    )
    if watermarked_head_frames is None or len(watermarked_head_frames) != len(head_frames_bgr):
        logging.critical("Ошибка при встраивании ЦВЗ в 'голову'. Прерывание.")
        return 1
    logging.info("Встраивание ЦВЗ в 'голову' завершено.")
    del head_frames_bgr;
    gc.collect();
    logging.debug("Память из-под исходных кадров 'головы' освобождена.")

    # --- Этап 3: Запись "Головы" во Временный Файл ---
    logging.info("Запись обработанной 'головы' во временный файл...")
    video_enc_opts_head = {'preset': 'medium', 'crf': '20'}
    # Используем оригинальный битрейт для аудио головы, если доступен, иначе дефолт
    audio_enc_opts_head = {
        'b:a': str(original_audio_bitrate)} if original_audio_bitrate and original_audio_bitrate >= 32000 else {
        'b:a': '128k'}

    head_duration_sec_calculated_from_pts = write_head_only(
        watermarked_head_frames=watermarked_head_frames,
        all_audio_packets=all_audio_packets if all_audio_packets else [],
        input_metadata=input_metadata, temp_head_path=temp_head_path,
        target_video_encoder_lib=target_video_encoder_lib_for_head,
        audio_action_for_head=audio_action_for_head_file,
        video_encoder_options=video_enc_opts_head, audio_encoder_options=audio_enc_opts_head
    )
    del watermarked_head_frames;
    gc.collect();
    logging.debug("Память из-под обработанных кадров 'головы' освобождена.")

    if head_duration_sec_calculated_from_pts is None or head_duration_sec_calculated_from_pts < 0:
        logging.critical(
            f"Ошибка при записи 'головы' или получена некорректная длительность ({head_duration_sec_calculated_from_pts}).")
        if os.path.exists(temp_head_path):
            try:
                os.remove(temp_head_path)
            except OSError as e_del_temp:
                logging.error(f"Не удалось удалить временный файл {temp_head_path}: {e_del_temp}")
        return 1
    logging.info(
        f"Обработанная 'голова' записана в '{temp_head_path}'. Расчетная видео длительность: {head_duration_sec_calculated_from_pts:.6f} сек.")

    # Используем расчетную длительность, возвращенную write_head_only
    duration_to_use_for_ffmpeg = head_duration_sec_calculated_from_pts
    logging.info(f"Длительность, передаваемая в FFmpeg для '-ss': {duration_to_use_for_ffmpeg:.6f}s")

    # --- Этап 4: Склейка с "Хвостом" через FFmpeg ---
    ffmpeg_success = False
    # Проверяем, если голова была пустой или покрывает весь файл
    if duration_to_use_for_ffmpeg == 0 and frames_to_process == 0:
        logging.info("Голова не содержала кадров для записи. Финальный файл не будет создан этим скриптом.")
        ffmpeg_success = True  # Не ошибка, просто нечего делать
        if os.path.exists(temp_head_path):  # На всякий случай, если он был создан (не должен)
            try:
                os.remove(temp_head_path);
            except OSError:
                pass
        print("ПРЕДУПРЕЖДЕНИЕ: Голова пуста. Финальный файл не создан.")
        return 0  # Выходим без ошибки, так как это ожидаемое поведение
    else:
        # Вызываем concatenate_with_ffmpeg (она сама проверит, нужен ли хвост)
        logging.info("Склейка 'головы' и 'хвоста' через FFmpeg...")
        ffmpeg_success = concatenate_with_ffmpeg(
            original_input_path=input_video_path, temp_head_path=temp_head_path,
            final_output_path=final_output_path, head_duration_sec=duration_to_use_for_ffmpeg,
            input_metadata=input_metadata, original_audio_bitrate=original_audio_bitrate
        )

    # Удалить Временный Файл Головы
    if os.path.exists(temp_head_path):
        # Удаляем, только если основной процесс (FFmpeg или переименование) прошел
        # или если FFmpeg не вызывался (голова была пуста, но этот случай обработан выше)
        if ffmpeg_success:
            try:
                os.remove(temp_head_path)
                logging.info(f"Временный файл 'головы' '{temp_head_path}' удален.")
            except OSError as e_remove:
                logging.error(f"Не удалось удалить временный файл 'головы' '{temp_head_path}': {e_remove}")
        else:
            logging.warning(f"Временный файл 'головы' '{temp_head_path}' не удален из-за предыдущей ошибки.")

    if not ffmpeg_success:
        logging.critical(f"Ошибка при создании финального файла '{final_output_path}'.")
        if os.path.exists(final_output_path):  # Почистить, если что-то создалось некорректно
            try:
                os.remove(final_output_path)
            except OSError:
                pass
        return 1

    # Проверка финального файла (только если он должен был быть создан)
    if duration_to_use_for_ffmpeg > 0 or (frames_to_process > 0 and os.path.exists(final_output_path)):
        if os.path.exists(final_output_path):
            logging.info(f"Финальный файл успешно создан: '{final_output_path}'")
            print(f"\nУспешно! Выходной файл: {final_output_path}")
        else:
            logging.error(
                f"Финальный файл '{final_output_path}' не найден после предполагаемого успеха (голова не была пустой).");
            return 1
    # Если голова была пуста, и ffmpeg_success=True, то файл не создавался, это нормально

    total_time_main = time.time() - start_time_main
    logging.info(f"--- Общее Время Выполнения: {total_time_main:.2f} сек ---")
    print(f"Завершено за {total_time_main:.2f} секунд.");
    print(f"Лог: {LOG_FILENAME}, ID: {ORIGINAL_WATERMARK_FILE}")
    return 0


# --- ПОЛНЫЙ БЛОК: Точка Входа (__name__ == "__main__") ---
if __name__ == "__main__":
    # Настройка логирования ДО всех операций
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                            format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)  # Установите logging.INFO для менее подробного лога

    # Проверка зависимостей на верхнем уровне
    missing_libs = []
    try:
        import cv2
    except ImportError:
        missing_libs.append("OpenCV (cv2)")
    try:
        import numpy  # np используется в add_ecc и т.д.
    except ImportError:
        missing_libs.append("NumPy")
    try:
        import torch
    except ImportError:
        missing_libs.append("PyTorch")
    try:
        import av
    except ImportError:
        missing_libs.append("PyAV (av)")

    if not PYTORCH_WAVELETS_AVAILABLE: missing_libs.append("pytorch_wavelets")
    if not TORCH_DCT_AVAILABLE: missing_libs.append("torch-dct")
    if USE_ECC and not GALOIS_AVAILABLE:  # GALOIS_AVAILABLE устанавливается при попытке импорта/инициализации BCH
        logging.warning("Библиотека 'galois' не найдена или не инициализирована корректно. ECC будет недоступен.")

    if missing_libs:
        error_msg = f"ОШИБКА: Отсутствуют необходимые библиотеки: {', '.join(missing_libs)}. Пожалуйста, установите их."
        print(error_msg)
        logging.critical(error_msg)
        sys.exit(1)

    DO_PROFILING = False  # Установите True для профилирования
    profiler = None
    if DO_PROFILING:
        if 'KERNPROF_VAR' not in os.environ and 'profile' not in globals():
            profiler = cProfile.Profile()
            profiler.enable()
            logging.info("cProfile профилирование включено.")
            print("cProfile профилирование включено.")

    final_exit_code = 1  # По умолчанию - ошибка
    try:
        final_exit_code = main()
    except FileNotFoundError as e_fnf:
        print(f"\nОШИБКА: Файл не найден: {e_fnf}")
        logging.critical(f"Файл не найден: {e_fnf}", exc_info=True)
    except av.FFmpegError as e_av:  # Убедитесь, что FFmpegError импортирован из av
        print(f"\nОШИБКА PyAV/FFmpeg: {e_av}")
        logging.critical(f"Ошибка PyAV/FFmpeg: {e_av}", exc_info=True)
    except torch.cuda.OutOfMemoryError as e_oom:
        print(f"\nОШИБКА: Недостаточно памяти CUDA: {e_oom}")
        logging.critical(f"Недостаточно памяти CUDA: {e_oom}", exc_info=True)
    except ImportError as e_imp:
        print(f"\nОШИБКА Импорта: {e_imp}")
        logging.critical(f"Ошибка импорта: {e_imp}", exc_info=True)
    except Exception as e_global:
        print(f"\nКРИТИЧЕСКАЯ НЕОБРАБОТАННАЯ ОШИБКА: {e_global}")
        logging.critical(f"Необработанная ошибка в __main__: {e_global}", exc_info=True)
    finally:
        if DO_PROFILING and profiler is not None:
            profiler.disable()
            logging.info("cProfile профилирование выключено.")
            stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
            print("\n--- Статистика Профилирования (cProfile, Top 30) ---");
            stats.print_stats(30)
            profile_stats_file = f"profile_embed_ffmpeg_t{BCH_T}.prof"
            try:
                stats.dump_stats(profile_stats_file)
                logging.info(f"Статистика профилирования cProfile сохранена: {profile_stats_file}")
                print(f"Статистика профилирования cProfile сохранена: {profile_stats_file}")
            except Exception as e_p_save:
                logging.error(f"Не удалось сохранить статистику профилирования cProfile: {e_p_save}")

        logging.info(f"Скрипт завершен с кодом выхода {final_exit_code}.")
        print(f"\nСкрипт завершен с кодом выхода {final_exit_code}.")
        sys.exit(final_exit_code)
