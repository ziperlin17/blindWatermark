# Файл: embedder_pytorch_wavelets.py (ПОСЛЕ РЕФАКТОРИНГА)
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
DEFAULT_OUTPUT_CONTAINER_EXT: str = ".mp4"
DEFAULT_VIDEO_ENCODER_LIB: str = "libx264"
DEFAULT_VIDEO_CODEC_NAME: str = "h264"
DEFAULT_AUDIO_ENCODER: str = "aac"
FALLBACK_CONTAINER_EXT: str = ".mkv"


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
MAX_WORKERS: Optional[int] = 1
MAX_TOTAL_PACKETS = 15
SAFE_MAX_WORKERS = 1

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


def check_compatibility_and_choose_output(input_metadata: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Анализирует метаданные, выбирает CPU-кодер, проверяет совместимость с MP4 (или MKV как fallback),
    и выбирает выходной контейнер и действие для аудио (перекодирование в AAC при несовместимости).

    Returns:
        Tuple[str, str, str]: (output_extension, target_video_encoder_lib, final_audio_action)
    """
    in_video_codec = input_metadata.get('video_codec')
    in_audio_codec = input_metadata.get('audio_codec') if input_metadata.get('has_audio') else None
    has_audio = input_metadata.get('has_audio', False)

    logging.info(f"Compatibility Check: Input Video='{in_video_codec}', Audio='{in_audio_codec}'")

    # --- 1. Выбираем ЦЕЛЕВУЮ БИБЛИОТЕКУ и ИМЯ КОДЕКА видео (CPU) ---
    target_video_encoder_lib = DEFAULT_VIDEO_ENCODER_LIB
    target_video_codec_name = DEFAULT_VIDEO_CODEC_NAME
    if in_video_codec == 'h264': target_video_encoder_lib = 'libx264'; target_video_codec_name = 'h264'
    elif in_video_codec == 'hevc': target_video_encoder_lib = 'libx265'; target_video_codec_name = 'hevc'
    elif in_video_codec == 'vp9': target_video_encoder_lib = 'libvpx-vp9'; target_video_codec_name = 'vp9'
    elif in_video_codec == 'vp8': target_video_encoder_lib = 'libvpx'; target_video_codec_name = 'vp8'
    elif in_video_codec == 'av1': target_video_encoder_lib = 'libaom-av1'; target_video_codec_name = 'av1'
    elif in_video_codec == 'mpeg4': target_video_encoder_lib = 'mpeg4'; target_video_codec_name = 'mpeg4'
    else:
        if in_video_codec: logging.warning(f"Input video codec '{in_video_codec}' unhandled. Using default: {DEFAULT_VIDEO_ENCODER_LIB} -> {DEFAULT_VIDEO_CODEC_NAME}.")
        target_video_encoder_lib = DEFAULT_VIDEO_ENCODER_LIB
        target_video_codec_name = DEFAULT_VIDEO_CODEC_NAME
    logging.info(f"Selected Target Video Encoder Lib (CPU): {target_video_encoder_lib} -> Codec Name: {target_video_codec_name}")

    # --- 2. Определяем начальное аудио действие и целевой кодек ---
    initial_audio_action = 'copy' if has_audio else 'none'
    target_audio_codec_name = None
    if initial_audio_action == 'copy': target_audio_codec_name = in_audio_codec
    elif has_audio: target_audio_codec_name = DEFAULT_AUDIO_ENCODER

    # --- 3. Определяем предпочтительный контейнер ---
    preferred_extension = DEFAULT_OUTPUT_CONTAINER_EXT # .mp4
    if target_video_codec_name in ('vp8', 'vp9', 'av1') and \
       (not has_audio or target_audio_codec_name in ('opus', 'vorbis')):
       preferred_extension = ".webm"
       logging.info("Preferring WebM container for VPx/AV1 + Opus/Vorbis.")
    elif target_video_codec_name in ('h264', 'hevc') and \
         (not has_audio or target_audio_codec_name in ('aac', 'mp3', 'alac')):
         preferred_extension = ".mp4"
         logging.info("Preferring MP4 container for H.264/HEVC + AAC/MP3/ALAC.")

    # --- 4. Проверяем совместимость с ПРЕДПОЧТИТЕЛЬНЫМ контейнером ---
    output_extension = preferred_extension
    final_audio_action = initial_audio_action
    logging.debug(f"Checking compatibility for preferred container: {output_extension}")
    allowed_codecs = CODEC_CONTAINER_COMPATIBILITY.get(output_extension, set())

    video_ok = ('video', target_video_codec_name) in allowed_codecs
    audio_ok = not has_audio or \
               (target_audio_codec_name and ('audio', target_audio_codec_name) in allowed_codecs)

    if not (video_ok and audio_ok):
        output_extension = FALLBACK_CONTAINER_EXT # .mkv
        logging.warning(f"Incompatibility detected with preferred '{preferred_extension}' (Video OK: {video_ok}, Audio OK: {audio_ok}). Switching to fallback '{output_extension}'.")
        if has_audio:
            # Для MKV пытаемся копировать исходное аудио, если возможно
            if in_audio_codec and ('audio', in_audio_codec) in CODEC_CONTAINER_COMPATIBILITY.get(output_extension, set()):
                 final_audio_action = 'copy'
                 logging.info(f"Audio action set to 'copy' for fallback MKV container.")
            else: # Если даже MKV не поддерживает или кодек небезопасен, перекодируем
                 final_audio_action = DEFAULT_AUDIO_ENCODER
                 logging.warning(f"Input audio codec '{in_audio_codec}' not suitable even for MKV fallback. Re-encoding to '{DEFAULT_AUDIO_ENCODER}'.")
        else: final_audio_action = 'none'
    else:
        # Предпочтительный контейнер совместим, но нужно ли перекодировать аудио?
        if initial_audio_action == 'copy' and has_audio and not audio_ok:
             # Эта ветка сработает, если, например, предпочли MP4, но аудио было Opus
             logging.warning(f"Input audio codec '{in_audio_codec}' not standard for preferred container '{output_extension}'. Switching audio action to re-encode ('{DEFAULT_AUDIO_ENCODER}').")
             final_audio_action = DEFAULT_AUDIO_ENCODER
        # Если audio_ok было True (либо не было аудио, либо оно совместимо для копии/перекодирования), то final_audio_action остается как initial

    # --- Финальный Лог ---
    logging.info(f"Final Decision: Output Extension='{output_extension}', Video Encoder Lib='{target_video_encoder_lib}', Audio Action='{final_audio_action}'")
    return output_extension, target_video_encoder_lib, final_audio_action

# --- Функция чтения видео (остается без изменений) ---
def get_input_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Читает только метаданные из видеофайла с использованием PyAV,
    с fallback на OpenCV для базовых параметров (W, H, FPS).
    """
    if not PYAV_AVAILABLE: logging.error("PyAV not available."); return None

    metadata: Dict[str, Any] = {
        'input_path': video_path, 'format_name': '',
        'video_codec': None, 'width': 0, 'height': 0, 'fps': float(FPS),
        'video_bitrate': None, 'video_time_base': None, 'pix_fmt': None,
        'color_space_tag': None, 'color_primaries_tag': None,
        'color_transfer_tag': None, 'color_range_tag': None,
        'has_audio': False, 'audio_codec': None, 'audio_rate': None,
        'audio_layout': None, 'audio_time_base': None, 'audio_codec_context_params': None,
        'video_stream_index': -1, 'audio_stream_index': -1 # Добавляем индексы
    }
    input_container: Optional['av.container.Container'] = None
    opencv_fallback_used = False

    try:
        # --- Попытка открыть с PyAV ---
        try:
            logging.info(f"Attempting to open '{video_path}' with PyAV to read metadata...")
            input_container = av.open(video_path, mode='r')
            if not input_container: raise FFmpegError("av.open returned None")
            logging.info("Opened with PyAV successfully.")
            if input_container.format: metadata['format_name'] = input_container.format.name
            else: logging.warning("Could not determine container format name.")

        except (FFmpegError, Exception) as e_open:
            logging.error(f"PyAV failed to open '{video_path}' for metadata: {e_open}", exc_info=True)
            logging.warning("Attempting OpenCV fallback for basic metadata.")
            opencv_fallback_used = True
            cap_cv2 = None
            try:
                cap_cv2 = cv2.VideoCapture(video_path)
                if cap_cv2.isOpened():
                    w_cv2 = int(cap_cv2.get(cv2.CAP_PROP_FRAME_WIDTH)); h_cv2 = int(cap_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps_cv2 = float(cap_cv2.get(cv2.CAP_PROP_FPS))
                    if w_cv2 > 0 and h_cv2 > 0: metadata['width'] = w_cv2; metadata['height'] = h_cv2; logging.info(f"OpenCV Fallback: Got dimensions {w_cv2}x{h_cv2}")
                    if fps_cv2 and fps_cv2 > 0: metadata['fps'] = fps_cv2; logging.info(f"OpenCV Fallback: Got FPS {fps_cv2:.2f}")
                else: logging.error("OpenCV fallback failed: Could not open video.")
            except Exception as e_cv2_meta: logging.error(f"Error during OpenCV metadata fallback: {e_cv2_meta}")
            finally:
                if cap_cv2: cap_cv2.release()
            # Если PyAV не открылся, дальше идти нет смысла, возвращаем что есть
            return metadata

        # --- Чтение метаданных потоков через PyAV ---
        # Видео
        try:
            video_stream = input_container.streams.video[0]
            metadata['video_stream_index'] = video_stream.index
            ctx = video_stream.codec_context
            metadata['video_codec'] = video_stream.codec.name
            metadata['width'] = ctx.width if ctx.width and ctx.width > 0 else metadata['width']
            metadata['height'] = ctx.height if ctx.height and ctx.height > 0 else metadata['height']
            if video_stream.average_rate: metadata['fps'] = float(video_stream.average_rate)
            metadata['pix_fmt'] = ctx.pix_fmt
            metadata['video_bitrate'] = ctx.bit_rate or input_container.bit_rate
            metadata['video_time_base'] = video_stream.time_base
            metadata['color_space_tag'] = video_stream.metadata.get('color_space')
            metadata['color_primaries_tag'] = video_stream.metadata.get('color_primaries')
            metadata['color_transfer_tag'] = video_stream.metadata.get('color_transfer')
            metadata['color_range_tag'] = video_stream.metadata.get('color_range')
            logging.info(f"  PyAV Video Stream Meta: Codec={metadata['video_codec']}, Res={metadata['width']}x{metadata['height']}, FPS={metadata['fps']:.2f}")
        except (IndexError, FFmpegError, AttributeError) as e:
            logging.error(f"PyAV: Error accessing video stream properties: {e}")
            if not (metadata['width'] > 0 and metadata['height'] > 0 and metadata['fps'] > 0):
                 logging.critical("Essential video metadata missing after checks."); return None # Не можем продолжить

        # Аудио
        input_audio_streams = input_container.streams.audio
        if input_audio_streams:
            try:
                audio_stream = input_audio_streams[0]
                metadata['audio_stream_index'] = audio_stream.index
                ctx = audio_stream.codec_context
                metadata['has_audio'] = True
                metadata['audio_codec'] = audio_stream.codec.name
                metadata['audio_rate'] = ctx.rate
                metadata['audio_layout'] = ctx.layout.name
                metadata['audio_time_base'] = audio_stream.time_base
                metadata['audio_codec_context_params'] = {
                    'format': ctx.format.name if ctx.format else None, 'layout': metadata['audio_layout'],
                    'rate': metadata['audio_rate'], 'bit_rate': ctx.bit_rate,
                    'codec_tag': ctx.codec_tag, 'extradata': bytes(ctx.extradata) if ctx.extradata else None,
                }
                logging.info(f"  PyAV Audio Stream Meta: Codec={metadata['audio_codec']}, Rate={metadata['audio_rate']}")
            except (IndexError, FFmpegError, AttributeError) as e:
                logging.warning(f"Could not get/process audio stream metadata: {e}"); metadata['has_audio'] = False
        else: logging.info("No audio streams found by PyAV."); metadata['has_audio'] = False

        # Проверяем еще раз критичные параметры
        if not (metadata['width'] > 0 and metadata['height'] > 0 and metadata['fps'] > 0):
             logging.critical("Essential video metadata missing after all checks."); return None

        return metadata

    except Exception as e: # Ловим другие неожиданные ошибки
        logging.error(f"Unexpected error getting metadata for '{video_path}': {e}", exc_info=True)
        return None # Возвращаем None при серьезной ошибке
    finally:
        if input_container:
            try: input_container.close(); logging.debug("Metadata reading: Input container closed.")
            except FFmpegError as e: logging.error(f"Error closing input container after metadata read: {e}")


# --- Новая функция для чтения ТОЛЬКО нужных кадров и ВСЕХ аудиопакетов ---
@profile
def read_processing_head(video_path: str, frames_to_read: int, video_stream_index: int, audio_stream_index: int) -> Tuple[Optional[List[np.ndarray]], Optional[List['av.Packet']]]:
    """
    Читает и декодирует ТОЛЬКО первые `frames_to_read` видеокадров
    и собирает ВСЕ аудиопакеты из указанных потоков.
    """
    if not PYAV_AVAILABLE: logging.error("PyAV not available."); return None, None
    if frames_to_read <= 0: logging.warning("Frames to read is zero or negative."); return [], []

    head_frames_bgr: List[np.ndarray] = []
    all_audio_packets: List['av.Packet'] = []
    input_container: Optional['av.container.Container'] = None
    frames_decoded_count = 0

    logging.info(f"Reading processing head: {frames_to_read} video frames and all audio packets...")
    try:
        input_container = av.open(video_path, mode='r')
        if input_container is None: raise FFmpegError("av.open returned None")

        logging.debug(f"Reading packets (target video_idx={video_stream_index}, audio_idx={audio_stream_index})")
        try:
            for packet in input_container.demux():
                # Собираем ВСЕ аудио пакеты нужного потока
                if packet.stream.index == audio_stream_index:
                    if packet.dts is not None: # Пропускаем flush пакеты
                         all_audio_packets.append(packet)

                # Декодируем видео пакеты, пока не наберем нужное количество кадров
                elif packet.stream.index == video_stream_index:
                    if frames_decoded_count >= frames_to_read:
                        continue # Больше видео кадры не нужны

                    if packet.dts is None: continue

                    try:
                        for frame in packet.decode():
                            if frame and isinstance(frame, av.VideoFrame):
                                # Конвертируем в BGR NumPy
                                try:
                                     np_frame = frame.to_ndarray(format='bgr24')
                                     head_frames_bgr.append(np_frame)
                                     frames_decoded_count += 1
                                     if frames_decoded_count % 100 == 0:
                                          logging.debug(f"Decoded {frames_decoded_count}/{frames_to_read} head video frames...")
                                     if frames_decoded_count >= frames_to_read:
                                          break # Выходим из внутреннего цикла по кадрам
                                except (FFmpegError, ValueError, TypeError) as e_conv:
                                     logging.warning(f"Failed convert frame to bgr24: {e_conv}. Try YUV.")
                                     try:
                                          frame_yuv = frame.reformat(format='yuv420p')
                                          np_frame_yuv = frame_yuv.to_ndarray()
                                          np_frame_bgr = cv2.cvtColor(np_frame_yuv, cv2.COLOR_YUV2BGR_I420)
                                          head_frames_bgr.append(np_frame_bgr)
                                          frames_decoded_count += 1
                                          if frames_decoded_count % 100 == 0: logging.debug(f"Decoded {frames_decoded_count}/{frames_to_read} head video frames...")
                                          if frames_decoded_count >= frames_to_read: break
                                     except Exception as e_reformat: logging.error(f"Failed reformat/convert frame via YUV: {e_reformat}"); continue
                    except (FFmpegError, ValueError) as e_decode:
                        logging.warning(f"Error decoding video packet: {e_decode} - skipping.")
                # Выходим из основного цикла, если набрали достаточно видеокадров
                if frames_decoded_count >= frames_to_read:
                    # Но продолжаем читать аудио до конца файла! (Упрощение)
                    # Чтобы читать аудио только для "головы", нужна логика сравнения PTS
                    pass # Пока читаем все аудио

            # Если вышли из цикла, а кадров не хватило (например, короткое видео)
            if frames_decoded_count < frames_to_read:
                 logging.warning(f"Read only {frames_decoded_count} video frames, less than requested {frames_to_read}.")

        except (FFmpegError, FFmpegEOFError) as e:
             logging.warning(f"Error or EOF during demuxing in read_processing_head: {e}")
             # Продолжаем с тем, что успели прочитать

        logging.info(f"Finished reading head. Decoded {len(head_frames_bgr)} video frames. Collected {len(all_audio_packets)} audio packets.")
        return head_frames_bgr, all_audio_packets

    except (FFmpegError, FileNotFoundError, Exception) as e:
        logging.error(f"Error reading processing head from '{video_path}': {e}", exc_info=True)
        return None, None
    finally:
        if input_container:
            try: input_container.close(); logging.debug("Head reading: Input container closed.")
            except FFmpegError as e: logging.error(f"Error closing input container after head read: {e}")


# --- Функция записи видео (остается без изменений) ---
# @profile # Добавьте, если нужно профилировать
def rescale_time(value: Optional[int], old_tb: Optional[Fraction], new_tb: Optional[Fraction], label: str = "") -> Optional[int]:
    """
    Пересчитывает значение времени (PTS, DTS, Duration) из одной time_base в другую.
    Возвращает None при ошибке или если входные данные некорректны.
    Добавлено логирование при ошибке.
    """
    if value is None or old_tb is None or new_tb is None \
       or not isinstance(old_tb, Fraction) or not isinstance(new_tb, Fraction) \
       or old_tb.denominator == 0 or new_tb.denominator == 0:
         # Не логируем None как ошибку, просто возвращаем None
         # logging.debug(f"Rescale ({label}): Input value or time base is None/invalid. value={value}, old={old_tb}, new={new_tb}")
         return None
    try:
         # Формула: новое = старое * (старый_tb / новый_tb)
         scaled_value = Fraction(value * old_tb.numerator * new_tb.denominator, old_tb.denominator * new_tb.numerator)
         # Округляем до ближайшего целого
         if scaled_value >= 0: result = int(scaled_value + Fraction(1, 2))
         else: result = int(scaled_value - Fraction(1, 2))
         # logging.debug(f"Rescale ({label}): {value} * ({old_tb} / {new_tb}) = {scaled_value} -> {result}")
         return result
    except (ZeroDivisionError, OverflowError, TypeError) as e:
         logging.warning(f"Rescale warning ({label}): value={value}, old={old_tb}, new={new_tb}. Error: {e}")
         return None

# --- Функция записи "Голова + Хвост" с детальным логированием ---
@profile
def write_video_head_tail(
    watermarked_head_frames: List[np.ndarray],
    all_audio_packets: List['av.Packet'],
    input_metadata: Dict[str, Any],
    output_path: str,
    target_video_encoder_lib: str, # Имя библиотеки кодера (e.g., 'libx264')
    audio_codec_action: str,      # 'copy' или 'aac'
    num_processed_frames: int     # Количество кадров в голове
    ) -> bool:
    """
    Записывает видео: кодирует обработанную "голову" и копирует "хвост" видео и аудио.
    Использует переданные параметры кодера и аудио действия.
    Выполняет ручной пересчет таймстемпов для копируемых пакетов.
    Реализована потоковая запись головы и детальное логирование.
    """
    if not PYAV_AVAILABLE: logging.error("PyAV not available."); return False

    output_container: Optional['av.container.Container'] = None
    input_container_tail: Optional['av.container.Container'] = None
    input_audio_decoder: Optional['av.codec.context.CodecContext'] = None
    video_stream: Optional['av.stream.Stream'] = None
    audio_stream: Optional['av.stream.Stream'] = None

    logging.info(f"Preparing output '{output_path}' (Head+Tail - Video: {target_video_encoder_lib}, Audio: {audio_codec_action})...")

    # --- Получение и проверка метаданных ---
    width = input_metadata.get('width'); height = input_metadata.get('height')
    fps = input_metadata.get('fps', float(FPS))
    input_video_bitrate = input_metadata.get('video_bitrate')
    input_pix_fmt = input_metadata.get('pix_fmt')
    input_video_time_base = input_metadata.get('video_time_base')
    input_video_stream_index = input_metadata.get('video_stream_index', -1)
    has_audio = input_metadata.get('has_audio', False) and audio_codec_action != 'none'
    input_audio_codec = input_metadata.get('audio_codec')
    input_audio_rate = input_metadata.get('audio_rate')
    input_audio_layout = input_metadata.get('audio_layout')
    input_audio_time_base = input_metadata.get('audio_time_base')
    input_audio_context_params = input_metadata.get('audio_codec_context_params')
    input_audio_stream_index = input_metadata.get('audio_stream_index', -1)
    in_color_space = input_metadata.get('color_space_tag')
    in_color_primaries = input_metadata.get('color_primaries_tag')
    in_color_transfer = input_metadata.get('color_transfer_tag')
    in_color_range = input_metadata.get('color_range_tag')

    if not (width and height and fps and fps > 0): logging.error("Invalid video metadata."); return False
    if has_audio and audio_codec_action == 'copy' and not input_audio_time_base: logging.error("Cannot copy audio: input audio time_base missing."); has_audio = False
    if has_audio and audio_codec_action != 'copy' and not (input_audio_codec and input_audio_rate and input_audio_layout): logging.warning("Essential audio metadata missing for re-encoding."); has_audio = False

    fps_fraction = Fraction(fps).limit_denominator() if fps and fps > 0 else None

    # --- Определение Видео Настроек для кодирования "Головы" ---
    codec_options = {}; pix_fmt_out = 'yuv420p'
    if target_video_encoder_lib == 'libx264': codec_options = {'preset': 'fast', 'crf': '20'};
    elif target_video_encoder_lib == 'libx265': codec_options = {'preset': 'fast', 'crf': '22'}; pix_fmt_out = 'yuv420p10le' if '10' in (input_pix_fmt or '') else 'yuv420p'
    elif target_video_encoder_lib == 'libvpx-vp9': codec_options = {'crf': '30', 'b:v': '0'}
    elif target_video_encoder_lib == 'mpeg4': codec_options = {'qscale:v': '4'}
    # Установка битрейта, если он есть и кодек не CRF/qscale/QP
    if input_video_bitrate and input_video_bitrate > 10000 and not any(k in codec_options for k in ['crf', 'qp', 'qscale:v']):
         codec_options['b'] = str(input_video_bitrate)
    logging.info(f"Using codec options for head ({target_video_encoder_lib}): {codec_options}")

    # --- Основной блок записи ---
    try:
        # --- Открытие выходного контейнера ---
        try:
            output_container = av.open(output_path, mode='w')
            if output_container is None: raise FFmpegError("av.open returned None")
            logging.info(f"Opened '{output_path}' for writing.")
        except (FFmpegError, Exception) as e: logging.error(f"Failed to open output container: {e}", exc_info=True); return False

        # --- Настройка Видео Потока ---
        logging.info(
            f"Setting up video stream: codec={target_video_encoder_lib}, rate={fps_fraction or 'auto'}, pix_fmt={pix_fmt_out}")
        try:
            video_stream = output_container.add_stream(target_video_encoder_lib, rate=fps_fraction)
            # !!! ИСПРАВЛЕНО: Настройка потока ВНУТРИ try после успешного add_stream !!!
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = pix_fmt_out
            if codec_options:
                video_stream.codec_context.options = codec_options
            # Устанавливаем стандартный time_base
            video_stream.time_base = Fraction(1, 90000)
            # --- Установка цветовых тегов (через metadata) ---
            # (Логика проверки и установки тегов как раньше)
            force_standard_tags = False
            # ... (код проверки in_color_space и т.д.) ...
            if force_standard_tags or not any(
                    t is not None for t in [in_color_space, in_color_primaries, in_color_transfer, in_color_range]):
                output_color_space, output_color_primaries, output_color_transfer, output_color_range = 'bt709', 'bt709', 'bt709', 'tv'
                logging.info("Forcing/Setting BT.709 TV color tags.")
            else:
                output_color_space, output_color_primaries, output_color_transfer, output_color_range = in_color_space, in_color_primaries, in_color_transfer, in_color_range
                logging.info("Using input color tags.")
            try:
                if output_color_space: video_stream.metadata['color_space'] = output_color_space
                if output_color_primaries: video_stream.metadata['color_primaries'] = output_color_primaries
                if output_color_transfer: video_stream.metadata['color_transfer'] = output_color_transfer
                if output_color_range: video_stream.metadata['color_range'] = output_color_range
                logging.info(f"Set output stream metadata color tags.")
            except Exception as e_meta:
                logging.warning(f"Could not set color tags via metadata: {e_meta}")
            logging.info(f"  Video stream time_base: {video_stream.time_base}")

        except Exception as e:
            # Ловим ошибку при add_stream или при настройке параметров
            logging.error(f"Error setting up video stream: {e}", exc_info=True)
            if output_container:
                try:
                    output_container.close()
                except FFmpegError as e_close:
                    logging.error(f"Error closing container after stream setup error: {e_close}")
            return False  # Возвращаем False, если не удалось настроить поток

        # --- Настройка Аудио Потока ---
        output_audio_time_base = None
        if has_audio:
            logging.info(f"Setting up audio stream: action={audio_codec_action}")
            if audio_codec_action == 'copy':
                try:
                    audio_stream = output_container.add_stream(input_audio_codec, rate=input_audio_rate)
                    out_ctx = audio_stream.codec_context
                    # ... (копирование параметров контекста) ...
                    out_ctx.layout = input_audio_layout; out_ctx.format = input_audio_context_params.get('format'); out_ctx.bit_rate = input_audio_context_params.get('bit_rate'); out_ctx.codec_tag = input_audio_context_params.get('codec_tag'); out_ctx.extradata = input_audio_context_params.get('extradata')
                    audio_stream.time_base = input_audio_time_base # Явно устанавливаем
                    output_audio_time_base = audio_stream.time_base
                    logging.info(f"  Audio Stream Setup (Copy): codec={audio_stream.codec.name}, "
                                 f"time_base={audio_stream.time_base}, "
                                 f"rate={audio_stream.codec_context.rate}, "
                                 f"layout={audio_stream.codec_context.layout.name}, "
                                 f"format={audio_stream.codec_context.format.name if audio_stream.codec_context.format else 'N/A'}")
                except Exception as e: logging.error(f"Error setting up audio copy: {e}"); has_audio = False
            else: # Перекодирование
                 target_audio_codec_actual = audio_codec_action
                 try:
                     audio_stream = output_container.add_stream(target_audio_codec_actual, rate=input_audio_rate)
                     audio_stream.codec_context.layout = input_audio_layout
                     if target_audio_codec_actual == 'aac': audio_stream.codec_context.bit_rate = 128000
                     output_audio_time_base = audio_stream.time_base
                     logging.info(f"  Audio Stream Setup (Re-encode): codec={audio_stream.codec.name}, "
                                  f"time_base={audio_stream.time_base}, "
                                  f"rate={audio_stream.codec_context.rate}, "
                                  f"layout={audio_stream.codec_context.layout.name}, "
                                  f"format={audio_stream.codec_context.format.name if audio_stream.codec_context.format else 'N/A'}, "
                                  f"bitrate={audio_stream.codec_context.bit_rate}")
                 except Exception as e: logging.error(f"Error setting up audio re-encode: {e}"); has_audio = False
        else: has_audio = False


        # --- Декодирование аудио для перекодирования "головы" ---
        audio_frames_to_encode = []
        audio_frame_iter = iter([])
        if has_audio and audio_codec_action != 'copy':
             head_duration_sec = float(num_processed_frames / fps) if fps > 0 else 0
             logging.debug(f"Decoding audio packets up to ~{head_duration_sec:.2f}s for re-encoding head...")
             try:
                 in_audio_codec_name = input_metadata.get('audio_codec')
                 if not in_audio_codec_name: raise ValueError("Input audio codec name missing")
                 input_audio_decoder = av.Codec(in_audio_codec_name, 'r').create()
                 packets_processed_for_decode = 0
                 for packet in all_audio_packets:
                      packet_time_sec = float(packet.pts * input_audio_time_base) if packet.pts is not None and input_audio_time_base else None
                      if packet_time_sec is not None and packet_time_sec > head_duration_sec and packets_processed_for_decode > 0: break
                      decoded = input_audio_decoder.decode(packet)
                      if decoded: audio_frames_to_encode.extend(decoded)
                      packets_processed_for_decode += 1
                 decoded = input_audio_decoder.decode(None)
                 if decoded: audio_frames_to_encode.extend(decoded)
                 logging.debug(f"Decoded {len(audio_frames_to_encode)} audio frames for head.")
                 audio_frame_iter = iter(audio_frames_to_encode)
             except (FFmpegError, ValueError, Exception) as e_adec: logging.error(f"Error decoding audio for head: {e_adec}. Disabling audio."); has_audio = False; audio_frames_to_encode = []
             finally:
                  if input_audio_decoder: input_audio_decoder.close()

        # --- ЧАСТЬ 1: Потоковая Запись "Головы" ---
        logging.info(f"Encoding and muxing video head ({num_processed_frames} frames)...")
        head_frame_count = 0; head_audio_packets_muxed = 0; head_audio_frames_encoded = 0
        reformatter = None
        audio_packet_iter = iter(all_audio_packets) if (has_audio and audio_codec_action == 'copy') else iter([])
        current_audio_packet = next(audio_packet_iter, None)
        current_audio_frame = next(audio_frame_iter, None) # Для перекодирования

        for frame_index, np_frame in enumerate(watermarked_head_frames):
            try: # Обертка для одного кадра
                video_frame: Optional[VideoFrame] = None
                if np_frame is None: logging.warning(f"Head frame {frame_index} is None."); continue
                # logging.debug(f"Frame {frame_index} shape: {np_frame.shape}, dtype: {np_frame.dtype}")

                video_frame = VideoFrame.from_ndarray(np_frame, format='bgr24')
                if video_frame.format.name != pix_fmt_out:
                    if reformatter is None: reformatter = VideoReformatter(video_frame.width, video_frame.height, pix_fmt_out)
                    video_frame = reformatter.reformat(video_frame)

                if video_stream.time_base and fps > 0:
                    time_sec = float(frame_index) / float(fps); video_frame.pts = int(time_sec / float(video_stream.time_base))
                else: video_frame.pts = frame_index
                if video_frame.pts is None: logging.warning(f"Head Video frame {frame_index} got None PTS."); continue

                encoded_video_packets = video_stream.encode(video_frame)
                for packet in encoded_video_packets: output_container.mux(packet)
                head_frame_count += 1

                # --- Мультиплексирование Аудио (чередование) ---
                if has_audio and video_frame.pts is not None and audio_stream:
                    current_video_pts_in_audio_tb = rescale_time(video_frame.pts, video_stream.time_base, output_audio_time_base, f"Vid{frame_index}_PTS") if video_stream.time_base and output_audio_time_base else None

                    if audio_codec_action == 'copy':
                        while current_audio_packet is not None:
                            packet_pts_in_out_tb = rescale_time(current_audio_packet.pts, input_audio_time_base, output_audio_time_base, f"AudPkt_PTS_Comp")
                            if current_audio_packet.pts is None or current_video_pts_in_audio_tb is None or packet_pts_in_out_tb is None or packet_pts_in_out_tb <= current_video_pts_in_audio_tb:
                                try:
                                    original_pts = current_audio_packet.pts; original_dts = current_audio_packet.dts
                                    current_audio_packet.pts = rescale_time(current_audio_packet.pts, input_audio_time_base, output_audio_time_base, "AudPkt_PTS_Mux")
                                    current_audio_packet.dts = rescale_time(current_audio_packet.dts, input_audio_time_base, output_audio_time_base, "AudPkt_DTS_Mux")
                                    if current_audio_packet.pts is None and original_pts is not None: logging.warning("Audio head PTS became None.")
                                    current_audio_packet.stream = audio_stream
                                    output_container.mux(current_audio_packet); head_audio_packets_muxed += 1
                                except Exception as e_mux_a: logging.warning(f"Muxing head audio packet failed: {e_mux_a}")
                                current_audio_packet = next(audio_packet_iter, None)
                            else: break
                    else: # Перекодирование
                         while current_audio_frame is not None:
                             frame_pts = current_audio_frame.pts # PTS должен быть в output_audio_time_base
                             if frame_pts is None or current_video_pts_in_audio_tb is None or frame_pts <= current_video_pts_in_audio_tb:
                                 try:
                                     encoded_audio_packets = audio_stream.encode(current_audio_frame)
                                     output_container.mux(encoded_audio_packets); head_audio_frames_encoded += 1
                                 except Exception as e_enc_a: logging.warning(f"Encoding/Muxing head audio frame failed: {e_enc_a}")
                                 current_audio_frame = next(audio_frame_iter, None)
                             else: break

            except (FFmpegError, ValueError, TypeError, ZeroDivisionError, MemoryError, AttributeError, Exception) as e_inner:
                 logging.error(f"!!! CRITICAL ERROR processing head frame {frame_index}: {e_inner} !!!", exc_info=True)
                 logging.error("Aborting write process due to critical error in head processing.")
                 if output_container: output_container.close();
                 return False # Прерываем запись

        logging.info(f"Finished encoding/muxing head: {head_frame_count} video frames processed.")
        if has_audio: logging.info(f"  Head audio: {head_audio_packets_muxed} packets copied / {head_audio_frames_encoded} frames re-encoded.")

        # --- ЧАСТЬ 2: Копирование "Хвоста" ---
        logging.info("Starting tail copying process...")
        tail_video_packets_copied = 0; tail_audio_packets_copied = 0
        input_container_tail = None
        last_video_pts_head_scaled = rescale_time(video_frame.pts, video_stream.time_base, input_video_time_base, "LastHeadVidPTS_In") if video_frame and video_frame.pts is not None and video_stream.time_base and input_video_time_base else None

        try:
            input_container_tail = av.open(input_metadata['input_path'], mode='r')
            logging.debug("Re-opened input container for tail reading.")

            # --- Логика пропуска пакетов "головы" (ОСНОВАНА НА PTS) ---
            head_skipped = False
            packets_scanned = 0
            logging.debug("Scanning input packets to find tail start...")
            for packet in input_container_tail.demux():
                packets_scanned += 1
                if packet.dts is None: continue

                # Пропуск пакетов головы
                if not head_skipped:
                    # Ищем первый видео пакет, чей PTS (в своей базе) БОЛЬШЕ последнего PTS головы (в той же базе)
                    if packet.stream.index == input_video_stream_index:
                        if last_video_pts_head_scaled is not None and packet.pts is not None and packet.pts > last_video_pts_head_scaled:
                             if packet.is_keyframe:
                                 head_skipped = True
                                 logging.info(f"Found starting keyframe for tail copy at PTS {packet.pts} (orig) after scanning ~{packets_scanned} packets.")
                                 # Этот пакет уже относится к хвосту, обрабатываем его ниже
                             else: continue # Ищем ключевой кадр
                        else: continue # Продолжаем пропускать
                    else: continue # Пропускаем не-видео

                # --- Логика Копирования Пакетов Хвоста ---
                if not head_skipped: continue # Если все еще не пропустили голову

                is_video = packet.stream.type == 'video'
                packet_log_prefix = f"Tail {'Vid' if is_video else 'Aud'} Pkt:" # Упрощенный префикс

                try:
                    input_tb = packet.stream.time_base
                    output_stream = video_stream if is_video else audio_stream
                    output_tb = output_stream.time_base if output_stream else None

                    if not (output_stream and input_tb and output_tb):
                         logging.log(logging.DEBUG if not has_audio and not is_video else logging.WARNING, # Логируем как DEBUG если аудио просто нет
                                     f"{packet_log_prefix} Skipping due to missing stream/time_base (Stream Idx: {packet.stream.index})")
                         continue

                    # --- Ручной пересчет таймстемпов ---
                    original_pts = packet.pts; original_dts = packet.dts; original_duration = packet.duration
                    new_pts = rescale_time(packet.pts, input_tb, output_tb, "PTS")
                    new_dts = rescale_time(packet.dts, input_tb, output_tb, "DTS")
                    new_duration = rescale_time(packet.duration, input_tb, output_tb, "DUR") if original_duration is not None and original_duration > 0 else 0

                    # !!! ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ПЕРЕД MUX !!!
                    log_level = logging.DEBUG # Уровень для детального лога пакетов
                    logging.log(log_level, f"{packet_log_prefix} "
                                     f"Key={packet.is_keyframe if is_video else '-'}, "
                                     f"Size={packet.size}, "
                                     f"In [PTS:{original_pts}, DTS:{original_dts}, DUR:{original_duration}, TB:{input_tb}] -> "
                                     f"Out [PTS:{new_pts}, DTS:{new_dts}, DUR:{new_duration}, TB:{output_tb}]")

                    if new_pts is None and original_pts is not None:
                        logging.error(f"{packet_log_prefix} FATAL: PTS became None after rescale! Skipping.")
                        continue
                    if new_dts is None and original_dts is not None:
                        logging.warning(f"{packet_log_prefix} DTS became None after rescale.")

                    packet.pts = new_pts
                    packet.dts = new_dts
                    packet.duration = new_duration if new_duration is not None else 0

                    packet.stream = output_stream
                    # logging.debug(f"{packet_log_prefix} Muxing...")
                    output_container.mux(packet)
                    # logging.debug(f"{packet_log_prefix} Mux OK.")

                    if is_video: tail_video_packets_copied += 1
                    elif has_audio: tail_audio_packets_copied += 1

                except (FFmpegError, ValueError, TypeError) as e:
                     logging.warning(f"{packet_log_prefix} Error rescaling/muxing tail packet (Orig PTS: {original_pts}): {e}")

            if not head_skipped: logging.warning("Could not find start of tail? Tail might be missing.")
            logging.info(f"Finished copying tail. Video packets: {tail_video_packets_copied}, Audio packets: {tail_audio_packets_copied}")

        except (FFmpegError, FFmpegEOFError) as e: logging.warning(f"Error or EOF during tail demuxing: {e}")
        finally:
             if input_container_tail: input_container_tail.close(); logging.debug("Tail reading: Input container closed.")


        # --- Flush Кодеров ---
        logging.info("Flushing encoders (if any)...")
        try: # Flush видео
            if video_stream is not None:
                encoded_packets = video_stream.encode(None)
                if encoded_packets: output_container.mux(encoded_packets)
        except (FFmpegError, ValueError, TypeError) as e:
             logging.warning(f"Non-critical error flushing video encoder: {e}")

        if has_audio and audio_stream is not None and audio_codec_action != 'copy':
            try: # Flush аудио (если перекодировали)
                 while current_audio_frame is not None: # Докодируем остатки
                     try: encoded_audio_packets = audio_stream.encode(current_audio_frame); output_container.mux(encoded_audio_packets)
                     except Exception as e_f: logging.warning(f"Encoding/Muxing final audio frame failed: {e_f}")
                     current_audio_frame = next(audio_frame_iter, None)
                 encoded_packets = audio_stream.encode(None) # Сам flush
                 if encoded_packets: output_container.mux(encoded_packets)
            except (FFmpegError, ValueError, TypeError) as e: logging.warning(f"Error flushing audio encoder: {e}")

        logging.info(f"Finished writing process.")
        return True

    # --- Обработка Основных Ошибок ---
    except FFmpegError as e: logging.error(f"PyAV/FFmpeg error during writing setup or muxing: {e}", exc_info=True); return False
    except Exception as e: logging.error(f"Unexpected error writing video: {e}", exc_info=True); return False
    finally:
        # --- Закрытие Ресурсов ---
        if input_audio_decoder:
             try: input_audio_decoder.close(); logging.debug("Input audio decoder closed.")
             except Exception as e_dclose: logging.warning(f"Error closing input audio decoder: {e_dclose}")
        if output_container:
            try: output_container.close(); logging.debug("Output container closed.")
            except FFmpegError as e: logging.error(f"Error closing output container: {e}")

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
@profile
def main():
    start_time_main = time.time()

    # --- Инициализация и проверки зависимостей ---
    if not PYAV_AVAILABLE: print("ERROR: PyAV required."); return 1
    if not PYTORCH_WAVELETS_AVAILABLE: print("ERROR: pytorch_wavelets required."); return 1
    if not TORCH_DCT_AVAILABLE: print("ERROR: torch-dct required."); return 1
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")

    # --- Настройка PyTorch ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Using CUDA: {gpu_name}")
        try: _ = torch.tensor([1.0], device=device); logging.info("CUDA check OK.")
        except RuntimeError as e: logging.error(f"CUDA error: {e}. Fallback CPU."); device = torch.device("cpu")
    else: logging.info("Using CPU.")

    # --- Создание экземпляров DTCWT ---
    dtcwt_fwd: Optional[DTCWTForward] = None; dtcwt_inv: Optional[DTCWTInverse] = None
    try:
        dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
        dtcwt_inv = DTCWTInverse(biort='near_sym_a', qshift='qshift_a').to(device)
        logging.info("DTCWT instances created.")
    except Exception as e: logging.critical(f"Failed to init DTCWT: {e}"); return 1

    # --- Определение имен файлов ---
    input_video = "test_video.mp4" # Укажите ваш входной файл
    base_output_filename = f"watermarked_pyav_h+t_t{BCH_T}" # Новое имя для "Голова+Хвост"
    # Полное имя выходного файла будет определено позже

    logging.info(f"--- Starting Embedding Main Process (Head+Tail Mode) ---")
    logging.info(f"Input video: '{input_video}'")
    logging.info(f"Base output filename: '{base_output_filename}'")

    # --- Шаг 1: Чтение Метаданных ---
    logging.info("Reading input metadata...")
    input_metadata = get_input_metadata(input_video)
    if input_metadata is None: logging.critical("Failed to read metadata. Aborting."); return 1
    # Получаем FPS сразу из прочитанных метаданных
    fps_to_use = input_metadata.get('fps', float(FPS))
    if fps_to_use <= 0: logging.warning(f"Invalid FPS ({fps_to_use}). Using default {float(FPS)}"); fps_to_use = float(FPS)
    logging.info(f"Using FPS: {fps_to_use:.2f}")

    # --- Шаг 1.5: Выбор параметров выхода и проверка совместимости ---
    logging.info("Checking compatibility and choosing output parameters...")
    output_extension, target_video_encoder_lib, final_audio_action = check_compatibility_and_choose_output(input_metadata)
    output_video = base_output_filename + output_extension # Формируем ПОЛНОЕ имя
    logging.info(f"Output settings: Extension='{output_extension}', Video Encoder='{target_video_encoder_lib}', Audio Action='{final_audio_action}'")
    logging.info(f"Output file path: '{output_video}'")

    # --- Шаг 1.7: Расчет необходимого количества кадров/пар для обработки ---
    payload_id_bytes_for_calc = os.urandom(PAYLOAD_LEN_BYTES) # Генерируем временный для расчета
    payload_len_bits = len(payload_id_bytes_for_calc) * 8
    bits_to_embed_list_for_calc = []
    raw_payload_bits_for_calc = np.unpackbits(np.frombuffer(payload_id_bytes_for_calc, dtype=np.uint8))
    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT and payload_len_bits <= BCH_CODE_OBJECT.k:
        first_packet = add_ecc(raw_payload_bits_for_calc, BCH_CODE_OBJECT)
        if first_packet is not None: bits_to_embed_list_for_calc.extend(first_packet.tolist())
        else: logging.error("ECC calculation failed for size estimation."); return 1
        num_raw_packets = max(0, MAX_TOTAL_PACKETS - 1)
        for _ in range(num_raw_packets): bits_to_embed_list_for_calc.extend(raw_payload_bits_for_calc.tolist())
    else: # Только Raw
        bits_to_embed_list_for_calc.extend(raw_payload_bits_for_calc.tolist()) # Только один пакет
    total_bits_to_embed = len(bits_to_embed_list_for_calc)
    if BITS_PER_PAIR <= 0: logging.error("bits_per_pair must be positive."); return 1
    pairs_needed = ceil(total_bits_to_embed / BITS_PER_PAIR)
    frames_to_process = pairs_needed * 2
    logging.info(f"Calculated frames to process (head): {frames_to_process} ({pairs_needed} pairs)")

    # --- Шаг 2: Чтение "Головы" видео и ВСЕГО аудио ---
    logging.info(f"Reading head ({frames_to_process} frames) and all audio packets...")
    head_frames_bgr, all_audio_packets = read_processing_head(
        input_video,
        frames_to_process,
        input_metadata.get('video_stream_index', -1),
        input_metadata.get('audio_stream_index', -1)
    )
    if head_frames_bgr is None: logging.critical("Failed to read processing head. Aborting."); return 1
    # all_audio_packets может быть None или []
    logging.info(f"Read head: {len(head_frames_bgr)} video frames. Collected {len(all_audio_packets or [])} audio packets.")
    if len(head_frames_bgr) < frames_to_process:
         logging.warning(f"Actual head frames read ({len(head_frames_bgr)}) is less than calculated ({frames_to_process}). Video might be too short.")
         # Обновляем количество реально обработанных кадров/пар, если видео короче
         frames_to_process = len(head_frames_bgr)
         pairs_needed = frames_to_process // (2 * BITS_PER_PAIR) * BITS_PER_PAIR # Пересчет пар, кратных BITS_PER_PAIR

    # --- Шаг 3: Генерация и Сохранение ID ---
    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES)
    original_id_hex = original_id_bytes.hex()
    logging.info(f"Generated Payload ID: {original_id_hex}")
    try:
        with open(ORIGINAL_WATERMARK_FILE, "w") as f: f.write(original_id_hex)
        logging.info(f"Original ID saved: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e: logging.error(f"Save ID failed: {e}")

    # --- Шаг 4: Встраивание ЦВЗ в "Голову" ---
    logging.info("Starting watermark embedding on head frames...")
    embed_kwargs = { # Собираем аргументы
        'frames_to_process': head_frames_bgr, # <--- Передаем только "голову"
        'payload_id_bytes': original_id_bytes,
        'use_hybrid_ecc': True, 'max_total_packets': 15, 'use_ecc_for_first': USE_ECC,
        'bch_code': BCH_CODE_OBJECT, 'device': device, 'dtcwt_fwd': dtcwt_fwd, 'dtcwt_inv': dtcwt_inv,
        'n_rings': N_RINGS, 'num_rings_to_use': NUM_RINGS_TO_USE, 'bits_per_pair': BITS_PER_PAIR,
        'candidate_pool_size': CANDIDATE_POOL_SIZE,
        'max_workers': SAFE_MAX_WORKERS, # <--- Используем БЕЗОПАСНОЕ значение
        'use_perceptual_masking': USE_PERCEPTUAL_MASKING, 'embed_component': EMBED_COMPONENT
    }
    watermarked_head_frames = embed_watermark_in_video(**embed_kwargs)
    if watermarked_head_frames is None or len(watermarked_head_frames) != len(head_frames_bgr):
        logging.error("Embedding failed or frame count mismatch for head."); return 1
    logging.info("Watermark embedding on head completed.")

    # --- Шаг 5: Запись "Головы" и Копирование "Хвоста" ---
    write_success = False
    logging.info("Attempting to write final video (Head+Tail)...")
    write_success = write_video_head_tail(
        watermarked_head_frames=watermarked_head_frames,
        all_audio_packets=all_audio_packets if all_audio_packets else [],
        input_metadata=input_metadata,
        output_path=output_video,
        target_video_encoder_lib=target_video_encoder_lib, # Передаем выбранный кодер
        audio_codec_action=final_audio_action,           # Передаем выбранное действие
        num_processed_frames=len(watermarked_head_frames) # Передаем реальное число обработанных кадров
    )

    if write_success: logging.info(f"Output saved successfully: {output_video}"); print(f"\nOutput: {output_video}")
    else: logging.error("Writing video failed."); print("\nERROR: Failed to write video.")

    # --- Завершение ---
    logging.info(f"--- Embedding Main Process Finished (Head+Tail Mode) ---")
    total_time_main = time.time() - start_time_main
    logging.info(f"--- Total Main Time: {total_time_main:.2f} sec ---")
    print(f"\nFinished in {total_time_main:.2f} seconds.")
    print(f"Log: {LOG_FILENAME}, ID: {ORIGINAL_WATERMARK_FILE}, Rings: {SELECTED_RINGS_FILE}")
    if write_success: print(f"\nRun extractor_pytorch.py on '{output_video}'")
    else: print("\nOutput file write failed.")
    return 0 if write_success else 1

# --- Точка Входа (__name__ == "__main__") ---
if __name__ == "__main__":
    # --- Настройка профилирования ---
    DO_PROFILING = False
    profiler = None
    if DO_PROFILING and 'KERNPROF_VAR' not in os.environ:
        profiler = cProfile.Profile(); profiler.enable()
        logging.info("cProfile profiling enabled."); print("cProfile profiling enabled.")

    final_exit_code = 1 # По умолчанию ошибка
    try:
        # --- Инициализация и проверки зависимостей ---
        if not PYAV_AVAILABLE: print("\nERROR: PyAV required."); sys.exit(1)
        if not PYTORCH_WAVELETS_AVAILABLE: print("\nERROR: pytorch_wavelets required."); sys.exit(1)
        if not TORCH_DCT_AVAILABLE: print("\nERROR: torch-dct required."); sys.exit(1)
        if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")

        # --- Запуск основной логики ---
        final_exit_code = main()

    # --- Обработка ошибок ---
    except FileNotFoundError as e: print(f"\nERROR: File not found: {e}"); logging.error(f"{e}", exc_info=True); final_exit_code = 1
    except FFmpegError as e: print(f"\nERROR: PyAV/FFmpeg: {e}"); logging.critical(f"{e}", exc_info=True); final_exit_code = 1
    except torch.cuda.OutOfMemoryError as e: print(f"\nERROR: CUDA OOM: {e}"); logging.critical(f"{e}", exc_info=True); final_exit_code = 1
    except ImportError as e: print(f"\nERROR: Missing library: {e}"); logging.critical(f"{e}", exc_info=True); final_exit_code = 1
    except Exception as e: logging.critical(f"Unhandled error: {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}"); final_exit_code = 1
    finally:
        # --- Сохранение профиля ---
        if DO_PROFILING and profiler is not None:
            profiler.disable(); logging.info("cProfile profiling disabled.")
            stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
            stats.print_stats(30)
            profile_stats_file = f"profile_embed_head_tail_t{BCH_T}.prof" # Новое имя файла профиля
            try: stats.dump_stats(profile_stats_file); logging.info(f"Profile saved: {profile_stats_file}")
            except Exception as e_p: logging.error(f"Save profile failed: {e_p}")

    # --- Завершение скрипта ---
    logging.info(f"Script finished with exit code {final_exit_code}.")
    print(f"\nScript finished with exit code {final_exit_code}.")
    sys.exit(final_exit_code)
