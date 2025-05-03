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
from typing import List, Tuple, Optional, Dict, Any
import uuid
from math import ceil
import cProfile
import pstats
from fractions import Fraction

# --- Импорт PyAV ---
try:
    import av
    # Импортируем базовый класс ошибок FFmpeg
    from av import FFmpegError # Базовый класс доступен напрямую из av
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
MAX_WORKERS: Optional[int] = 14

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



# --- Функция чтения видео (остается без изменений) ---
def read_video_pyav(video_path: str) -> Tuple[Optional[List[np.ndarray]], Optional[List['av.Packet']], Optional[Dict[str, Any]]]:
    """
    Читает видеофайл с использованием PyAV, извлекая видеокадры (BGR NumPy),
    аудиопакеты и метаданные. (Исправлена ошибка AVError и логика demux).
    """
    if not PYAV_AVAILABLE:
        logging.error("PyAV is not available. Cannot read video.")
        return None, None, None

    video_frames_bgr: List[np.ndarray] = []
    audio_packets: List['av.Packet'] = []
    # Инициализируем метаданные с разумными значениями по умолчанию
    metadata: Dict[str, Any] = {
        'input_path': video_path,
        'video_codec': None, 'width': 0, 'height': 0, 'fps': float(FPS), # Используем глобальный FPS как fallback
        'video_bitrate': None, 'video_time_base': None, 'pix_fmt': None,
        'has_audio': False, 'audio_codec': None, 'audio_rate': None,
        'audio_layout': None, 'audio_time_base': None, 'audio_codec_context_params': None
    }

    input_container: Optional['av.container.Container'] = None
    video_stream_index: int = -1
    audio_stream_index: int = -1

    try:
        logging.info(f"Opening input '{video_path}' with PyAV...")
        input_container = av.open(video_path, mode='r')

        # --- Определение индекса видеопотока ---
        video_stream = None
        try:
            video_stream = input_container.streams.video[0]
            video_stream.codec_context.thread_count = max(1, os.cpu_count() // 2)
            video_stream_index = video_stream.index # Получаем индекс
            # Заполняем метаданные видео
            metadata['video_codec'] = video_stream.codec.name
            metadata['width'] = video_stream.codec_context.width
            metadata['height'] = video_stream.codec_context.height
            metadata['pix_fmt'] = video_stream.codec_context.pix_fmt
            if video_stream.average_rate:
                metadata['fps'] = float(video_stream.average_rate)
            metadata['video_bitrate'] = video_stream.codec_context.bit_rate or input_container.bit_rate # Исправлено ранее
            metadata['video_time_base'] = video_stream.time_base
            logging.info(f"  Video Stream #{video_stream_index}: {metadata['video_codec']}, {metadata['width']}x{metadata['height']}, "
                         f"FPS: {metadata['fps']:.2f}, Bitrate: {metadata['video_bitrate']}, "
                         f"PixFmt: {metadata['pix_fmt']}, TimeBase: {metadata['video_time_base']}")
        except (IndexError, FFmpegError, AttributeError): # Ловим возможные ошибки
            logging.error("No video stream found or error accessing its properties.", exc_info=True)
            return None, None, None # Видео обязательно

        # --- Определение индекса аудиопотока ---
        audio_stream = None
        input_audio_streams = input_container.streams.audio
        if input_audio_streams:
            try:
                audio_stream = input_audio_streams[0] # Берем первый аудиопоток
                audio_stream_index = audio_stream.index # Получаем индекс
                # Заполняем метаданные аудио
                input_audio_ctx = audio_stream.codec_context
                metadata['has_audio'] = True
                metadata['audio_codec'] = audio_stream.codec.name
                metadata['audio_rate'] = input_audio_ctx.rate
                metadata['audio_layout'] = input_audio_ctx.layout.name
                metadata['audio_time_base'] = audio_stream.time_base
                metadata['audio_codec_context_params'] = {
                    'format': input_audio_ctx.format.name if input_audio_ctx.format else None,
                    'layout': metadata['audio_layout'],
                    'rate': metadata['audio_rate'],
                    'bit_rate': input_audio_ctx.bit_rate,
                    'codec_tag': input_audio_ctx.codec_tag,
                    'extradata': bytes(input_audio_ctx.extradata) if input_audio_ctx.extradata else None,
                }
                logging.info(f"  Audio Stream #{audio_stream_index}: {metadata['audio_codec']}, Rate: {metadata['audio_rate']}, "
                             f"Layout: {metadata['audio_layout']}, TimeBase: {metadata['audio_time_base']}")
            except (IndexError, FFmpegError, AttributeError) as e:
                logging.warning(f"Could not get or process first audio stream: {e}", exc_info=True)
                metadata['has_audio'] = False
                audio_stream_index = -1 # Сбрасываем индекс
        else:
            logging.warning("No audio streams found in the input file.")
            metadata['has_audio'] = False

        # --- Чтение Пакетов (без фильтрации в demux) ---
        logging.info(f"Reading and processing packets (seeking video_idx={video_stream_index}, audio_idx={audio_stream_index})...")
        processed_frames = 0
        try:
            # Демультиплексируем ВСЕ пакеты
            for packet in input_container.demux():

                if packet.dts is None: continue # Пропускаем пакеты без таймстемпов

                # Обработка видео пакетов по ИНДЕКСУ
                if packet.stream.index == video_stream_index:
                    try:
                        for frame in packet.decode():
                            if frame and isinstance(frame, av.VideoFrame):
                                np_frame = frame.to_ndarray(format='bgr24')
                                video_frames_bgr.append(np_frame)
                                processed_frames += 1
                                if processed_frames % 100 == 0:
                                    logging.debug(f"Decoded {processed_frames} video frames...")
                    except (FFmpegError, ValueError) as e: # Ловим ошибки декодирования
                        logging.warning(f"Error decoding video packet (stream {packet.stream.index}): {e} - skipping packet.")
                        continue

                # Сохранение аудио пакетов по ИНДЕКСУ
                elif packet.stream.index == audio_stream_index:
                    audio_packets.append(packet)

                # Пакеты других потоков игнорируются

        except (FFmpegError, EOFError) as e: # Ловим ошибки демультиплексирования или конец файла
             logging.warning(f"Error or EOF during demuxing: {e}", exc_info=False) # Логируем как предупреждение
             if not video_frames_bgr: # Если видео не успели прочитать, это фатально
                 logging.error("Demuxing failed before any video frames were read.")
                 return None, None, None

        logging.info(f"Finished reading. Decoded {len(video_frames_bgr)} video frames. Collected {len(audio_packets)} audio packets.")
        if not video_frames_bgr:
            logging.error("No video frames were decoded successfully.")
            return None, None, None

        return video_frames_bgr, audio_packets, metadata

    except FileNotFoundError:
        logging.error(f"Input file not found: {video_path}")
        return None, None, None
    except FFmpegError as e: # Ловим ошибки PyAV/FFmpeg при открытии файла
        logging.error(f"PyAV/FFmpeg error opening '{video_path}': {e}", exc_info=True)
        return None, None, None
    except Exception as e: # Ловим другие неожиданные ошибки
        logging.error(f"Unexpected error reading video '{video_path}': {e}", exc_info=True)
        return None, None, None
    finally:
        if input_container:
            try:
                input_container.close()
                logging.debug("Input container closed.")
            except FFmpegError as e:
                 logging.error(f"Error closing input container: {e}")

# --- Функция записи видео (остается без изменений) ---
# @profile # Добавьте, если нужно профилировать
def write_video_pyav(
    processed_video_frames: List[np.ndarray],
    original_audio_packets: List['av.Packet'],
    input_metadata: Dict[str, Any],
    output_path: str,
    target_video_codec: str = 'libx264',
    video_quality_crf: int = 23,
    audio_codec_action: str = 'copy'
    ) -> bool:
    """
    Записывает видео и аудио, полагаясь на mux для обработки таймстемпов
    при копировании аудио. Упрощенный расчет PTS видео.
    """
    if not PYAV_AVAILABLE: logging.error("PyAV not available."); return False
    if not processed_video_frames: logging.error("No video frames."); return False

    output_container: Optional['av.container.Container'] = None
    logging.info(f"Preparing output '{output_path}' (Simplified PTS)...")

    # --- Метаданные ---
    width = input_metadata.get('width'); height = input_metadata.get('height')
    fps = input_metadata.get('fps', float(FPS))
    input_video_bitrate = input_metadata.get('video_bitrate')
    has_audio = input_metadata.get('has_audio', False)
    input_audio_codec = input_metadata.get('audio_codec')
    input_audio_rate = input_metadata.get('audio_rate')
    input_audio_layout = input_metadata.get('audio_layout')
    input_audio_time_base = input_metadata.get('audio_time_base') # Нужен для копирования stream.time_base
    input_audio_context_params = input_metadata.get('audio_codec_context_params')

    if not (width and height and fps and fps > 0): logging.error("Invalid video metadata."); return False
    if has_audio and not (input_audio_codec and input_audio_rate and input_audio_layout and input_audio_time_base and input_audio_context_params):
        logging.warning("Audio metadata incomplete. Writing video only.")
        has_audio = False

    # --- Преобразование FPS ---
    fps_fraction = Fraction(fps).limit_denominator() if fps and fps > 0 else None

    try:
        # --- Открытие контейнера ---
        try:
            output_container = av.open(output_path, mode='w')
            if output_container is None: raise FFmpegError("av.open returned None")
            logging.info(f"Opened '{output_path}' for writing.")
        except (FFmpegError, Exception) as e:
            logging.error(f"Failed to open output container '{output_path}': {e}", exc_info=True)
            return False

        # --- Настройка Видео Потока ---
        logging.info(f"Setting up video stream: codec={target_video_codec}, rate={fps_fraction or 'auto'}")
        try:
            # Попробуем установить time_base = 1/fps
            video_time_base = None
            if fps_fraction:
                 try:
                      # Инвертируем дробь, чтобы получить 1/fps
                      video_time_base = Fraction(fps_fraction.denominator, fps_fraction.numerator)
                 except ZeroDivisionError:
                      logging.warning("FPS denominator is zero, cannot set video time_base from FPS.")
                      video_time_base = None

            # Если не удалось из FPS, используем стандартный 1/90000
            if video_time_base is None:
                 video_time_base = Fraction(1, 90000)

            video_stream = output_container.add_stream(target_video_codec, rate=fps_fraction)
            video_stream.time_base = video_time_base # !!! УСТАНАВЛИВАЕМ time_base !!!
        except Exception as e:
             logging.error(f"Error adding video stream: {e}", exc_info=True)
             if output_container: output_container.close();
             return False

        video_stream.width = width; video_stream.height = height; video_stream.pix_fmt = 'yuv420p'
        if input_video_bitrate and input_video_bitrate > 0:
            video_stream.codec_context.bit_rate = input_video_bitrate
        elif target_video_codec in ['libx264', 'libx265']:
            video_stream.codec_context.options = {'crf': str(video_quality_crf)}
        logging.info(f"  Video stream time_base set to: {video_stream.time_base}")

        # --- Настройка Аудио Потока ---
        audio_stream = None
        if has_audio and original_audio_packets:
            logging.info(f"Setting up audio stream: action={audio_codec_action}")
            if audio_codec_action == 'copy':
                if not input_audio_time_base:
                     logging.error("Cannot copy audio: input time_base missing."); has_audio = False
                else:
                    try:
                        audio_stream = output_container.add_stream(input_audio_codec, rate=input_audio_rate)
                        out_ctx = audio_stream.codec_context
                        # Копируем основные параметры
                        out_ctx.layout = input_audio_layout; out_ctx.format = input_audio_context_params.get('format')
                        out_ctx.bit_rate = input_audio_context_params.get('bit_rate')
                        out_ctx.codec_tag = input_audio_context_params.get('codec_tag')
                        out_ctx.extradata = input_audio_context_params.get('extradata')
                        # ЯВНО УСТАНАВЛИВАЕМ time_base
                        audio_stream.time_base = input_audio_time_base
                        logging.info(f"  Audio stream time_base set to: {audio_stream.time_base}")
                    except Exception as e: logging.error(f"Error setting up audio copy: {e}"); has_audio = False
            else: # Перекодирование
                 try:
                     audio_stream = output_container.add_stream(audio_codec_action, rate=input_audio_rate)
                     audio_stream.codec_context.layout = input_audio_layout
                     if audio_codec_action == 'aac': audio_stream.codec_context.bit_rate = 128000
                     logging.info(f"  Audio stream time_base (re-encode): {audio_stream.time_base}")
                 except Exception as e: logging.error(f"Error setting up audio re-encode: {e}"); has_audio = False
        else: has_audio = False


        # --- Кодирование и Мультиплексирование (Упрощенное) ---
        logging.info("Starting encoding and simplified muxing...")
        frame_count = 0; audio_packet_count = 0

        # Сначала кодируем и мультиплексируем все видео
        logging.debug("Encoding video frames...")
        video_packets_to_mux = []
        for frame_index, np_frame in enumerate(processed_video_frames):
            try:
                video_frame = av.VideoFrame.from_ndarray(np_frame, format='bgr24')
                # !!! УСТАНАВЛИВАЕМ PTS КАК ПРОСТОЙ ИНДЕКС !!!
                # FFmpeg должен использовать time_base потока (1/fps) для интерпретации
                video_frame.pts = frame_index

                encoded_packets = video_stream.encode(video_frame)
                video_packets_to_mux.extend(encoded_packets) # Собираем пакеты
                frame_count += 1
            except (FFmpegError, ValueError, TypeError) as e:
                 logging.warning(f"Error encoding video frame {frame_index}: {e} - skipping.")
        # Flush видео кодера
        try:
            encoded_packets = video_stream.encode(None)
            video_packets_to_mux.extend(encoded_packets)
        except (FFmpegError, ValueError, TypeError) as e: logging.warning(f"Error flushing video: {e}")
        logging.debug(f"Total video frames encoded: {frame_count}. Total packets generated: {len(video_packets_to_mux)}")

        # Мультиплексируем все видео пакеты
        logging.debug("Muxing video packets...")
        for packet in video_packets_to_mux:
             try: output_container.mux(packet)
             except (FFmpegError, ValueError) as e: logging.warning(f"Error muxing video packet (PTS:{packet.pts}): {e}")


        # Теперь копируем все аудио пакеты
        if has_audio:
             logging.debug("Muxing audio packets...")
             for packet in original_audio_packets:
                 try:
                     # !!! ВАЖНО: Присваиваем пакет правильному выходному потоку !!!
                     packet.stream = audio_stream
                     # Мультиплексируем (без rescale_ts)
                     output_container.mux(packet)
                     audio_packet_count += 1
                 except (FFmpegError, ValueError, TypeError) as e:
                     logging.warning(f"Error muxing audio packet {audio_packet_count} (PTS:{packet.pts}): {e}")
             logging.debug(f"Total audio packets muxed: {audio_packet_count}")

        logging.info(f"Finished writing process.")
        return True

    except FFmpegError as e:
        logging.error(f"PyAV/FFmpeg error during writing: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"Unexpected error writing video: {e}", exc_info=True)
        return False
    finally:
        if output_container:
            try: output_container.close(); logging.debug("Output container closed.")
            except FFmpegError as e: logging.error(f"Error closing container: {e}")

# --- ИЗМЕНЕННАЯ embed_frame_pair для PyTorch ---
# @profile # Добавьте, если нужно профилировать
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
                    logging.info(f"{prefix} PT Delta1 Stats: mean={delta1_pt.mean():.6e}, std={delta1_pt.std():.6e}")
                    logging.info(f"{prefix} PT Delta2 Stats: mean={delta2_pt.mean():.6e}, std={delta2_pt.std():.6e}")

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


def embed_watermark_in_video(
        frames: List[np.ndarray],
        payload_id_bytes: bytes, # Принимаем байты ID
        n_rings: int = N_RINGS, num_rings_to_use: int = NUM_RINGS_TO_USE,
        bits_per_pair: int = BITS_PER_PAIR, candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        # --- Параметры гибридного режима ---
        use_hybrid_ecc: bool = True,       # Включить гибридный режим?
        max_total_packets: int = 15,       # Макс. общее число пакетов (1 ECC + N Raw)
        use_ecc_for_first: bool = USE_ECC, # Использовать ECC для первого пакета? (берем из глоб. USE_ECC)
        bch_code: Optional[BCH_TYPE] = BCH_CODE_OBJECT, # Глобальный объект BCH
        # --- Новые параметры для PyTorch ---
        device: torch.device = torch.device("cpu"),      # Устройство по умолчанию CPU
        dtcwt_fwd: Optional[DTCWTForward] = None, # Объект прямого преобр.
        dtcwt_inv: Optional[DTCWTInverse] = None, # Объект обратного преобр.
        # ------------------------------------
        fps: float = FPS, max_workers: Optional[int] = MAX_WORKERS,
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT):
    """
    Основная функция, управляющая процессом встраивания с использованием PyTorch Wavelets,
    ThreadPoolExecutor, батчинга и гибридного ECC/Raw режима.
    """
    # Проверка наличия объектов PyTorch
    if not PYTORCH_WAVELETS_AVAILABLE:
        logging.critical("PyTorch Wavelets не доступна!")
        return frames[:] # Возвращаем без изменений
    if dtcwt_fwd is None or dtcwt_inv is None:
        logging.critical("Экземпляры DTCWTForward/DTCWTInverse не переданы в embed_watermark_in_video!")
        return frames[:] # Возвращаем без изменений

    num_frames = len(frames);
    total_pairs = num_frames // 2
    payload_len_bytes = len(payload_id_bytes);
    payload_len_bits = payload_len_bytes * 8

    # Проверки входных данных
    if payload_len_bits == 0:
        logging.error("Payload ID пустой! Встраивание отменено.")
        return frames[:]
    if total_pairs == 0:
        logging.warning("Нет пар кадров для встраивания")
        return frames[:]
    if max_total_packets <= 0:
         logging.warning(f"max_total_packets ({max_total_packets}) должен быть > 0. Установлено в 1.")
         max_total_packets = 1

    logging.info(f"--- Embed Start (PyTorch, Hybrid: {use_hybrid_ecc}, Max Pkts: {max_total_packets}) ---")

    # --- Формирование последовательности бит для встраивания ---
    bits_to_embed_list = []
    raw_payload_bits: Optional[np.ndarray] = None # Указываем тип
    try:
        raw_payload_bits = np.unpackbits(np.frombuffer(payload_id_bytes, dtype=np.uint8))
        if raw_payload_bits.size != payload_len_bits:
             raise ValueError(f"Ошибка unpackbits: ожидалось {payload_len_bits} бит, получено {raw_payload_bits.size}")
    except Exception as e:
        logging.error(f"Ошибка подготовки raw_payload_bits: {e}", exc_info=True)
        return frames[:]
    # Добавляем проверку на None после try-except, хотя она избыточна при текущей логике
    if raw_payload_bits is None:
         logging.error("Не удалось создать raw_payload_bits.")
         return frames[:]

    first_packet_bits: Optional[np.ndarray] = None
    packet1_type_str = "N/A"
    packet1_len = 0
    num_raw_packets_added = 0
    # Проверяем возможность использовать ECC
    can_use_ecc = use_ecc_for_first and GALOIS_AVAILABLE and bch_code is not None and payload_len_bits <= bch_code.k

    # 1. Формируем ПЕРВЫЙ пакет
    if use_hybrid_ecc and can_use_ecc:
        first_packet_bits = add_ecc(raw_payload_bits, bch_code)
        if first_packet_bits is not None:
            bits_to_embed_list.extend(first_packet_bits.tolist())
            packet1_len = len(first_packet_bits)
            packet1_type_str = f"ECC(n={packet1_len}, t={bch_code.t})"
            logging.info(f"Гибридный режим: Первый пакет создан как {packet1_type_str}.")
        else:
            # Если add_ecc вернул None, это критично для гибридного режима с ECC
            logging.error("Не удалось создать первый пакет с ECC (ошибка add_ecc)! Встраивание отменено.")
            return frames[:]
    else:
        # Используем Raw payload для первого пакета
        first_packet_bits = raw_payload_bits # Он уже создан и проверен
        bits_to_embed_list.extend(first_packet_bits.tolist())
        packet1_len = len(first_packet_bits)
        packet1_type_str = f"Raw({packet1_len})"
        # Логируем причину, почему первый пакет - Raw
        if not use_hybrid_ecc:
             logging.info(f"Режим НЕ гибридный: Первый (и единственный) пакет - {packet1_type_str}.")
        elif not can_use_ecc:
             logging.info(f"Гибридный режим: ECC для первого пакета невозможен/выключен. Первый пакет - {packet1_type_str}.")
        use_hybrid_ecc = False # Важно: Отключаем гибридный режим, если первый пакет Raw

    # 2. Формируем ОСТАЛЬНЫЕ пакеты (только если гибридный режим остался активен)
    if use_hybrid_ecc:
        num_raw_repeats_to_add = max(0, max_total_packets - 1)
        for _ in range(num_raw_repeats_to_add):
            bits_to_embed_list.extend(raw_payload_bits.tolist()) # Используем проверенный raw_payload_bits
            num_raw_packets_added += 1
        if num_raw_packets_added > 0:
             logging.info(f"Гибридный режим: Добавлено {num_raw_packets_added} Raw payload пакетов ({payload_len_bits} бит каждый).")

    total_packets_actual = 1 + num_raw_packets_added # Общее число подготовленных пакетов
    total_bits_to_embed = len(bits_to_embed_list)

    if total_bits_to_embed == 0:
        logging.error("Нет бит для встраивания после формирования пакетов.")
        return frames[:]

    # 3. Определяем, сколько пар кадров нужно и обрезаем биты
    if bits_per_pair <= 0: # Защита от деления на ноль
         logging.error(f"Некорректное значение bits_per_pair: {bits_per_pair}")
         return frames[:]
    pairs_needed = ceil(total_bits_to_embed / bits_per_pair)
    pairs_to_process = min(total_pairs, pairs_needed)

    # Создаем финальный массив бит нужной длины
    bits_flat_final = np.array(bits_to_embed_list[:pairs_to_process * bits_per_pair], dtype=np.uint8)
    actual_bits_embedded = len(bits_flat_final)

    # Логирование финальных параметров
    logging.info(f"Подготовка к встраиванию:")
    logging.info(f"  Режим: {'Гибридный' if use_hybrid_ecc else ('Только ECC' if can_use_ecc and use_ecc_for_first else 'Только Raw')}")
    logging.info(f"  Целевое число пакетов: {total_packets_actual} ({packet1_type_str} + {num_raw_packets_added} Raw)")
    logging.info(f"  Всего бит подготовлено: {total_bits_to_embed}")
    logging.info(f"  Доступно пар кадров: {total_pairs}, Требуется пар: {pairs_needed}, Будет обработано пар: {pairs_to_process}")
    logging.info(f"  Фактически будет встроено бит: {actual_bits_embedded}")
    if actual_bits_embedded < total_bits_to_embed:
         logging.warning(f"Не хватает пар кадров для встраивания всех подготовленных бит! Встроено только {actual_bits_embedded}.")

    # --- Подготовка аргументов для батчей ---
    start_time_embed_loop = time.time()
    watermarked_frames=frames[:] # Создаем копию списка оригинальных кадров
    rings_log: Dict[int, List[int]] = {} # Словарь для логирования колец
    pc, ec, uc = 0, 0, 0 # Счетчики: processed_ok, errors, updated_frames
    skipped_pairs = 0
    all_pairs_args = [] # Список словарей аргументов для воркеров

    for pair_idx in range(pairs_to_process):
        i1 = 2 * pair_idx
        i2 = i1 + 1
        # Проверка валидности кадров
        if i2 >= num_frames or frames[i1] is None or frames[i2] is None:
            skipped_pairs += 1
            logging.debug(f"Пропуск пары {pair_idx}: невалидные индексы/кадры {i1} или {i2}")
            continue

        # Извлечение бит для текущей пары
        sbi = pair_idx * bits_per_pair
        ebi = sbi + bits_per_pair
        # Проверка, что мы не вышли за пределы подготовленных бит
        if sbi >= len(bits_flat_final):
             logging.warning(f"Закончились биты для встраивания на паре {pair_idx}. Прерывание подготовки задач.")
             break # Если биты кончились, выходим из цикла подготовки задач
        # Обрезаем конечный индекс, если это последняя порция бит
        if ebi > len(bits_flat_final):
            ebi = len(bits_flat_final)

        cb = bits_flat_final[sbi:ebi].tolist()

        # Пропускаем, если по какой-то причине нет бит для этой пары
        if len(cb) == 0:
             logging.warning(f"Пропуск пары {pair_idx}: нет бит для встраивания (sbi={sbi}, ebi={ebi}).")
             continue

        # Формируем словарь аргументов для воркера
        args = {'pair_idx': pair_idx, 'frame1': frames[i1], 'frame2': frames[i2], 'bits': cb,
                'n_rings': n_rings, 'num_rings_to_use': num_rings_to_use, 'candidate_pool_size': candidate_pool_size,
                'frame_number': i1, 'use_perceptual_masking': use_perceptual_masking,
                'embed_component': embed_component,
                'device': device, 'dtcwt_fwd': dtcwt_fwd, 'dtcwt_inv': dtcwt_inv} # Передаем объекты PyTorch
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0:
        logging.warning(f"Пропущено {skipped_pairs} пар при подготовке задач из-за невалидных кадров.")
    if num_valid_tasks == 0:
        logging.error("Нет валидных задач для встраивания после подготовки.")
        return watermarked_frames # Возвращаем исходные кадры

    # --- Запуск ThreadPoolExecutor ---
    # Определяем количество воркеров
    num_workers = max_workers if max_workers is not None and max_workers > 0 else (os.cpu_count() or 1)
    # Рассчитываем размер батча
    # Уменьшение делителя (например, до num_workers) может увеличить размер батча
    batch_size = max(1, ceil(num_valid_tasks / num_workers));
    num_batches = ceil(num_valid_tasks / batch_size)
    # Создаем список батчей
    batched_args_list = [all_pairs_args[i:i + batch_size] for i in range(0, num_valid_tasks, batch_size)]
    # Убираем пустые батчи (на всякий случай)
    batched_args_list = [batch for batch in batched_args_list if batch]
    actual_num_batches = len(batched_args_list)

    logging.info(f"Запуск {actual_num_batches} батчей ({num_valid_tasks} пар) в ThreadPool (mw={num_workers}, batch≈{batch_size})...")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Создаем словарь {future: batch_index} для отслеживания
            future_to_batch_idx = {executor.submit(_embed_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}

            # Обрабатываем результаты по мере завершения
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                original_batch = batched_args_list[batch_idx]
                try:
                    batch_results = future.result() # Получаем список результатов из воркера батча
                    # Проверяем консистентность результата
                    if not isinstance(batch_results, list) or len(batch_results) != len(original_batch):
                        logging.error(f"Размер результата батча {batch_idx} не совпадает с размером задачи! ({len(batch_results) if isinstance(batch_results, list) else 'N/A'} vs {len(original_batch)})")
                        ec += len(original_batch) # Считаем все пары в батче ошибками
                        continue

                    # Обрабатываем результат каждой пары в батче
                    for i, single_res in enumerate(batch_results):
                        original_args = original_batch[i]
                        pair_idx = original_args.get('pair_idx', -1) # Получаем pair_idx из исходных аргументов

                        if pair_idx == -1:
                             logging.error(f"Не найден pair_idx в результате из батча {batch_idx}, элемент {i}")
                             ec += 1
                             continue

                        # Проверяем структуру результата для одной пары
                        if isinstance(single_res, tuple) and len(single_res) == 4:
                            fn_res, mod_f1, mod_f2, sel_rings = single_res
                            i1 = 2 * pair_idx; i2 = i1 + 1

                            # Логируем выбранные кольца
                            if isinstance(sel_rings, list): # Проверяем тип перед записью
                                rings_log[pair_idx] = sel_rings
                            else:
                                logging.warning(f"Некорректный формат selected_rings для пары {pair_idx}")

                            # Обновляем кадры, если встраивание успешно
                            if isinstance(mod_f1, np.ndarray) and isinstance(mod_f2, np.ndarray):
                                # Проверка на случай, если индексы выходят за пределы списка
                                if i1 < len(watermarked_frames):
                                    watermarked_frames[i1] = mod_f1
                                    uc += 1 # Считаем обновленные кадры
                                else:
                                     logging.error(f"Индекс кадра {i1} (из пары {pair_idx}) выходит за пределы списка watermarked_frames!")
                                if i2 < len(watermarked_frames):
                                    watermarked_frames[i2] = mod_f2
                                    uc += 1
                                else:
                                    logging.error(f"Индекс кадра {i2} (из пары {pair_idx}) выходит за пределы списка watermarked_frames!")
                                pc += 1 # Считаем успешно обработанные пары
                            else:
                                # Встраивание для пары не удалось (вернуло None)
                                logging.warning(f"Встраивание не удалось для пары {pair_idx} (функция вернула None для кадров).")
                                ec += 1 # Считаем как ошибку пары
                        else:
                            # Некорректная структура результата от _embed_single_pair_task
                            logging.warning(f"Некорректная структура результата для пары {pair_idx} в батче {batch_idx}. Получено: {type(single_res)}")
                            ec += 1
                except Exception as e:
                    # Ошибка при получении результата future.result()
                    failed_pairs_count = len(original_batch)
                    logging.error(f"Ошибка обработки батча {batch_idx} (future.result()): {e}", exc_info=True)
                    ec += failed_pairs_count # Считаем все пары ошибками

    except Exception as e: # Ошибка самого ThreadPoolExecutor
        logging.critical(f"Критическая ошибка ThreadPoolExecutor: {e}", exc_info=True)
        return frames[:] # Возвращаем исходные кадры

    # --- Завершение и запись логов колец ---
    processing_time = time.time() - start_time_embed_loop
    logging.info(f"Обработка пар в потоках завершена за {processing_time:.2f} сек.")
    logging.info(f"Итог: Обработано пар OK: {pc}, Ошибки/Пропуски пар: {ec + skipped_pairs}, Обновлено кадров: {uc}.")

    # Запись лога выбранных колец
    if rings_log:
        try:
            # Конвертируем ключи в строки для JSON
            serializable_log = {str(k): v for k, v in rings_log.items()}
            filepath = SELECTED_RINGS_FILE
            with open(filepath, 'w', encoding='utf-8') as f:
                 json.dump(serializable_log, f, indent=4)
            logging.info(f"Лог выбранных колец сохранен: {filepath}")
        except TypeError as e_json:
             logging.error(f"Ошибка сериализации лога колец в JSON: {e_json}")
        except IOError as e_io:
             logging.error(f"Ошибка записи лога колец в файл {SELECTED_RINGS_FILE}: {e_io}")
        except Exception as e:
             logging.error(f"Неожиданная ошибка при сохранении лога колец: {e}", exc_info=True)
    else:
        logging.warning("Лог выбранных колец пуст или не был сгенерирован.")

    total_function_time = time.time() - start_time_embed_loop # Пересчитываем время только на обработку
    logging.info(f"Функция embed_watermark_in_video завершена. Время обработки пар: {total_function_time:.2f}s.")
    return watermarked_frames

# --- ИЗМЕНЕННАЯ main ---
def main():
    start_time_main = time.time()

    # --- Инициализация и проверки зависимостей ---
    if not PYAV_AVAILABLE:
        print("ERROR: PyAV library is required. Install: pip install av")
        logging.critical("PyAV library is required but not found.")
        return
    if not PYTORCH_WAVELETS_AVAILABLE:
        print("ERROR: pytorch_wavelets library required.")
        logging.critical("pytorch_wavelets library required but not found.")
        return
    if not TORCH_DCT_AVAILABLE:
        print("ERROR: torch-dct library required.")
        logging.critical("torch-dct library required but not found.")
        return
    if USE_ECC and not GALOIS_AVAILABLE:
        print("\nWARNING: ECC requested but galois library is unavailable or failed tests.")
        logging.warning("ECC requested but galois unavailable/failed.")

    # --- Настройка PyTorch ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        try:
            # Попытка выделить немного памяти для проверки доступности CUDA
            _ = torch.tensor([1.0], device=device)
            logging.info("CUDA device check successful.")
        except RuntimeError as e:
            logging.error(f"CUDA device error: {e}. Falling back to CPU.", exc_info=True)
            print(f"\nWARNING: CUDA error ({e}), falling back to CPU.")
            device = torch.device("cpu")
    else:
        logging.info("Using CPU.")

    # --- Создание экземпляров DTCWT ---
    dtcwt_fwd: Optional[DTCWTForward] = None
    dtcwt_inv: Optional[DTCWTInverse] = None
    try:
        # Используем параметры из вашего оригинального кода
        dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
        dtcwt_inv = DTCWTInverse(biort='near_sym_a', qshift='qshift_a').to(device)
        logging.info("PyTorch DTCWTForward and DTCWTInverse instances created.")
    except Exception as e:
        logging.critical(f"Failed to initialize pytorch-wavelets DTCWT instances: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Failed to initialize DTCWT: {e}")
        return

    # --- Определение имен файлов ---
    input_video = "test_video.mp4" # Или ваш входной файл
    # Имя выходного файла зависит от параметра BCH_T для наглядности
    base_output_filename = f"watermarked_pyav_hybrid_t{BCH_T}"
    # Используем глобальное расширение OUTPUT_EXTENSION (например, '.mp4')
    output_video = base_output_filename + OUTPUT_EXTENSION

    logging.info(f"--- Starting Embedding Main Process (PyAV I/O) ---")
    logging.info(f"Input video: '{input_video}'")
    logging.info(f"Output video: '{output_video}'")

    # --- ШАГ 1: Чтение видео и аудио с PyAV ---
    logging.info("Attempting to read video and audio using PyAV...")
    video_frames_bgr, audio_packets, input_metadata = read_video_pyav(input_video)

    if video_frames_bgr is None or input_metadata is None:
        logging.critical("Video read failed using PyAV. Aborting.")
        print("ERROR: Failed to read video file using PyAV.")
        return
    logging.info(f"Successfully read {len(video_frames_bgr)} video frames.")
    if input_metadata.get('has_audio'):
        logging.info(f"Successfully collected {len(audio_packets if audio_packets else [])} audio packets.")
    else:
        logging.info("No audio stream detected or processed in the input.")

    # --- Проверка достаточного количества кадров ---
    if len(video_frames_bgr) < 2:
        logging.critical("Not enough video frames (< 2) for processing. Aborting.")
        print("ERROR: Not enough video frames read.")
        return

    # --- Получение FPS для использования ---
    # Берем FPS из метаданных, если доступно, иначе используем глобальное значение по умолчанию
    fps_to_use = input_metadata.get('fps', float(FPS))
    if fps_to_use <= 0:
        logging.warning(f"Invalid FPS detected ({fps_to_use}). Using default FPS: {float(FPS)}")
        fps_to_use = float(FPS)
    logging.info(f"Using FPS for processing/writing: {fps_to_use:.2f}")

    # --- Генерация Payload ID и сохранение ---
    payload_len_bits = PAYLOAD_LEN_BYTES * 8
    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES)
    original_id_hex = original_id_bytes.hex()
    logging.info(f"Generated Payload ID ({payload_len_bits} bit, Hex): {original_id_hex}")
    try:
        with open(ORIGINAL_WATERMARK_FILE, "w", encoding='utf-8') as f:
            f.write(original_id_hex)
        logging.info(f"Original ID saved to: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e:
        logging.error(f"Failed to save original ID to '{ORIGINAL_WATERMARK_FILE}': {e}")
        print(f"WARNING: Failed to save original ID file: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving original ID: {e}", exc_info=True)


    # --- ШАГ 2: Встраивание ЦВЗ в видеокадры ---
    # Эта функция ожидает список NumPy BGR кадров, которые мы получили от read_video_pyav
    logging.info("Starting watermark embedding process...")
    watermarked_frames = embed_watermark_in_video(
        frames=video_frames_bgr, # Используем прочитанные BGR кадры
        payload_id_bytes=original_id_bytes,
        # Параметры гибридного ECC/Raw режима
        use_hybrid_ecc=True,       # Используем гибридный режим
        max_total_packets=15,      # Макс. пакетов (1 ECC + 14 Raw)
        use_ecc_for_first=USE_ECC, # Использовать ли ECC для первого (из глоб. переменной)
        bch_code=BCH_CODE_OBJECT,  # Передаем объект Galois BCH
        # Параметры PyTorch
        device=device,             # Устройство (CPU/GPU)
        dtcwt_fwd=dtcwt_fwd,       # Экземпляр прямого DTCWT
        dtcwt_inv=dtcwt_inv,       # Экземпляр обратного DTCWT
        # Параметры самого алгоритма
        n_rings=N_RINGS,
        num_rings_to_use=NUM_RINGS_TO_USE,
        bits_per_pair=BITS_PER_PAIR,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        fps=fps_to_use,             # FPS используется внутри для логирования? Если нет - можно убрать
        max_workers=MAX_WORKERS,    # Кол-во потоков для ThreadPoolExecutor
        use_perceptual_masking=USE_PERCEPTUAL_MASKING,
        embed_component=EMBED_COMPONENT
    )

    if watermarked_frames is None:
        logging.error("Watermark embedding function returned None. Aborting write.")
        print("ERROR: Watermark embedding process failed.")
        return # Не можем продолжать без обработанных кадров

    if len(watermarked_frames) != len(video_frames_bgr):
         logging.error(f"Frame count mismatch after embedding: Expected {len(video_frames_bgr)}, Got {len(watermarked_frames)}. Aborting write.")
         print("ERROR: Frame count mismatch after embedding process.")
         return

    logging.info("Watermark embedding process completed.")

    # --- ШАГ 3: Запись обработанного видео и оригинального аудио с PyAV ---
    write_success = False
    logging.info("Attempting to write final video using PyAV...")

    # Определяем целевой кодек на основе входного (если хотим соответствовать)
    # Пример: если вход был h264, использовать libx264. Если hevc, то libx265.
    # Для простоты пока оставим libx264 как основной CPU кодек.
    target_video_codec = 'libx264'
    # Можно добавить логику выбора:
    # input_codec = input_metadata.get('video_codec')
    # if input_codec == 'hevc':
    #     target_video_codec = 'libx265'
    # elif input_codec == 'mpeg4':
    #      target_video_codec = 'mpeg4' # Используем стандартный mpeg4 кодер FFMpeg
    # else: # По умолчанию h264
    #      target_video_codec = 'libx264'

    write_success = write_video_pyav(
        processed_video_frames=watermarked_frames,
        original_audio_packets=audio_packets if audio_packets else [], # Передаем пустой список, если аудио не было
        input_metadata=input_metadata, # Передаем метаданные для настроек
        output_path=output_video,
        target_video_codec=target_video_codec, # Можно поменять на 'h264_nvenc' и т.д., если настроено HW
        video_quality_crf=23,          # Используем CRF 23 (хорошее качество) если битрейт не задан
        audio_codec_action='copy'      # Копируем аудиодорожку без перекодирования
    )

    if write_success:
        logging.info(f"Watermarked video with audio (if present) saved successfully to: {output_video}")
        print(f"\nSuccessfully wrote watermarked video to: {output_video}")
    else:
        logging.error(f"Writing video using PyAV failed for path: {output_video}")
        print("\nERROR: Failed to write the final video file.")

    # --- Завершение ---
    logging.info(f"--- Embedding Main Process Finished (PyAV I/O) ---")
    total_time_main = time.time() - start_time_main
    logging.info(f"--- Total Main Execution Time: {total_time_main:.2f} sec ---")
    print(f"\nEmbedding process finished in {total_time_main:.2f} seconds.")
    print(f"Log file: {LOG_FILENAME}")
    print(f"Original ID file: {ORIGINAL_WATERMARK_FILE}")
    print(f"Selected Rings log file: {SELECTED_RINGS_FILE}")

    if write_success:
        print(f"\nRun extractor_pytorch.py on '{output_video}' to extract the watermark.")
    else:
         print("\nExtraction cannot be performed as the output file was not written successfully.")

# --- Точка Входа (__name__ == "__main__") ---
if __name__ == "__main__":
    # --- Инициализация и проверки зависимостей ---
    # Эти проверки можно вынести в отдельную функцию или оставить здесь
    if not PYAV_AVAILABLE: print("\nERROR: PyAV library is required. Install: pip install av"); sys.exit(1)
    if not PYTORCH_WAVELETS_AVAILABLE: print("\nERROR: pytorch_wavelets library required."); sys.exit(1)
    if not TORCH_DCT_AVAILABLE: print("\nERROR: torch-dct library required."); sys.exit(1)
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois library is unavailable or failed tests.")
    # Можно добавить проверку доступности входного файла здесь, если main() вызывается только один раз

    # --- Настройка профилирования ---
    DO_PROFILING = False  # Установите True для включения cProfile
    profiler = None
    if DO_PROFILING:
        profiler = cProfile.Profile()
        profiler.enable()
        logging.info("cProfile profiling enabled.")
        print("cProfile profiling enabled.")

    final_exit_code = 0
    try:
        # --- Запуск основной логики ---
        main()

    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found: {e}")
        logging.error(f"File not found error during main execution: {e}", exc_info=True)
        final_exit_code = 1
    except FFmpegError  as e:  # <--- ИСПРАВЛЕНО: Используем импортированный AVError
        print(f"\nERROR: PyAV/FFmpeg error occurred: {e}")
        logging.critical(f"PyAV/FFmpeg error during main execution: {e}", exc_info=True)
        final_exit_code = 1
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nERROR: CUDA out of memory: {e}. Try reducing resolution or batch size.")
        logging.critical(f"CUDA out of memory error: {e}", exc_info=True)
        final_exit_code = 1
    except Exception as e:
        logging.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: An unexpected error occurred: {e}")
        print(f"Please check the log file for details: {LOG_FILENAME}")
        final_exit_code = 1
    finally:
        # --- Сохранение результатов профилирования (если включено) ---
        if DO_PROFILING and profiler is not None:
            profiler.disable()
            logging.info("cProfile profiling disabled.")
            print("\n--- cProfile Stats (Top 30 Cumulative Time) ---")
            stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
            stats.print_stats(30)
            print("-------------------------------------------------")

            # Имя файла для сохранения статистики профилирования
            profile_stats_file = f"profile_embed_pyav_hybrid_t{BCH_T}.prof" # Используем .prof расширение
            try:
                stats.dump_stats(profile_stats_file)
                logging.info(f"Profiling statistics saved to: {profile_stats_file}")
                print(f"Profiling statistics saved to: {profile_stats_file}")
                print("You can view stats later using tools like 'snakeviz'.")
                # Опционально: сохранить читаемый текстовый отчет
                # txt_profile_file = f"profile_embed_pyav_hybrid_t{BCH_T}.txt"
                # with open(txt_profile_file, "w", encoding='utf-8') as f_txt:
                #     stats_txt = pstats.Stats(profiler, stream=f_txt)
                #     stats_txt.strip_dirs().sort_stats("cumulative").print_stats()
                # logging.info(f"Readable profiling stats saved to: {txt_profile_file}")

            except Exception as e_prof_save:
                logging.error(f"Failed to save profiling stats: {e_prof_save}")
                print(f"WARNING: Failed to save profiling stats: {e_prof_save}")

    # --- Завершение работы скрипта ---
    print(f"\nScript finished with exit code {final_exit_code}.")
    logging.info(f"Script finished with exit code {final_exit_code}.")
    sys.exit(final_exit_code)

