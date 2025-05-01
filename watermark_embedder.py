# Файл: embedder_pytorch_wavelets.py (ЧАСТЬ 1)
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
from PIL import Image # Оставлен на всякий случай, но пока не используется напрямую
# line_profiler убран из примера, добавьте если нужно
# from line_profiler import profile
from scipy.fftpack import dct as scipy_dct, idct as scipy_idct # Переименовал для ясности
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
import uuid
from math import ceil
import cProfile
import pstats

# --- Глобальные Параметры ---
LAMBDA_PARAM: float = 0.05
ALPHA_MIN: float = 1.13
ALPHA_MAX: float = 1.28
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2 # Cb
USE_PERCEPTUAL_MASKING: bool = True
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection' # Метод реализуется в коде
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 9 # Оставляем T=9
FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed_pytorch.log' # Изменил имя лога
OUTPUT_CODEC: str = 'mp4v' # Или другой
OUTPUT_EXTENSION: str = '.mp4' # Или другой
SELECTED_RINGS_FILE: str = 'selected_rings_embed_pytorch.json' # Изменил имя
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS: Optional[int] = 14 # Оставляем
# MAX_TOTAL_PACKETS используется как аргумент функции

# --- Инициализация Галуа (с t=9, k=187) ---
BCH_CODE_OBJECT: Optional['galois.BCH'] = None # Аннотация в кавычках для отложенного импорта
GALOIS_AVAILABLE = False
try:
    import galois
    logging.info("galois: импортирован.")
    _test_bch_ok = False; _test_decode_ok = False
    try:
        _test_m = BCH_M
        _test_t = BCH_T
        _test_n = (1 << _test_m) - 1
        _test_d = 2 * _test_t + 1

        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)

        # Определяем ожидаемое k на основе t
        if _test_t == 5: expected_k = 215
        elif _test_t == 7: expected_k = 201
        elif _test_t == 9: expected_k = 187 # <-- Для t=9
        elif _test_t == 11: expected_k = 173
        elif _test_t == 15: expected_k = 131
        else:
             logging.error(f"Неизвестное ожидаемое k для t={_test_t}")
             expected_k = -1

        if expected_k != -1 and _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
             logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
             _test_bch_ok = True; BCH_CODE_OBJECT = _test_bch_galois
        else:
             logging.error(f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t={_test_bch_galois.t}, k={_test_bch_galois.k}.")
             _test_bch_ok = False; BCH_CODE_OBJECT = None

        # Тест декодирования (если инициализация прошла)
        if _test_bch_ok and BCH_CODE_OBJECT is not None:
            _n_bits = BCH_CODE_OBJECT.n
            _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)
            GF2 = galois.GF(2); _dummy_cw_vec = GF2(_dummy_cw_bits)
            # Подавляем вывод ошибок декодера в тесте
            _msg, _flips = BCH_CODE_OBJECT.decode(_dummy_cw_vec, errors=True)
            if _flips is not None: # Проверяем, что декодирование не вызвало исключение
                 logging.info(f"galois: decode() test OK (flips={_flips}).")
                 _test_decode_ok = True
            else:
                 # decode() может вернуть None для _flips если ошибок 0, проверим и так
                 _test_decode_ok = True # Считаем тест пройденным, если исключения не было
                 logging.info("galois: decode() test potentially OK (flips is None/0).")
    except ValueError as ve:
         logging.error(f"galois: ОШИБКА ValueError при инициализации BCH: {ve}")
         BCH_CODE_OBJECT = None; _test_bch_ok = False
    except Exception as test_err:
         logging.error(f"galois: ОШИБКА теста инициализации/декодирования: {test_err}", exc_info=True)
         BCH_CODE_OBJECT = None; _test_bch_ok = False

    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if GALOIS_AVAILABLE: logging.info("galois: Тесты пройдены, объект BCH доступен.")
    else: logging.warning("galois: Тесты НЕ ПРОЙДЕНЫ или объект BCH не создан.")

except ImportError:
    GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None
    logging.info("galois library not found.")
except Exception as import_err:
    GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None
    logging.error(f"galois: Ошибка импорта: {import_err}", exc_info=True)


# --- Настройка логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG) # Раскомментировать для детального лога

# --- Логирование конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Встраивания (PyTorch Wavelets) ---")
logging.info(f"PyTorch Wavelets Доступно: {PYTORCH_WAVELETS_AVAILABLE}")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Payload: {PAYLOAD_LEN_BYTES * 8}bit, ECC for 1st: {effective_use_ecc} (Galois BCH m={BCH_M}, t={BCH_T})")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(f"Маскировка: {USE_PERCEPTUAL_MASKING} (Lambda={LAMBDA_PARAM}), Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS or 'default'}) с батчингом.")
if USE_ECC and not GALOIS_AVAILABLE: logging.warning("ECC вкл, но galois недоступна/не работает! Первый пакет будет Raw.")
elif not USE_ECC: logging.info("ECC выкл для первого пакета.")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG', 'DIVX']: logging.warning(f"Используется кодек с потерями '{OUTPUT_CODEC}'.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE:
    logging.error(f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) > CANDIDATE_POOL_SIZE ({CANDIDATE_POOL_SIZE})! Исправлено.")
    NUM_RINGS_TO_USE = CANDIDATE_POOL_SIZE
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) != BITS_PER_PAIR ({BITS_PER_PAIR}).")

# --- Базовые Функции (с изменениями для PyTorch) ---

def dct_1d(s: np.ndarray) -> np.ndarray:
    """1D DCT используя SciPy (для NumPy массивов)."""
    return scipy_dct(s, type=2, norm='ortho')

def idct_1d(c: np.ndarray) -> np.ndarray:
    """1D IDCT используя SciPy (для NumPy массивов)."""
    return scipy_idct(c, type=2, norm='ortho')

# --- НОВАЯ функция обертка для PyTorch DTCWT Forward ---
def dtcwt_pytorch_forward(yp_tensor: torch.Tensor, xfm: DTCWTForward, device: torch.device, fn: int = -1) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Применяет прямое DTCWT PyTorch к одному каналу (2D тензору)."""
    if not PYTORCH_WAVELETS_AVAILABLE: # Проверка доступности библиотеки
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
        yp_tensor = yp_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32) # -> (1, 1, H, W)
        xfm = xfm.to(device)

        with torch.no_grad(): # Отключаем расчет градиентов для экономии памяти/скорости
            Yl, Yh = xfm(yp_tensor) # Yh - список

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
def dtcwt_pytorch_inverse(Yl: torch.Tensor, Yh: List[torch.Tensor], ifm: DTCWTInverse, device: torch.device, target_shape: Tuple[int, int], fn: int = -1) -> Optional[np.ndarray]:
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
         Yh = [h.to(device) for h in Yh if h is not None and h.numel() > 0] # Фильтруем пустые/None
         ifm = ifm.to(device)

         # Убедимся, что Yh не стал пустым после фильтрации
         if not Yh:
             logging.error(f"[Frame:{fn}] Yh list is empty after filtering Nones/empty tensors.")
             return None

         with torch.no_grad(): # Отключаем градиенты
             reconstructed_X_tensor = ifm((Yl, Yh))

         # Убираем batch и channel измерения
         if reconstructed_X_tensor.dim() == 4 and reconstructed_X_tensor.shape[0] == 1 and reconstructed_X_tensor.shape[1] == 1:
              reconstructed_X_tensor = reconstructed_X_tensor.squeeze(0).squeeze(0) # (H, W)
         elif reconstructed_X_tensor.dim() != 2:
              logging.error(f"[Frame:{fn}] Unexpected output dimension from inverse: {reconstructed_X_tensor.dim()}")
              return None

         # logging.debug(f"[Frame:{fn}] PyTorch DTCWT INV done. Output shape: {reconstructed_X_tensor.shape}")

         # Обрезаем до нужного размера
         current_h, current_w = reconstructed_X_tensor.shape
         target_h, target_w = target_shape
         if current_h > target_h or current_w > target_w:
             logging.warning(f"[Frame:{fn}] Inverse result shape {reconstructed_X_tensor.shape} > target {target_shape}. Cropping.")
             # Убедимся, что не обрезаем до нуля или отрицательного размера
             if target_h > 0 and target_w > 0:
                  reconstructed_X_tensor = reconstructed_X_tensor[:target_h, :target_w]
             else:
                   logging.error(f"[Frame:{fn}] Invalid target shape for cropping: {target_shape}")
                   return None
         elif current_h < target_h or current_w < target_w:
              logging.warning(f"[Frame:{fn}] Inverse result shape {reconstructed_X_tensor.shape} < target {target_shape}. Padding might be needed if this causes issues.")
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
        logging.error(f"[Frame:{fn}] Invalid input for ring_division (expected 2D torch.Tensor). Got {type(lp_tensor)} with ndim {lp_tensor.ndim if hasattr(lp_tensor, 'ndim') else 'N/A'}")
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
                                indexing='ij') # indexing='ij' важно для H, W порядка

        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
        distances = torch.sqrt((rr - center_r)**2 + (cc - center_c)**2)

        min_dist, max_dist = torch.tensor(0.0, device=device), torch.max(distances)

        # Границы колец
        if max_dist < 1e-9: # Избегаем деления на ноль или проблем с linspace
             logging.warning(f"[Frame:{fn}] Max distance in ring division is near zero ({max_dist}).")
             # Все пиксели попадут в первое кольцо
             ring_bins = torch.tensor([0.0, max_dist + 1e-6] + [max_dist + 1e-6] * (nr-1), device=device)
        else:
             ring_bins = torch.linspace(min_dist.item(), (max_dist + 1e-6).item(), nr + 1, device=device)

        # Назначение индексов кольца
        # Используем маски для большей ясности и избежания проблем с bucketize
        ring_indices = torch.zeros_like(distances, dtype=torch.long) - 1 # Инициализируем -1
        for i in range(nr):
             lower_bound = ring_bins[i]
             upper_bound = ring_bins[i+1]
             # Маска для текущего кольца
             # Включаем нижнюю границу, исключаем верхнюю (кроме последнего кольца)
             if i < nr - 1:
                  mask = (distances >= lower_bound) & (distances < upper_bound)
             else: # Последнее кольцо включает верхнюю границу
                  mask = (distances >= lower_bound) & (distances <= upper_bound) # <= для max_dist
             ring_indices[mask] = i

        # Убедимся, что центр в первом кольце (если вдруг не попал из-за точности float)
        ring_indices[distances < ring_bins[1]] = 0

        rings: List[Optional[torch.Tensor]] = [None] * nr
        for rdx in range(nr):
            # Находим координаты (индексы) пикселей для кольца rdx
            coords_tensor = torch.nonzero(ring_indices == rdx, as_tuple=False) # -> shape (N_pixels, 2)
            if coords_tensor.shape[0] > 0:
                rings[rdx] = coords_tensor.long() # Сохраняем как LongTensor
            else: # Кольцо пустое
               logging.debug(f"[Frame:{fn}] Ring {rdx} is empty.")

        return rings
    except Exception as e:
         logging.error(f"Ring division PyTorch error Frame {fn}: {e}", exc_info=True)
         return [None] * nr


# --- calculate_entropies, compute_adaptive_alpha_entropy - остаются на NumPy ---
# Они принимают 1D NumPy массив значений кольца
def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    eps=1e-12; shannon_entropy=0.; collision_entropy=0. # Ваша старая формула давала ee

    if rv.size > 0:
        rv_processed = np.clip(rv.copy(), 0.0, 1.0) # Работаем с копией и клиппингом
        if np.all(rv_processed == rv_processed[0]): return 0.0, 0.0 # Энтропия константы 0

        hist, _ = np.histogram(rv_processed, bins=256, range=(0., 1.), density=False)
        total_count = rv_processed.size
        if total_count > 0:
            probabilities = hist / total_count
            p = probabilities[probabilities > eps] # Убираем нулевые вероятности
            if p.size > 0:
                shannon_entropy = -np.sum(p * np.log2(p))
                # Используем Реньи 2-го порядка как collision entropy
                # collision_entropy = -np.log2(np.sum(p**2)) if np.sum(p**2) > eps else 0.0
                # Возвращаем вашу старую ee для совместимости, если она нужна где-то еще
                ee = -np.sum(p*np.exp(1.-p))
                collision_entropy = ee # Присваиваем ee, чтобы сигнатура не менялась
    return shannon_entropy, collision_entropy # Возвращаем (ve, ee)

def compute_adaptive_alpha_entropy(rv: np.ndarray, ri: int, fn: int) -> float:
    if rv.size < 10: return ALPHA_MIN # Мало данных для статистики
    # Используем shannon_entropy (первый элемент кортежа)
    ve, _ = calculate_entropies(rv, fn, ri)
    lv = np.var(rv)
    # Проверка на NaN/inf перед использованием
    if not np.isfinite(ve) or not np.isfinite(lv):
         logging.warning(f"[F:{fn}, R:{ri}] Non-finite entropy ({ve}) or variance ({lv}). Using ALPHA_MIN.")
         return ALPHA_MIN

    en = np.clip(ve / MAX_THEORETICAL_ENTROPY, 0., 1.)
    vmp = 0.005; vsc = 500
    # Использование np.exp может дать overflow если lv очень велико, добавим clip или обработку
    try:
        exp_term = np.exp(-vsc * (lv - vmp))
    except OverflowError:
        exp_term = 0.0 # Если экспонента уходит в -inf, результат 0
    tn = 1. / (1. + exp_term) if (1. + exp_term) != 0 else 1.0 # Избегаем деления на ноль

    we=.6; wt=.4
    mf=np.clip((we*en+wt*tn),0.,1.)
    fa=ALPHA_MIN+(ALPHA_MAX-ALPHA_MIN)*mf
    logging.debug(f"[F:{fn}, R:{ri}] Alpha={fa:.4f} (E={ve:.3f},V={lv:.6f})")
    return np.clip(fa, ALPHA_MIN, ALPHA_MAX)

# --- get_fixed_pseudo_random_rings - остается без изменений ---
def get_fixed_pseudo_random_rings(pi:int, nr:int, ps:int)->List[int]:
    # ... (код без изменений) ...
    if ps<=0: return []
    if ps>nr: ps=nr
    sd=str(pi).encode('utf-8'); hd=hashlib.sha256(sd).digest()
    sv=int.from_bytes(hd,'big'); prng=random.Random(sv)
    try: ci=prng.sample(range(nr),ps)
    except ValueError: ci=list(range(nr)); random.shuffle(ci); ci = ci[:ps] # Добавил shuffle и срез
    # logging.debug(f"[P:{pi}] Candidates: {ci}");
    return ci

# --- calculate_perceptual_mask - АДАПТИРОВАНА под тензоры ---
def calculate_perceptual_mask(ip_tensor: torch.Tensor, device: torch.device, fn: int = -1) -> Optional[torch.Tensor]:
    """Вычисляет перцептуальную маску для 2D тензора."""
    if not isinstance(ip_tensor, torch.Tensor) or ip_tensor.ndim != 2:
         logging.error(f"Mask error F{fn}: Input is not a 2D tensor.")
         return torch.ones_like(ip_tensor, device=device) # Возвращаем единичную маску
    try:
        # Конвертация в NumPy для OpenCV
        pf = ip_tensor.cpu().numpy().astype(np.float32)
        # Проверка на NaN/inf после конвертации
        if not np.all(np.isfinite(pf)):
             logging.warning(f"Mask error F{fn}: Input tensor contains NaN/inf.")
             return torch.ones_like(ip_tensor, device=device)

        # Операции OpenCV
        gx=cv2.Sobel(pf,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(pf,cv2.CV_32F,0,1,ksize=3)
        # Проверка на NaN/inf после Sobel
        if not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
             logging.warning(f"Mask error F{fn}: Sobel result contains NaN/inf.")
             return torch.ones_like(ip_tensor, device=device)

        gm=np.sqrt(gx**2+gy**2)
        ks=(11,11); s=5
        lm=cv2.GaussianBlur(pf,ks,s); lms=cv2.GaussianBlur(pf**2,ks,s)
        # Проверка на NaN/inf после GaussianBlur
        if not np.all(np.isfinite(lm)) or not np.all(np.isfinite(lms)):
             logging.warning(f"Mask error F{fn}: GaussianBlur result contains NaN/inf.")
             return torch.ones_like(ip_tensor, device=device)

        # np.maximum(lms-lm**2,0) - защита от отрицательных под корнем
        lv=np.sqrt(np.maximum(lms-lm**2, 0))
        if not np.all(np.isfinite(lv)): # Проверка после sqrt
             logging.warning(f"Mask error F{fn}: Local variance result contains NaN/inf.")
             # Попробуем заменить NaN/inf нулями перед np.maximum(gm, lv)
             lv = np.nan_to_num(lv, nan=0.0, posinf=0.0, neginf=0.0)
             # return torch.ones_like(ip_tensor, device=device) # Раньше возвращали единицы

        cm=np.maximum(gm,lv)
        eps=1e-9; mc=np.max(cm)

        # Проверка mc
        if not np.isfinite(mc):
             logging.warning(f"Mask error F{fn}: Max complexity (mc) is not finite.")
             return torch.ones_like(ip_tensor, device=device)

        mn=cm/(mc+eps) if mc>eps else np.zeros_like(cm)
        mask_np = np.clip(mn,0.,1.).astype(np.float32)

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
         return data_bits # Возвращаем как есть, если ECC нет
    try:
        k=bch_code.k; n=bch_code.n
        if data_bits.size > k:
            logging.error(f"ECC Error: Data size ({data_bits.size}) > k ({k})")
            return None
        pad_len = k - data_bits.size
        # Убедимся, что data_bits - это 1D массив uint8
        msg_bits = data_bits.astype(np.uint8).flatten()
        if pad_len > 0:
             msg_bits = np.pad(msg_bits, (0, pad_len), 'constant')

        GF = bch_code.field; msg_vec = GF(msg_bits); cw_vec = bch_code.encode(msg_vec)
        pkt_bits = cw_vec.view(np.ndarray).astype(np.uint8)

        if pkt_bits.size != n: # Используем if вместо assert для мягкой ошибки
             logging.error(f"ECC Error: Output packet size ({pkt_bits.size}) != n ({n})")
             return None
        logging.info(f"Galois ECC: Data({data_bits.size}b->{k}b) -> Packet({pkt_bits.size}b).")
        return pkt_bits
    except Exception as e:
        logging.error(f"Galois encode error: {e}", exc_info=True)
        return None

# Файл: embedder_pytorch_wavelets.py (ЧАСТЬ 2)

# --- ПРЕДПОЛАГАЕТСЯ, ЧТО ВЕСЬ КОД ИЗ ЧАСТИ 1 НАХОДИТСЯ ВЫШЕ ---
# ... (импорты, константы, инициализация Galois) ...
# ... (dct_1d, idct_1d, dtcwt_pytorch_forward, dtcwt_pytorch_inverse) ...
# ... (ring_division, calculate_entropies, compute_adaptive_alpha_entropy) ...
# ... (get_fixed_pseudo_random_rings, calculate_perceptual_mask, add_ecc) ...
# -------------------------------------------------------------

# --- Функция чтения видео (остается без изменений) ---
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
             logging.warning("Не удалось получить корректные размеры кадра из видео.")
             # Попробуем прочитать первый кадр, чтобы узнать размер
             ret, f = cap.read()
             if ret and f is not None:
                  h, w, _ = f.shape
                  logging.info(f"Размеры кадра определены по первому кадру: {w}x{h}")
                  frames.append(f) # Добавляем первый кадр
                  cap.set(cv2.CAP_PROP_POS_FRAMES, 1) # Сбрасываем на следующий
             else:
                  raise ValueError("Не удалось определить размеры кадра видео.")
        logging.info(f"Props: {w}x{h}@{fps:.2f} FPS ~{fc if fc > 0 else '?'} frames")

        frame_count = 0
        read_count = 0
        none_count = 0
        invalid_count = 0

        while True:
            ret, f = cap.read()
            frame_count += 1
            if not ret:
                logging.debug(f"End of video reached or read error at frame approx {frame_count}.")
                break # Выход из цикла while

            if f is None:
                none_count += 1
                logging.warning(f"Получен пустой кадр (None) на позиции {frame_count-1}")
                continue # К следующей итерации while

            # Проверяем размерность и тип внимательнее
            if f.ndim == 3 and f.shape[0] == h and f.shape[1] == w and f.dtype == np.uint8:
                frames.append(f)
                read_count += 1
            else:
                invalid_count += 1
                logging.warning(f"Пропущен невалидный кадр #{frame_count-1}. Shape: {f.shape}, Dtype: {f.dtype}, Expected: ({h},{w},3) uint8")

        logging.info(f"Read loop finished. Valid frames read: {read_count}, None frames encountered: {none_count}, Invalid frames skipped: {invalid_count}")
        if read_count == 0:
             logging.error("Не прочитано ни одного валидного кадра.")
             # raise ValueError("Не прочитано ни одного валидного кадра.") # Или вернуть пустой список

    except Exception as e:
        logging.error(f"Ошибка чтения видео: {e}", exc_info=True)
        frames=[] # Очищаем кадры при ошибке
    finally:
        if cap and cap.isOpened():
            logging.debug("Releasing video capture resource.")
            cap.release()
    return frames, fps

# --- Функция записи видео (остается без изменений) ---
# @profile # Добавьте, если нужно профилировать
def write_video(frames: List[np.ndarray], out_path: str, fps: float, codec: str = OUTPUT_CODEC):
    """Записывает список BGR NumPy кадров в видеофайл."""
    if not frames: logging.error("Нет кадров для записи."); return;
    logging.info(f"Writing video: {out_path} (FPS:{fps:.2f}, Codec:{codec})");
    writer = None
    try:
        # Находим первый валидный кадр для определения размера
        fv = next((f for f in frames if isinstance(f, np.ndarray) and f.ndim==3), None)
        if fv is None: raise ValueError("Не найдено валидных кадров для определения размера записи.")

        h, w, _ = fv.shape
        if w <= 0 or h <= 0: raise ValueError(f"Неверный размер кадра для записи: {w}x{h}")

        fourcc = cv2.VideoWriter_fourcc(*codec)
        base, _ = os.path.splitext(out_path)
        actual_out_path = base + OUTPUT_EXTENSION # Используем глобальное расширение
        writer = cv2.VideoWriter(actual_out_path, fourcc, fps, (w, h))
        final_codec = codec

        if not writer.isOpened():
            logging.error(f"Не удалось открыть VideoWriter с кодеком {codec} для файла {actual_out_path}. Проверьте кодек и путь.")
            # Попытка отката на MJPG для AVI, если исходный кодек не сработал
            fbk_codec = 'MJPG'
            fbk_ext = '.avi'
            if codec.upper() != fbk_codec and OUTPUT_EXTENSION.lower() == fbk_ext:
                 logging.warning(f"Попытка отката на кодек {fbk_codec} и расширение {fbk_ext}")
                 fourcc_fbk = cv2.VideoWriter_fourcc(*fbk_codec)
                 actual_out_path = base + fbk_ext # Меняем расширение
                 writer = cv2.VideoWriter(actual_out_path, fourcc_fbk, fps, (w, h))
                 if writer.isOpened():
                     logging.info(f"Откат на {fbk_codec} успешен.")
                     final_codec = fbk_codec
                 else:
                      raise IOError(f"Не удалось открыть VideoWriter даже с кодеком {fbk_codec}.")
            else:
                 raise IOError(f"Не удалось открыть VideoWriter с кодеком {codec}.")

        # Запись кадров
        written_count, skipped_count = 0, 0
        placeholder_frame = np.zeros((h, w, 3), dtype=np.uint8) # Для замены невалидных

        for i, f in enumerate(frames):
            # Проверяем кадр перед записью
            if isinstance(f, np.ndarray) and f.ndim == 3 and f.shape[0] == h and f.shape[1] == w and f.dtype == np.uint8:
                writer.write(f)
                written_count += 1
            else:
                # Записываем пустой кадр вместо невалидного
                logging.warning(f"Пропущен невалидный кадр #{i} при записи. Записывается пустой кадр.")
                writer.write(placeholder_frame)
                skipped_count += 1

        logging.info(f"Запись завершена ({final_codec}). Записано кадров: {written_count}, Пропущено/Заменено: {skipped_count}. Файл: {actual_out_path}")

    except Exception as e:
        logging.error(f"Ошибка записи видео: {e}", exc_info=True)
    finally:
        if writer and writer.isOpened():
             logging.debug("Releasing video writer resource.")
             writer.release()

# --- ИЗМЕНЕННАЯ embed_frame_pair для PyTorch ---
# @profile # Добавьте, если нужно профилировать
def embed_frame_pair(
        frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bits: List[int],
        selected_ring_indices: List[int], n_rings: int, frame_number: int,
        use_perceptual_masking: bool, embed_component: int,
        # --- Новые аргументы ---
        device: torch.device,
        dtcwt_fwd: DTCWTForward,
        dtcwt_inv: DTCWTInverse
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Встраивает биты в выбранные кольца пары кадров (PyTorch версия с детальным логированием).
    """
    pair_index = frame_number // 2
    prefix_base = f"[P:{pair_index}]" # Базовый префикс для логов пары

    # Проверка длины бит/колец
    if len(bits) != len(selected_ring_indices):
        if len(bits) < len(selected_ring_indices):
             logging.warning(f"{prefix_base} Fewer bits ({len(bits)}) than rings ({len(selected_ring_indices)}). Embedding only {len(bits)}.")
             selected_ring_indices = selected_ring_indices[:len(bits)]
        else:
             logging.error(f"{prefix_base} More bits than rings. Mismatch: {len(bits)} vs {len(selected_ring_indices)}.")
             return None, None
    if not bits:
        logging.debug(f"{prefix_base} No bits to embed.")
        return frame1_bgr, frame2_bgr

    logging.debug(f"{prefix_base} --- Starting Embedding Pair ---")
    try:
        # 1. Проверка и Преобразование в Тензоры
        if frame1_bgr is None or frame2_bgr is None: logging.error(f"{prefix_base} Input BGR frames are None."); return None, None
        if frame1_bgr.ndim != 3 or frame2_bgr.ndim != 3: logging.error(f"{prefix_base} Input BGR frames are not 3D."); return None, None

        f1_ycrcb_np = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb)
        f2_ycrcb_np = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2YCrCb)
        comp1_tensor = torch.from_numpy(f1_ycrcb_np[:, :, embed_component].copy()).to(device=device, dtype=torch.float32) / 255.0
        comp2_tensor = torch.from_numpy(f2_ycrcb_np[:, :, embed_component].copy()).to(device=device, dtype=torch.float32) / 255.0
        Y1_np, Cr1_np, Cb1_np = f1_ycrcb_np[:,:,0], f1_ycrcb_np[:,:,1], f1_ycrcb_np[:,:,2]
        Y2_np, Cr2_np, Cb2_np = f2_ycrcb_np[:,:,0], f2_ycrcb_np[:,:,1], f2_ycrcb_np[:,:,2]
        target_shape_hw = (frame1_bgr.shape[0], frame1_bgr.shape[1])
        logging.debug(f"{prefix_base} Input tensors prepared. Shape: {comp1_tensor.shape}")

        # 2. Прямое DTCWT (PyTorch)
        Yl_t, Yh_t = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, frame_number)
        Yl_t1, Yh_t1 = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, frame_number + 1)
        if Yl_t is None or Yh_t is None or Yl_t1 is None or Yh_t1 is None: logging.error(f"{prefix_base} Forward DTCWT failed."); return None, None
        if Yl_t.dim() > 2: Yl_t = Yl_t.squeeze()
        if Yl_t1.dim() > 2: Yl_t1 = Yl_t1.squeeze()
        logging.debug(f"{prefix_base} Forward DTCWT done. Yl shape: {Yl_t.shape}")

        # 3. Модификация Lowpass (Yl_t, Yl_t1)
        ring_coords_t = ring_division(Yl_t, n_rings, frame_number)
        ring_coords_t1 = ring_division(Yl_t1, n_rings, frame_number + 1)
        if ring_coords_t is None or ring_coords_t1 is None: logging.error(f"{prefix_base} Ring division failed."); return None, None
        logging.debug(f"{prefix_base} Ring division done.")

        # Расчет перцептуальной маски
        perceptual_mask_tensor = torch.ones_like(Yl_t, device=device)
        if use_perceptual_masking:
            mask_full_res_tensor = calculate_perceptual_mask(comp1_tensor, device, frame_number)
            if mask_full_res_tensor is not None:
                if mask_full_res_tensor.shape != Yl_t.shape:
                     try: perceptual_mask_tensor = F.interpolate(mask_full_res_tensor.unsqueeze(0).unsqueeze(0), size=Yl_t.shape, mode='bilinear', align_corners=False).squeeze().to(device=device, dtype=torch.float32)
                     except Exception as e_interp: logging.error(f"Mask interpolation error P:{pair_index}: {e_interp}", exc_info=True); perceptual_mask_tensor = torch.ones_like(Yl_t, device=device)
                else: perceptual_mask_tensor = mask_full_res_tensor.to(device=device, dtype=torch.float32)
            logging.debug(f"{prefix_base} Perceptual mask calculated. Min: {torch.min(perceptual_mask_tensor):.4f}, Max: {torch.max(perceptual_mask_tensor):.4f}")

        modifications_count = 0
        Yl_t_mod = Yl_t.clone(); Yl_t1_mod = Yl_t1.clone()

        # --- Цикл по кольцам ---
        logging.debug(f"{prefix_base} --- Start Ring Loop (Embedding {len(bits)} bits) ---")
        for ring_idx, bit_to_embed in zip(selected_ring_indices, bits):
            prefix = f"[P:{pair_index} R:{ring_idx}]" # Префикс для кольца
            logging.debug(f"{prefix} Processing...")
            if not (0 <= ring_idx < n_rings and ring_idx < len(ring_coords_t) and ring_idx < len(ring_coords_t1)): logging.warning(f"{prefix} Invalid ring index. Skipping."); continue
            coords1_tensor = ring_coords_t[ring_idx]; coords2_tensor = ring_coords_t1[ring_idx]
            if coords1_tensor is None or coords2_tensor is None or coords1_tensor.shape[0] < 10 or coords2_tensor.shape[0] < 10: logging.debug(f"{prefix} Ring empty or too small. Skipping."); continue

            try:
                 rows1, cols1 = coords1_tensor[:, 0], coords1_tensor[:, 1]; rows2, cols2 = coords2_tensor[:, 0], coords2_tensor[:, 1]
                 v1_tensor = Yl_t_mod[rows1, cols1]; v2_tensor = Yl_t1_mod[rows2, cols2]
                 min_s = min(v1_tensor.numel(), v2_tensor.numel())
                 if min_s == 0: logging.debug(f"{prefix} Zero size after indexing. Skipping."); continue
                 if v1_tensor.numel() != v2_tensor.numel(): v1_tensor=v1_tensor[:min_s]; v2_tensor=v2_tensor[:min_s]; rows1,cols1=rows1[:min_s],cols1[:min_s]; rows2,cols2=rows2[:min_s],cols2[:min_s]
            except IndexError: logging.warning(f"{prefix} Tensor Indexing error. Skipping."); continue

            v1_np = v1_tensor.cpu().numpy(); v2_np = v2_tensor.cpu().numpy()
            if not np.all(np.isfinite(v1_np)) or not np.all(np.isfinite(v2_np)): logging.warning(f"{prefix} Non-finite values in ring data. Skipping."); continue
            logging.debug(f"{prefix} Ring values extracted. Size={v1_np.size}. v1_mean={np.mean(v1_np):.4f}, v2_mean={np.mean(v2_np):.4f}")

            alpha = compute_adaptive_alpha_entropy(v1_np, ring_idx, frame_number)
            logging.info(
                f"{prefix} [Embed - Перед DCT] v1_np (из Yl_t): size={v1_np.size}, min={np.min(v1_np):.4e}, max={np.max(v1_np):.4e}, mean={np.mean(v1_np):.4e}, std={np.std(v1_np):.4e}")
            # Логируем также v2_np
            logging.info(
                f"{prefix} [Embed - Перед DCT] v2_np (из Yl_t1): size={v2_np.size}, min={np.min(v2_np):.4e}, max={np.max(v2_np):.4e}, mean={np.mean(v2_np):.4e}, std={np.std(v2_np):.4e}")

            d1 = dct_1d(v1_np); d2 = dct_1d(v2_np)
            if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(d2)): logging.warning(f"{prefix} NaN/inf after DCT. Skipping."); continue
            logging.debug(f"{prefix} DCT done. d1[0]={d1[0]:.4f}, d2[0]={d2[0]:.4f}")

            try: U1, S1v, Vt1 = scipy_svd(d1.reshape(-1,1), compute_uv=True, full_matrices=False); U2, S2v, Vt2 = scipy_svd(d2.reshape(-1,1), compute_uv=True, full_matrices=False)
            except np.linalg.LinAlgError: logging.warning(f"{prefix} SVD failed. Skipping."); continue
            if S1v is None or S1v.size == 0 or S2v is None or S2v.size == 0: logging.warning(f"{prefix} SVD empty result. Skipping."); continue
            if not np.all(np.isfinite(S1v)) or not np.all(np.isfinite(S2v)): logging.warning(f"{prefix} SVD NaN/inf. Skipping."); continue
            s1 = S1v[0]; s2 = S2v[0]
            eps=1e-12; original_ratio = s1/(s2+eps) if abs(s2) > eps else float('inf') # Безопасное вычисление
            original_implied_bit = 0 if original_ratio >= 1.0 else 1
            logging.debug(f"{prefix} PRE: s1={s1:.4e}, s2={s2:.4e}, OrigRatio={original_ratio:.6f} -> ImpliedBit={original_implied_bit} (Target={bit_to_embed}, Alpha={alpha:.4f})")

            ns1,ns2=s1,s2; modified=False; a2=alpha*alpha; inv_a=1/(alpha+eps) if abs(alpha)>eps else float('inf')
            action = "No change needed"
            if bit_to_embed == 0:
                if original_ratio >= inv_a: ns1=(s1+alpha*s2)/(1+a2); ns2=(alpha*s1+a2*s2)/(1+a2); modified=True; action=f"Modifying to 0 (ratio {original_ratio:.4f} >= {inv_a:.4f})"
            else: # bit_to_embed == 1
                if original_ratio <= alpha: ns1=(s1*a2+alpha*s2)/(a2+1); ns2=(alpha*s1+s2)/(a2+1); modified=True; action=f"Modifying to 1 (ratio {original_ratio:.4f} <= {alpha:.4f})"

            if modified:
                modifications_count+=1
                new_ratio_check = ns1 / (ns2 + eps) if abs(ns2) > eps else float('inf')
                logging.debug(f"{prefix} POST: Action: {action}. New s1={ns1:.4e}, s2={ns2:.4e}, NewRatio≈{new_ratio_check:.6f}")
                try:
                    d1m = U1[:, 0] * ns1 * Vt1[0, 0]; d2m = U2[:, 0] * ns2 * Vt2[0, 0]
                    v1m_np=idct_1d(d1m); v2m_np=idct_1d(d2m)
                    if not np.all(np.isfinite(v1m_np)) or not np.all(np.isfinite(v2m_np)): raise ValueError("NaN/inf after IDCT")
                except Exception as recon_err: logging.warning(f"{prefix} Recon/IDCT err: {recon_err}. Skipping modification."); continue
                if v1m_np.size != v1_np.size or v2m_np.size != v2_np.size: logging.warning(f"{prefix} Size mismatch after IDCT. Skipping."); continue

                delta1_np=v1m_np-v1_np; delta2_np=v2m_np-v2_np
                delta1 = torch.from_numpy(delta1_np).to(device); delta2 = torch.from_numpy(delta2_np).to(device)
                mf1 = torch.ones_like(delta1); mf2 = torch.ones_like(delta2)
                if use_perceptual_masking and perceptual_mask_tensor is not None:
                    try:
                        mv1=perceptual_mask_tensor[rows1, cols1]; mv2=perceptual_mask_tensor[rows2, cols2]
                        mf1.mul_(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mv1); mf2.mul_(LAMBDA_PARAM+(1.0-LAMBDA_PARAM)*mv2)
                        # Добавим логирование маскирующих факторов
                        logging.debug(f"{prefix} Mask factors applied. mf1 range: [{torch.min(mf1):.3f}-{torch.max(mf1):.3f}], mf2 range: [{torch.min(mf2):.3f}-{torch.max(mf2):.3f}]")
                    except Exception as mask_err: logging.warning(f"{prefix} Mask apply error: {mask_err}")
                try:
                    Yl_t_mod[rows1, cols1] += delta1 * mf1; Yl_t1_mod[rows2, cols2] += delta2 * mf2
                    logging.debug(f"{prefix} Deltas applied. Delta1 mean={torch.mean(delta1*mf1):.4e}, Delta2 mean={torch.mean(delta2*mf2):.4e}")
                except IndexError: logging.warning(f"{prefix} Tensor Delta apply error. Skipping."); continue
            else:
                 logging.debug(f"{prefix} POST: Action: {action}.")

        logging.debug(f"{prefix_base} --- End Ring Loop ---")
        # --- Конец цикла модификации ---

        # 5. Обратное DTCWT (PyTorch) - УБРАН ХАК
        if Yl_t_mod.dim() == 2: Yl_t_mod = Yl_t_mod.unsqueeze(0).unsqueeze(0)
        if Yl_t1_mod.dim() == 2: Yl_t1_mod = Yl_t1_mod.unsqueeze(0).unsqueeze(0)
        logging.debug(f"{prefix_base} Calling Inverse DTCWT...")
        c1m_np = dtcwt_pytorch_inverse(Yl_t_mod, Yh_t, dtcwt_inv, device, target_shape_hw, frame_number)
        c2m_np = dtcwt_pytorch_inverse(Yl_t1_mod, Yh_t1, dtcwt_inv, device, target_shape_hw, frame_number + 1)
        if c1m_np is None or c2m_np is None: logging.error(f"{prefix_base} Inverse DTCWT failed."); return None, None
        logging.debug(f"{prefix_base} Inverse DTCWT done.")

        # 6. Постобработка и сборка кадра (NumPy)
        c1s_np = np.clip(c1m_np * 255.0, 0, 255).astype(np.uint8)
        c2s_np = np.clip(c2m_np * 255.0, 0, 255).astype(np.uint8)
        if c1s_np.shape != target_shape_hw: c1s_np = cv2.resize(c1s_np, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        if c2s_np.shape != target_shape_hw: c2s_np = cv2.resize(c2s_np, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        ny1_np = f1_ycrcb_np.copy(); ny2_np = f2_ycrcb_np.copy() # Копируем исходные YCrCb
        ny1_np[:,:,embed_component] = c1s_np; ny2_np[:,:,embed_component] = c2s_np
        f1m = cv2.cvtColor(ny1_np, cv2.COLOR_YCrCb2BGR); f2m = cv2.cvtColor(ny2_np, cv2.COLOR_YCrCb2BGR)
        logging.debug(f"{prefix_base} Embed Pair Finished. Total Mods: {modifications_count}")
        return f1m, f2m

    except cv2.error as cv_err: logging.error(f"OpenCV error P:{pair_index}: {cv_err}", exc_info=True); return None, None
    except MemoryError: logging.error(f"Memory error P:{pair_index}", exc_info=True); return None, None
    except Exception as e: logging.error(f"Critical error in embed_frame_pair P:{pair_index}: {e}", exc_info=True); return None, None


# --- ИЗМЕНЕННЫЙ _embed_single_pair_task ---
# @profile
def _embed_single_pair_task(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    # ... (код функции _embed_single_pair_task из предыдущего ответа) ...
    pair_idx=args.get('pair_idx',-1); f1_bgr=args.get('frame1'); f2_bgr=args.get('frame2'); bits_for_this_pair=args.get('bits',[])
    nr=args.get('n_rings', N_RINGS); nrtu=args.get('num_rings_to_use', NUM_RINGS_TO_USE); cps=args.get('candidate_pool_size', CANDIDATE_POOL_SIZE)
    ec=args.get('embed_component', EMBED_COMPONENT); upm=args.get('use_perceptual_masking', USE_PERCEPTUAL_MASKING)
    device=args.get('device'); dtcwt_fwd=args.get('dtcwt_fwd'); dtcwt_inv=args.get('dtcwt_inv')
    fn=2*pair_idx; selected_rings=[]
    if pair_idx==-1 or f1_bgr is None or f2_bgr is None or not bits_for_this_pair or device is None or dtcwt_fwd is None or dtcwt_inv is None: logging.error(f"Missing args P:{pair_idx}"); return fn,None,None,[]
    try:
        candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
        if len(candidate_rings) < nrtu: logging.warning(f"[P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{nrtu}. Using all."); nrtu=len(candidate_rings)
        if nrtu == 0: raise ValueError("No candidates and nrtu=0") # Добавил проверку
        f1_ycrcb_np = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
        comp1_tensor = torch.from_numpy(f1_ycrcb_np[:, :, ec].copy()).to(device=device, dtype=torch.float32) / 255.0
        Yl_t_select, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
        if Yl_t_select is None: raise RuntimeError(f"DTCWT FWD failed P:{pair_idx}")
        if Yl_t_select.dim() > 2: Yl_t_select = Yl_t_select.squeeze()
        coords = ring_division(Yl_t_select, nr, fn)
        if coords is None or len(coords) != nr: raise RuntimeError(f"Ring division failed P:{pair_idx}")
        entropies = []
        min_pixels_for_entropy = 10
        for r_idx in candidate_rings:
            entropy_val = -float('inf')
            # Проверка валидности индекса и наличия координат
            if 0 <= r_idx < len(coords) and coords[r_idx] is not None and coords[r_idx].shape[0] >= min_pixels_for_entropy: # Теперь переменная определена
                 c_tensor = coords[r_idx]
                 try:
                       rows, cols = c_tensor[:, 0], c_tensor[:, 1]
                       rv_tensor = Yl_t_select[rows, cols]
                       rv_np = rv_tensor.cpu().numpy()
                       shannon_entropy, _ = calculate_entropies(rv_np, fn, r_idx)
                       if np.isfinite(shannon_entropy): entropy_val = shannon_entropy
                 except IndexError:
                       logging.warning(f"[P:{pair_idx},R:{r_idx}] IndexError during entropy calculation (tensor indexing)")
                 except Exception as e:
                       logging.warning(f"[P:{pair_idx},R:{r_idx}] Entropy calc/conv error: {e}")
            entropies.append((entropy_val, r_idx)) # Сохраняем (энтропия, индекс_кольца)
        entropies.sort(key=lambda x: x[0], reverse=True); selected_rings=[idx for e,idx in entropies if e>-float('inf')][:nrtu]
        if len(selected_rings)<nrtu:
             logging.warning(f"[P:{pair_idx}] Fallback for ring selection ({len(selected_rings)}<{nrtu}).")
             det_fallback = candidate_rings[:nrtu];
             for ring in det_fallback:
                  if ring not in selected_rings: selected_rings.append(ring)
                  if len(selected_rings) == nrtu: break
             if len(selected_rings)<nrtu: raise RuntimeError(f"Fallback failed P:{pair_idx}")
        # logging.info(f"[P:{pair_idx}] Selected rings: {selected_rings}") # Уменьшаем лог
        bits_to_embed_now = bits_for_this_pair[:len(selected_rings)]
        # for bit_val, ring_i in zip(bits_to_embed_now, selected_rings): logging.info(f"[P:{pair_idx} EMBED] Bit={bit_val} -> Ring={ring_i}")
        mod_f1, mod_f2 = embed_frame_pair(f1_bgr, f2_bgr, bits_to_embed_now, selected_rings, nr, fn, upm, ec, device, dtcwt_fwd, dtcwt_inv)
        return fn, mod_f1, mod_f2, selected_rings
    except Exception as e: logging.error(f"Error in _embed_single_pair_task P:{pair_idx}: {e}", exc_info=True); return fn,None,None,[]


# --- _embed_batch_worker - остается БЕЗ ИЗМЕНЕНИЙ (вызывает _embed_single_pair_task) ---
def _embed_batch_worker(batch_args_list: List[Dict]) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]]:
    batch_results = []
    for args in batch_args_list:
        result = _embed_single_pair_task(args)
        batch_results.append(result)
    return batch_results


# --- ВОССТАНОВЛЕННАЯ embed_watermark_in_video ---
# @profile # Добавьте, если нужно профилировать
def embed_watermark_in_video(
        frames: List[np.ndarray],
        payload_id_bytes: bytes, # Принимаем байты ID
        n_rings: int = N_RINGS, num_rings_to_use: int = NUM_RINGS_TO_USE,
        bits_per_pair: int = BITS_PER_PAIR, candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        # --- Параметры гибридного режима ---
        use_hybrid_ecc: bool = True,       # Включить гибридный режим?
        max_total_packets: int = 15,       # Макс. общее число пакетов (1 ECC + N Raw)
        use_ecc_for_first: bool = USE_ECC, # Использовать ECC для первого пакета? (берем из глоб. USE_ECC)
        bch_code: Optional[galois.BCH] = BCH_CODE_OBJECT, # Глобальный объект BCH
        # --- Новые параметры для PyTorch ---
        device: torch.device = torch.device("cpu"), # Устройство по умолчанию CPU
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
    if dtcwt_fwd is None or dtcwt_inv is None:
         logging.error("Экземпляры DTCWTForward/DTCWTInverse не переданы в embed_watermark_in_video!")
         return frames[:] # Возвращаем без изменений

    num_frames = len(frames); total_pairs = num_frames // 2
    payload_len_bytes = len(payload_id_bytes); payload_len_bits = payload_len_bytes * 8
    if payload_len_bits == 0 or total_pairs == 0: logging.warning("Нет данных или пар кадров."); return frames[:]
    if max_total_packets <= 0: max_total_packets = 1

    logging.info(f"--- Embed Start (PyTorch, Hybrid: {use_hybrid_ecc}, Max Pkts: {max_total_packets}) ---")

    # --- Формирование бит ---
    bits_to_embed_list = []; raw_payload_bits: Optional[np.ndarray] = None
    try: raw_payload_bits = np.unpackbits(np.frombuffer(payload_id_bytes, dtype=np.uint8)); assert raw_payload_bits.size == payload_len_bits
    except Exception as e: logging.error(f"Ошибка unpackbits: {e}"); return frames[:]
    if raw_payload_bits is None: return frames[:] # Дополнительная проверка

    packet1_type_str = "N/A"; packet1_len = 0; num_raw_packets_added = 0
    can_use_ecc = use_ecc_for_first and GALOIS_AVAILABLE and bch_code is not None and payload_len_bits <= bch_code.k

    if use_hybrid_ecc and can_use_ecc:
        first_packet_bits = add_ecc(raw_payload_bits, bch_code)
        if first_packet_bits is not None: bits_to_embed_list.extend(first_packet_bits.tolist()); packet1_len=len(first_packet_bits); packet1_type_str=f"ECC(n={packet1_len}, t={bch_code.t})"
        else: logging.error("Не удалось создать ECC пакет!"); return frames[:]
    else:
        bits_to_embed_list.extend(raw_payload_bits.tolist()); packet1_len=len(raw_payload_bits); packet1_type_str=f"Raw({packet1_len})"
        if not use_hybrid_ecc: logging.info(f"Режим НЕ гибридный: Используется {packet1_type_str}.")
        else: logging.info(f"Гибридный режим: ECC невозможен/выкл. Первый пакет {packet1_type_str}.")
        use_hybrid_ecc = False # Выключаем гибрид, если первый Raw

    if use_hybrid_ecc:
        num_raw_repeats_to_add = max(0, max_total_packets - 1)
        for _ in range(num_raw_repeats_to_add): bits_to_embed_list.extend(raw_payload_bits.tolist()); num_raw_packets_added += 1
        if num_raw_packets_added > 0: logging.info(f"Добавлено {num_raw_packets_added} Raw payload пакетов.")

    total_packets_actual = 1 + num_raw_packets_added
    total_bits_to_embed = len(bits_to_embed_list)
    if total_bits_to_embed == 0: logging.error("Нет бит для встраивания."); return frames[:]

    pairs_needed = ceil(total_bits_to_embed / bits_per_pair); pairs_to_process = min(total_pairs, pairs_needed)
    bits_flat_final = np.array(bits_to_embed_list[:pairs_to_process * bits_per_pair], dtype=np.uint8)
    actual_bits_embedded = len(bits_flat_final)

    logging.info(f"Подготовка: Цел.пакетов={total_packets_actual}, Бит={total_bits_to_embed}, НужноПар={pairs_needed}, ДоступноПар={total_pairs}, Обраб.Пар={pairs_to_process}, ВстроеноБит={actual_bits_embedded}")
    if actual_bits_embedded < total_bits_to_embed: logging.warning(f"Не хватает пар кадров!")

    # --- Подготовка аргументов и запуск потоков ---
    start_time=time.time(); watermarked_frames=frames[:]; rings_log: Dict[int,List[int]]={}; pc,ec,uc=0,0,0; skipped_pairs=0; all_pairs_args=[]

    for pair_idx in range(pairs_to_process):
        i1=2*pair_idx; i2=i1+1
        if i2>=num_frames or frames[i1] is None or frames[i2] is None: skipped_pairs+=1; continue
        sbi=pair_idx*bits_per_pair; ebi=sbi+bits_per_pair
        if sbi >= len(bits_flat_final): break
        if ebi > len(bits_flat_final): ebi = len(bits_flat_final)
        cb=bits_flat_final[sbi:ebi].tolist()
        if len(cb) == 0: continue

        args={'pair_idx':pair_idx, 'frame1':frames[i1], 'frame2':frames[i2], 'bits':cb,
              'n_rings':n_rings, 'num_rings_to_use':num_rings_to_use, 'candidate_pool_size':candidate_pool_size,
              'frame_number':i1, 'use_perceptual_masking':use_perceptual_masking, 'embed_component':embed_component,
              'device': device, 'dtcwt_fwd': dtcwt_fwd, 'dtcwt_inv': dtcwt_inv} # Передаем объекты
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0: logging.warning(f"Пропущено {skipped_pairs} пар при подготовке.")
    if num_valid_tasks == 0: logging.error("Нет валидных задач для встраивания."); return watermarked_frames

    # --- Запуск ThreadPoolExecutor ---
    num_workers = max_workers if max_workers is not None and max_workers>0 else (os.cpu_count() or 1)
    batch_size = max(1, ceil(num_valid_tasks / num_workers)); num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i:i+batch_size] for i in range(0, num_valid_tasks, batch_size) if all_pairs_args[i:i+batch_size]]
    logging.info(f"Запуск {len(batched_args_list)} батчей ({num_valid_tasks} пар) в ThreadPool (mw={num_workers}, batch≈{batch_size})...")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch_idx = {executor.submit(_embed_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]; original_batch = batched_args_list[batch_idx]
                try:
                    batch_results = future.result()
                    if len(batch_results) != len(original_batch): ec += len(original_batch); logging.error(f"Batch {batch_idx} size mismatch!"); continue
                    for i, single_res in enumerate(batch_results):
                        original_args = original_batch[i]; pair_idx = original_args.get('pair_idx', -1)
                        if pair_idx == -1: logging.error(f"Не найден pair_idx в результате батча {batch_idx}"); ec+=1; continue
                        if single_res and len(single_res)==4:
                            fn_res, mod_f1, mod_f2, sel_rings = single_res; i1=2*pair_idx; i2=i1+1
                            if sel_rings: rings_log[pair_idx] = sel_rings
                            if mod_f1 is not None and mod_f2 is not None:
                                if i1<len(watermarked_frames): watermarked_frames[i1]=mod_f1; uc+=1
                                if i2<len(watermarked_frames): watermarked_frames[i2]=mod_f2; uc+=1
                                pc+=1
                            else: logging.warning(f"Embedding failed P:{pair_idx} (None frames)."); ec+=1
                        else: logging.warning(f"Invalid result structure P:{pair_idx} in batch {batch_idx}."); ec+=1
                except Exception as e: failed_pairs_count=len(original_batch); logging.error(f"Batch {batch_idx} failed: {e}", exc_info=True); ec += failed_pairs_count
    except Exception as e: logging.critical(f"ThreadPool critical error: {e}", exc_info=True); return frames[:]

    # --- Завершение и запись логов ---
    logging.info(f"Batch done. OK pairs: {pc}, Error/Skip pairs: {ec+skipped_pairs}. Frames updated: {uc}.")
    if rings_log:
        try:
            ser_log={str(k):v for k,v in rings_log.items()}
            with open(SELECTED_RINGS_FILE,'w') as f: json.dump(ser_log, f, indent=4)
            logging.info(f"Rings log saved: {SELECTED_RINGS_FILE}")
        except Exception as e: logging.error(f"Save rings log failed: {e}")
    else: logging.warning("Rings log empty.")
    end_time=time.time(); logging.info(f"Embed function done. Time: {end_time-start_time:.2f}s.")
    return watermarked_frames

# --- ИЗМЕНЕННАЯ main ---
def main():
    start_time_main = time.time()
    # --- Инициализация PyTorch / Galois ---
    if not PYTORCH_WAVELETS_AVAILABLE: print("ERROR: pytorch_wavelets not found."); return
    if USE_ECC and not GALOIS_AVAILABLE: print("\nWARNING: ECC requested but galois unavailable.")
    # Определяем устройство
    if torch.cuda.is_available():
        try: device = torch.device("cuda"); torch.cuda.get_device_name(0); logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        except Exception as e: logging.warning(f"CUDA init failed ({e}). Using CPU."); device = torch.device("cpu")
    else: device = torch.device("cpu"); logging.info("Using CPU.")
    # Создаем экземпляры трансформаций
    try:
        dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
        dtcwt_inv = DTCWTInverse(biort='near_sym_a', qshift='qshift_a').to(device)
        logging.info("PyTorch DTCWT instances created.")
    except Exception as e: logging.critical(f"Failed to init pytorch-wavelets: {e}"); return

    input_video = "input.mp4"
    base_output_filename = f"watermarked_pytorch_hybrid_t{BCH_T}"
    output_video = base_output_filename + OUTPUT_EXTENSION
    logging.info(f"--- Starting Embedding Main Process (PyTorch) ---")
    logging.info(f"Input: {input_video}, Output: {output_video}")

    frames, fps_read = read_video(input_video)
    payload_len_bits = PAYLOAD_LEN_BYTES * 8
    if not frames: logging.critical("Video read failed."); return
    fps_to_use = float(FPS) if fps_read <= 0 else fps_read
    if len(frames) < 2: logging.critical("Not enough frames."); return

    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES); original_id_hex = original_id_bytes.hex()
    logging.info(f"Generated Payload ID ({payload_len_bits} bit, Hex): {original_id_hex}")
    try:
        with open(ORIGINAL_WATERMARK_FILE, "w") as f: f.write(original_id_hex)
        logging.info(f"Original ID saved: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e: logging.error(f"Save ID failed: {e}")

    # --- Вызов основной функции встраивания ---
    watermarked_frames = embed_watermark_in_video(
        frames=frames, payload_id_bytes=original_id_bytes,
        use_hybrid_ecc=True, max_total_packets=15, # Включаем гибридный режим
        use_ecc_for_first=USE_ECC, bch_code=BCH_CODE_OBJECT,
        device=device, dtcwt_fwd=dtcwt_fwd, dtcwt_inv=dtcwt_inv, # Передаем объекты
        n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE,
        bits_per_pair=BITS_PER_PAIR, candidate_pool_size=CANDIDATE_POOL_SIZE,
        fps=fps_to_use, max_workers=MAX_WORKERS,
        use_perceptual_masking=USE_PERCEPTUAL_MASKING,
        embed_component=EMBED_COMPONENT
    )

    # --- Запись видео ---
    if watermarked_frames and len(watermarked_frames) == len(frames):
        write_video(frames=watermarked_frames, out_path=output_video, fps=fps_to_use, codec=OUTPUT_CODEC)
        logging.info(f"Watermarked video saved: {output_video}")
    else: logging.error("Embedding failed. No output.")

    logging.info(f"--- Embedding Main Process Finished (PyTorch) ---")
    total_time_main = time.time() - start_time_main
    logging.info(f"--- Total Main Time: {total_time_main:.2f} sec ---")
    print(f"\nEmbedding process (PyTorch) finished.")
    print(f"Output: {output_video}, Log: {LOG_FILENAME}, ID file: {ORIGINAL_WATERMARK_FILE}, Rings file: {SELECTED_RINGS_FILE}")
    print("\nRun extractor_pytorch.py to extract.")# --- Точка Входа (__name__ == "__main__") ---
if __name__ == "__main__":
    # --- Инициализация и проверки ---

    # Проверка доступности PyTorch Wavelets (импорт происходит в начале файла)
    if not PYTORCH_WAVELETS_AVAILABLE:
         print("\nERROR: Библиотека pytorch_wavelets не найдена или не может быть импортирована!")
         print("Пожалуйста, установите ее: pip install pytorch_wavelets")
         sys.exit(1) # Критическая ошибка, выходим

    # Проверка доступности Galois и его инициализации (происходит в начале файла)
    # Выводим предупреждение, если ECC включен глобально, но библиотека недоступна
    if USE_ECC and not GALOIS_AVAILABLE:
        print("\nWARNING: USE_ECC=True в настройках, но библиотека galois недоступна или не инициализировалась.")
        print("Первый пакет будет встроен как Raw payload, даже если запрошен гибридный режим с ECC.")
        # Нет необходимости выходить, эмбеддер обработает это

    # --- Профилирование (опционально) ---
    DO_PROFILING = False # Установите True, чтобы включить cProfile
    prof = None
    if DO_PROFILING:
        prof = cProfile.Profile()
        prof.enable()
        logging.info("cProfile включен.")

    # --- Запуск основной логики ---
    final_exit_code = 0
    try:
        main() # Вызов основной функции main()
    except FileNotFoundError as e:
         print(f"\nERROR: Файл не найден: {e}")
         logging.error(f"Файл не найден: {e}", exc_info=True)
         final_exit_code = 1
    except Exception as e:
         # Логируем критическую ошибку перед выходом
         logging.critical(f"Необработанное исключение в main: {e}", exc_info=True)
         print(f"\nCRITICAL ERROR: {e}. См. лог: {LOG_FILENAME}")
         final_exit_code = 1
    finally:
        # --- Сохранение результатов профилирования (если включено) ---
        if DO_PROFILING and prof is not None:
            prof.disable()
            stats = pstats.Stats(prof)
            print("\n--- Profiling Stats (Top 30 Cumulative Time) ---")
            try:
                 stats.strip_dirs().sort_stats("cumulative").print_stats(30)
            except Exception as e_stats:
                 print(f"Ошибка вывода статистики профилирования: {e_stats}")
            print("-------------------------------------------------")

            # Формируем имя файла профиля
            # (Убрал зависимость от backend_str, т.к. теперь только PyTorch)
            pfile = f"profile_embed_pytorch_hybrid_t{BCH_T}.txt"
            try:
                # Сохраняем полную статистику в файл
                with open(pfile, "w", encoding='utf-8') as f: # Добавил encoding
                    sf = pstats.Stats(prof, stream=f)
                    sf.strip_dirs().sort_stats("cumulative").print_stats() # Сохраняем все
                logging.info(f"Статистика профилирования сохранена: {pfile}")
                print(f"Статистика профилирования сохранена: {pfile}")
            except IOError as e:
                logging.error(f"Не удалось сохранить файл профилирования: {e}")
                print(f"Предупреждение: Не удалось сохранить статистику профилирования.")
            except Exception as e_prof_save:
                 logging.error(f"Ошибка сохранения профиля: {e_prof_save}", exc_info=True)
        # --- Восстановление бэкенда dtcwt БОЛЬШЕ НЕ НУЖНО ---

    # Завершаем скрипт с соответствующим кодом выхода
    sys.exit(final_exit_code)
