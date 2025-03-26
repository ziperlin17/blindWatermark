# -*- coding: utf-8 -*-
# Файл: embedder.py

import cv2
import numpy as np
import random
import logging
import time
import concurrent.futures
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any

# --- Константы (Общие) ---
# Настройки адаптивной альфы
LAMBDA_PARAM: float = 0.04
ALPHA_MIN: float = 1.01 # Минимальная альфа
ALPHA_MAX: float = 1.10 # Максимальная альфа (УМЕНЬШЕНО для теста видимости)

# Константы для модификации альфы на основе Ev (добавлено)
EV_THRESHOLD: float = 5.0   # Порог Ev для "гладких" областей (ПОДБИРАЕТСЯ)
EV_SCALE_MIN: float = 0.3    # Минимальный множитель для адаптивной части альфы (ПОДБИРАЕТСЯ)

# Общие настройки
N_RINGS: int = 8 # Количество колец
DEFAULT_RING_INDEX: int = 4 # Индекс по умолчанию (СМЕНЕНО на 4 для теста)
FPS: int = 30 # Частота кадров по умолчанию
LOG_FILENAME: str = 'watermarking_embed.log' # Имя лог-файла для эмбеддера
MAX_WORKERS: Optional[int] = None # None - автовыбор потоков

# --- Настройки Адаптивности (ВАЖНО!) ---
ADAPTIVE_RING_SELECTION: bool = False # False - использовать DEFAULT_RING_INDEX
RING_SELECTION_METRIC: str = 'entropy' # 'entropy', 'energy', 'mean' (Используется, только если ADAPTIVE_RING_SELECTION = True)

# --- Настройка Логирования ---
# Удаляем старые хендлеры, если они есть
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode='w', # 'w' - Перезаписывать лог при каждом запуске ЭМБЕДДЕРА
    level=logging.DEBUG, # Максимальная детализация логов
    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s'
)
logging.info(f"--- Запуск Скрипта Встраивания ---")
logging.info(f"Настройки: Адаптивное кольцо={ADAPTIVE_RING_SELECTION}, Метрика='{RING_SELECTION_METRIC if ADAPTIVE_RING_SELECTION else 'N/A'}', N_RINGS={N_RINGS}, Default Ring={DEFAULT_RING_INDEX}")
logging.info(f"Альфа: LAMBDA={LAMBDA_PARAM}, MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, EV_THRESHOLD={EV_THRESHOLD}, EV_SCALE_MIN={EV_SCALE_MIN}")

# ============================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ОБЩИЕ) ---
# ============================================================

def dct_1d(signal_1d: np.ndarray) -> np.ndarray:
    logging.debug(f"Input shape: {signal_1d.shape}")
    result = dct(signal_1d, type=2, norm='ortho')
    logging.debug(f"Output shape: {result.shape}")
    return result

def idct_1d(coeffs_1d: np.ndarray) -> np.ndarray:
    logging.debug(f"Input shape: {coeffs_1d.shape}")
    result = idct(coeffs_1d, type=2, norm='ortho')
    logging.debug(f"Output shape: {result.shape}")
    return result

def dtcwt_transform(y_plane: np.ndarray) -> Optional[Pyramid]:
    func_start_time = time.time()
    logging.debug(f"Input Y plane shape: {y_plane.shape}, dtype: {y_plane.dtype}, Min: {np.min(y_plane):.4f}, Max: {np.max(y_plane):.4f}")
    if np.any(np.isnan(y_plane)): logging.warning("NaNs detected in input Y plane!")
    try:
        t = Transform2d()
        pyramid = t.forward(y_plane.astype(np.float32), nlevels=1)
        if hasattr(pyramid, 'lowpass') and pyramid.lowpass is not None:
            lp = pyramid.lowpass
            logging.debug(f"DTCWT lowpass shape: {lp.shape}, dtype: {lp.dtype}, Min: {np.min(lp):.4f}, Max: {np.max(lp):.4f}")
            if np.any(np.isnan(lp)): logging.warning("NaNs detected in lowpass component!")
        else:
            logging.error("DTCWT forward did not produce a valid lowpass attribute!")
            return None
        logging.debug(f"DTCWT transform time: {time.time() - func_start_time:.4f}s")
        return pyramid
    except Exception as e:
        logging.error(f"Exception during DTCWT transform: {e}", exc_info=True)
        return None

def dtcwt_inverse(pyramid: Pyramid) -> Optional[np.ndarray]:
    func_start_time = time.time()
    if not isinstance(pyramid, Pyramid) or not hasattr(pyramid, 'lowpass'):
        logging.error("Invalid input pyramid object.")
        return None
    logging.debug(f"Input lowpass shape: {pyramid.lowpass.shape}")
    try:
        t = Transform2d()
        reconstructed_y = t.inverse(pyramid).astype(np.float32)
        logging.debug(f"DTCWT inverse output shape: {reconstructed_y.shape}, dtype: {reconstructed_y.dtype}, Min: {np.min(reconstructed_y):.4f}, Max: {np.max(reconstructed_y):.4f}")
        if np.any(np.isnan(reconstructed_y)): logging.warning("NaNs detected after inverse DTCWT!")
        logging.debug(f"DTCWT inverse time: {time.time() - func_start_time:.4f}s")
        return reconstructed_y
    except Exception as e:
        logging.error(f"Exception during DTCWT inverse: {e}", exc_info=True)
        return None

# --- Переписанная ring_division с Максимальным Логированием ---
def ring_division(lowpass_subband: np.ndarray, n_rings: int = N_RINGS) -> List[List[Tuple[int, int]]]:
    """
    Векторизованное разбиение низкочастотной подполосы на кольца с подробным логированием.
    Гарантирует центральное расположение.
    """
    func_start_time = time.time()
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error(f"Input is not a 2D numpy array! Type: {type(lowpass_subband)}")
        return [[] for _ in range(n_rings)]

    H, W = lowpass_subband.shape
    logging.info(f"Input shape H={H}, W={W}")

    if H < 2 or W < 2:
        logging.error(f"Subband dimensions too small: H={H}, W={W}. Cannot create rings.")
        return [[] for _ in range(n_rings)]

    # 1. Определяем ЦЕНТР (индексы строки и столбца) - ЯВНО
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    logging.debug(f"Calculated precise center: row={center_r:.2f}, col={center_c:.2f}")
    int_center_r, int_center_c = H // 2, W // 2
    logging.debug(f"Integer center for reference: row={int_center_r}, col={int_center_c}")

    # 2. Генерируем сетки координат ИНДЕКСОВ пикселей
    rr, cc = np.indices((H, W), dtype=np.float32) # rr[i,j]=i, cc[i,j]=j

    # 3. Вычисляем расстояние от ТОЧНОГО центра для каждого пикселя
    distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2)
    min_dist, max_dist = np.min(distances), np.max(distances)
    logging.debug(f"Distances calculated. Shape: {distances.shape}, Min: {min_dist:.4f}, Max: {max_dist:.4f}")

    # 4. Определяем границы колец
    if max_dist < 1e-6:
        logging.warning(f"Max distance is near zero ({max_dist:.4f}).")
        delta_r = 1.0; ring_bins = np.array([0.0, 1.0]); n_rings_eff = 1
        logging.warning("Forcing effective n_rings=1 due to zero max distance.")
    else:
        delta_r = max_dist / n_rings
        ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1)
        n_rings_eff = n_rings

    logging.debug(f"delta_r={delta_r:.4f}, ring_bins={np.round(ring_bins, 2)}")
    if len(ring_bins) < 2:
        logging.error("Invalid ring bins generated!"); return [[] for _ in range(n_rings)]

    # 5. Назначаем индекс кольца каждому пикселю
    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0 # Коррекция для центра
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1) # Ограничиваем сверху

    min_idx, max_idx = np.min(ring_indices), np.max(ring_indices)
    logging.debug(f"Ring indices calculated. Shape: {ring_indices.shape}, Min: {min_idx}, Max: {max_idx}")

    # 6. Собираем координаты для каждого кольца
    rings_coords = [[] for _ in range(n_rings)] # Создаем список нужной длины n_rings
    pixel_counts = np.zeros(n_rings, dtype=int)

    for ring_idx in range(n_rings_eff): # Итерируем до n_rings_eff
        coords_for_ring = np.argwhere(ring_indices == ring_idx)
        rings_coords[ring_idx] = [tuple(coord) for coord in coords_for_ring]
        pixel_counts[ring_idx] = len(rings_coords[ring_idx])

    logging.info(f"Pixel counts per ring (up to {n_rings_eff-1}): {pixel_counts[:n_rings_eff]}")
    total_pixels_in_rings = np.sum(pixel_counts)
    total_pixels_in_subband = H * W
    if total_pixels_in_rings != total_pixels_in_subband:
        logging.warning(f"Pixel count mismatch! Rings: {total_pixels_in_rings}, Subband: {total_pixels_in_subband}. Diff: {total_pixels_in_subband - total_pixels_in_rings}")
    for i in range(n_rings): # Проверяем все n_rings на пустоту
        if not rings_coords[i]:
            logging.warning(f"Ring {i} is empty!")

    logging.debug(f"Ring division time: {time.time() - func_start_time:.4f}s")
    return rings_coords


def calculate_entropies(ring_vals: np.ndarray) -> Tuple[float, float]:
    # Использует фиксированный диапазон [0, 1] для гистограммы
    eps = 1e-12
    if ring_vals.size == 0: return 0.0, 0.0
    hist, _ = np.histogram(ring_vals, bins=256, range=(0.0, 1.0), density=False)
    total_count = ring_vals.size
    if total_count == 0: return 0.0, 0.0
    probabilities = hist / total_count
    probabilities = probabilities[probabilities > eps]
    if probabilities.size == 0: return 0.0, 0.0
    visual_entropy = -np.sum(probabilities * np.log2(probabilities))
    edge_entropy = -np.sum(probabilities * np.exp(1.0 - probabilities))
    logging.debug(f"Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f} (Size: {ring_vals.size})")
    return visual_entropy, edge_entropy

# --- ИЗМЕНЕННАЯ Функция вычисления Альфа (с учетом Ev) ---
def compute_adaptive_alpha_entropy(ring_vals: np.ndarray, ring_index: int, frame_number: int) -> float:
    """
    Вычисляет адаптивный коэффициент альфа на основе энтропий (Ee/Ev)
    И ДОПОЛНИТЕЛЬНО ослабляет его для областей с низкой визуальной энтропией (Ev).
    """
    func_start_time = time.time()
    eps = 1e-12

    # 1. Рассчитываем энтропии
    visual_entropy, edge_entropy = calculate_entropies(ring_vals)

    # 2. Рассчитываем базовый адаптивный фактор на основе отношения Ee/Ev
    if abs(visual_entropy) < eps: entropy_ratio = 0.0
    else: entropy_ratio = edge_entropy / visual_entropy
    sigmoid_ratio = 1 / (1 + np.exp(-entropy_ratio)) # В диапазоне (0, 1)

    # 3. Рассчитываем фактор ослабления на основе АБСОЛЮТНОЙ Ev
    ev_scale_factor = np.clip(visual_entropy / EV_THRESHOLD, EV_SCALE_MIN, 1.0)
    logging.debug(f"Ev={visual_entropy:.4f}, Threshold={EV_THRESHOLD}, ScaleFactor={ev_scale_factor:.4f}")

    # 4. Вычисляем финальную альфу
    adaptive_part = (ALPHA_MAX - ALPHA_MIN) * sigmoid_ratio * ev_scale_factor
    final_alpha = ALPHA_MIN + adaptive_part

    # 5. Финальное ограничение
    final_alpha = np.clip(final_alpha, ALPHA_MIN, ALPHA_MAX)

    logging.info(
        f"Frame={frame_number}, ring={ring_index}, Ev={visual_entropy:.4f}, Ee={edge_entropy:.4f}, "
        f"EvScale={ev_scale_factor:.3f} -> final_alpha={final_alpha:.4f}"
    )
    logging.debug(f"Alpha calculation time: {time.time() - func_start_time:.4f}s")
    return final_alpha

def select_embedding_ring(
    lowpass_subband: np.ndarray, rings_coords: List[List[Tuple[int, int]]],
    metric: str = RING_SELECTION_METRIC
) -> int:
    # Без изменений в логике, но с добавленным логированием
    func_start_time = time.time()
    best_metric_value = -float('inf')
    selected_index = DEFAULT_RING_INDEX
    metric_values = []
    if not isinstance(lowpass_subband, np.ndarray) or lowpass_subband.ndim != 2:
        logging.error("Invalid lowpass_subband input!"); return DEFAULT_RING_INDEX
    logging.debug(f"Selecting ring using metric: '{metric}'")

    for i, coords in enumerate(rings_coords):
        current_metric = -float('inf')
        if not coords: metric_values.append(current_metric); continue
        try:
            valid_coords = coords
            if not valid_coords: logging.warning(f"Ring {i} coords empty."); metric_values.append(current_metric); continue
            ring_vals = np.array([lowpass_subband[r, c] for (r, c) in valid_coords], dtype=np.float32)
            if ring_vals.size == 0: metric_values.append(current_metric); continue

            if metric == 'entropy':
                visual_entropy, _ = calculate_entropies(ring_vals); current_metric = visual_entropy
            elif metric == 'energy':
                current_metric = np.sum(ring_vals**2)
            elif metric == 'mean':
                 current_metric = np.abs(np.mean(ring_vals) - 0.5)
            else:
                if i==0: logging.warning(f"Unknown metric '{metric}', using 'entropy'.")
                visual_entropy, _ = calculate_entropies(ring_vals); current_metric = visual_entropy
            metric_values.append(current_metric)
            if current_metric > best_metric_value: best_metric_value = current_metric; selected_index = i
        except Exception as e:
             logging.error(f"Error calculating metric for ring {i}: {e}", exc_info=False); metric_values.append(-float('inf'))

    metric_log_str = ", ".join([f"{i}:{v:.4f}" for i, v in enumerate(metric_values)])
    logging.debug(f"Ring metrics ({metric}): [{metric_log_str}]")
    logging.info(f"Selected ring: {selected_index} (Value: {best_metric_value:.4f})")

    # Проверка на пустые кольца (без изменений)
    if selected_index < 0 or selected_index >= len(rings_coords) or not rings_coords[selected_index]:
         logging.error(f"Selected ring {selected_index} is empty/invalid! Checking default {DEFAULT_RING_INDEX}.")
         if 0 <= DEFAULT_RING_INDEX < len(rings_coords) and rings_coords[DEFAULT_RING_INDEX]:
              selected_index = DEFAULT_RING_INDEX; logging.warning(f"Using default ring {selected_index}.")
         else:
              logging.warning(f"Default ring {DEFAULT_RING_INDEX} also empty/invalid. Searching...")
              for idx, coords in enumerate(rings_coords):
                   if coords: selected_index = idx; logging.warning(f"Using first non-empty ring {selected_index}."); break
              else: logging.critical("All rings are empty!"); raise ValueError("All rings are empty.")

    logging.debug(f"Ring selection time: {time.time() - func_start_time:.4f}s")
    return selected_index

# ============================================================
# --- Функции Работы с Видео (I/O) ---
# ============================================================

def read_video(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Считывает видео и возвращает список кадров BGR и FPS."""
    func_start_time = time.time()
    logging.info(f"Reading video from: {video_path}")
    frames = []
    fps = float(FPS) # Значение по умолчанию
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return frames, fps

        fps_read = cap.get(cv2.CAP_PROP_FPS)
        if fps_read > 0: fps = float(fps_read); logging.info(f"Detected FPS: {fps:.2f}")
        else: logging.warning(f"Failed to get FPS. Using default: {fps}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Video resolution: {width}x{height}")

        frame_count = 0; none_frame_count = 0; invalid_shape_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: logging.info(f"End of video stream after {frame_count} frames."); break
            if frame is not None:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    if frame.shape[0] == height and frame.shape[1] == width:
                        frames.append(frame); frame_count += 1
                        if frame_count % 100 == 0: logging.debug(f"Read {frame_count} frames...")
                    else: logging.warning(f"Frame {frame_count} shape {frame.shape} != expected ({height},{width},3). Skipping."); invalid_shape_count += 1
                else: logging.warning(f"Frame {frame_count} not 3-channel (shape: {frame.shape}). Skipping."); invalid_shape_count += 1
            else: logging.warning(f"Received None frame near index {frame_count + none_frame_count}."); none_frame_count += 1
        logging.info(f"Finished reading. Frames read: {len(frames)}. Skipped None: {none_frame_count}, Invalid shape: {invalid_shape_count}")
    except Exception as e: logging.error(f"Exception during video reading: {e}", exc_info=True)
    finally:
        if cap is not None and cap.isOpened(): cap.release(); logging.debug("Video capture released.")
    if not frames: logging.error(f"No valid frames read from {video_path}")
    logging.debug(f"Read video time: {time.time() - func_start_time:.4f}s")
    return frames, fps

def write_video(frames: List[np.ndarray], out_path: str, fps: float):
    """Записывает список кадров BGR в выходное видео."""
    func_start_time = time.time()
    if not frames: logging.error("No frames provided to write video."); return
    logging.info(f"Starting video writing to: {out_path} (FPS: {fps:.2f})")
    writer = None
    try:
        first_valid_frame = next((f for f in frames if f is not None and f.ndim==3 and f.shape[2]==3), None)
        if first_valid_frame is None: logging.error("No valid frames to determine size."); return
        h, w, c = first_valid_frame.shape
        logging.info(f"Output resolution: {w}x{h}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened(): logging.error(f"Failed to create VideoWriter for {out_path}"); return

        written_count = 0; skipped_count = 0
        start_write_loop = time.time()
        for i, frame in enumerate(frames):
            if frame is not None and frame.shape == (h, w, c) and frame.dtype == np.uint8:
                writer.write(frame); written_count += 1
                if written_count % 100 == 0: logging.debug(f"Written {written_count} frames...")
            else:
                 shape_info = frame.shape if frame is not None else 'None'; dtype_info = frame.dtype if frame is not None else 'N/A'
                 logging.warning(f"Skipping invalid frame #{i}. Shape: {shape_info}, Dtype: {dtype_info}. Writing black frame.")
                 writer.write(np.zeros((h, w, 3), dtype=np.uint8)); skipped_count += 1
        logging.debug(f"Write loop time: {time.time() - start_write_loop:.4f}s")
        logging.info(f"Finished writing. Frames written: {written_count}, Skipped/Replaced: {skipped_count}")
    except Exception as e: logging.error(f"Exception during video writing: {e}", exc_info=True)
    finally:
        if writer is not None and writer.isOpened(): writer.release(); logging.debug("Video writer released.")
    logging.debug(f"Write video total time: {time.time() - func_start_time:.4f}s")

# ============================================================
# --- ЛОГИКА ВСТРАИВАНИЯ (Embed) ---
# ============================================================

def embed_frame_pair(
    frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bit: int,
    n_rings: int = N_RINGS, adaptive_ring: bool = ADAPTIVE_RING_SELECTION,
    ring_selection_metric: str = RING_SELECTION_METRIC,
    default_ring_index: int = DEFAULT_RING_INDEX, frame_number: int = 0,
    visualize_mask: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    func_start_time = time.time()
    logging.debug(f"--- Embed Start: Pair {frame_number//2} (Frame {frame_number}) ---")
    try:
        # 1. Цвет -> YCrCb
        color_conv_start = time.time()
        try:
            if frame1_bgr is None or frame2_bgr is None: raise ValueError("Input frame is None")
            frame1_ycrcb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb)
            frame2_ycrcb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2YCrCb)
        except (cv2.error, ValueError) as e: logging.error(f"Color conversion failed: {e}"); return frame1_bgr, frame2_bgr
        Y1_orig = frame1_ycrcb[:, :, 0]; Cr1 = frame1_ycrcb[:, :, 1]; Cb1 = frame1_ycrcb[:, :, 2]
        Y2_orig = frame2_ycrcb[:, :, 0]; Cr2 = frame2_ycrcb[:, :, 1]; Cb2 = frame2_ycrcb[:, :, 2]
        Y1 = Y1_orig.astype(np.float32) / 255.0; Y2 = Y2_orig.astype(np.float32) / 255.0
        logging.debug(f"Color conversion time: {time.time() - color_conv_start:.4f}s")

        # 2. DTCWT
        dtcwt_start = time.time()
        pyr1 = dtcwt_transform(Y1); pyr2 = dtcwt_transform(Y2)
        if pyr1 is None or pyr2 is None: logging.error("DTCWT failed."); return frame1_bgr, frame2_bgr
        L1 = pyr1.lowpass.copy(); L2 = pyr2.lowpass.copy()
        logging.debug(f"DTCWT time: {time.time() - dtcwt_start:.4f}s. L1 shape: {L1.shape}")

        # 3. Выбор кольца
        ring_select_start = time.time()
        rings1_coords = ring_division(L1, n_rings=n_rings)
        current_ring_index: int
        if adaptive_ring:
            current_ring_index = select_embedding_ring(L1, rings1_coords, metric=ring_selection_metric)
        else: # Используем фиксированный индекс
            current_ring_index = default_ring_index
            logging.info(f"Using fixed ring index: {current_ring_index}")
            if not (0 <= current_ring_index < n_rings and current_ring_index < len(rings1_coords) and rings1_coords[current_ring_index]):
                 logging.error(f"Default ring {current_ring_index} is invalid or empty. Searching..."); # ... (код фоллбэка) ...
                 found = False
                 for idx, coords in enumerate(rings1_coords):
                      if coords: current_ring_index = idx; logging.warning(f"Using first non-empty ring {current_ring_index}."); found=True; break
                 if not found: logging.critical("All rings empty!"); return frame1_bgr, frame2_bgr

        logging.info(f"Ring index for pair {frame_number//2}: {current_ring_index}")
        coords_1 = rings1_coords[current_ring_index]
        rings2_coords = ring_division(L2, n_rings=n_rings)
        if current_ring_index >= len(rings2_coords) or not rings2_coords[current_ring_index]:
             logging.error(f"Selected ring {current_ring_index} empty/invalid in L2. Skipping."); return frame1_bgr, frame2_bgr
        coords_2 = rings2_coords[current_ring_index]
        logging.debug(f"Ring selection time: {time.time() - ring_select_start:.4f}s")

        # --- ВИЗУАЛИЗАЦИЯ МАСКИ ---
        if visualize_mask and (frame_number < 6 or (40 < frame_number < 46) or (88 < frame_number < 94)):
            logging.debug(f"Visualizing mask for frame {frame_number}, ring {current_ring_index}")
            mask = np.zeros_like(L1); count = 0
            for r, c in coords_1:
                if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]: mask[r, c] = 1.0; count += 1
            logging.debug(f"Drew {count} pixels for mask.")
            mask_display = (mask * 255).astype(np.uint8)
            try: cv2.imshow(f"Ring Mask (F{frame_number}, R{current_ring_index})", mask_display); cv2.waitKey(1)
            except cv2.error as e: logging.error(f"Failed to display mask: {e}")
        # --- Конец Визуализации ---

        ring_vals_1 = np.array([L1[r, c] for (r, c) in coords_1], dtype=np.float32)
        ring_vals_2 = np.array([L2[r, c] for (r, c) in coords_2], dtype=np.float32)
        if len(ring_vals_1) == 0 or len(ring_vals_2) == 0: logging.error("Empty ring values. Skipping."); return frame1_bgr, frame2_bgr

        # 4. Адаптивная альфа
        alpha = compute_adaptive_alpha_entropy(ring_vals_1, current_ring_index, frame_number)
        timestamp = frame_number / FPS
        logging.info(f"Embedding Bit={bit}, Alpha={alpha:.4f}, Time={timestamp:.3f}s")

        # 5. DCT, SVD, Модификация S
        svd_mod_start = time.time()
        dct1 = dct_1d(ring_vals_1); dct2 = dct_1d(ring_vals_2)
        try:
            U1, S1_vals, Vt1 = svd(dct1.reshape(-1, 1), full_matrices=False)
            U2, S2_vals, Vt2 = svd(dct2.reshape(-1, 1), full_matrices=False)
        except np.linalg.LinAlgError as e: logging.error(f"SVD failed: {e}. Skipping."); return frame1_bgr, frame2_bgr
        s1 = S1_vals[0] if S1_vals.size > 0 else 0.0; s2 = S2_vals[0] if S2_vals.size > 0 else 0.0
        logging.debug(f"SVD before: s1={s1:.4f}, s2={s2:.4f}")
        eps = 1e-12; ratio = s1 / (s2 + eps); new_s1, new_s2 = s1, s2
        alpha_sq = alpha * alpha; inv_alpha = 1.0 / (alpha + eps)
        modified = False
        if bit == 0:
            if ratio < alpha:
                new_s1 = (s1 * alpha_sq + alpha * s2) / (alpha_sq + 1.0); new_s2 = (alpha * s1 + s2) / (alpha_sq + 1.0); modified = True
        else: # bit == 1
            if ratio >= inv_alpha:
                 new_s1 = (s1 + alpha * s2) / (1.0 + alpha_sq); new_s2 = (alpha * s1 + alpha_sq * s2) / (1.0 + alpha_sq); modified = True
        logging.info(f"SVD Modification Applied: {modified}. New s1={new_s1:.4f}, s2={new_s2:.4f}")

        # 6. Восстановление DCT -> IDCT
        new_S1_diag = np.diag([new_s1] if S1_vals.size > 0 else [0.0]); new_S2_diag = np.diag([new_s2] if S2_vals.size > 0 else [0.0])
        dct1_modified = (U1 @ new_S1_diag @ Vt1).flatten(); dct2_modified = (U2 @ new_S2_diag @ Vt2).flatten()
        ring_vals_1_mod = idct_1d(dct1_modified); ring_vals_2_mod = idct_1d(dct2_modified)
        logging.debug(f"SVD mod & IDCT time: {time.time() - svd_mod_start:.4f}s")

        # 7. Запись в L1, L2
        write_back_start = time.time()
        modified_count1 = 0; modified_count2 = 0
        for idx, (r, c) in enumerate(coords_1):
            if idx < len(ring_vals_1_mod):
                 if 0 <= r < L1.shape[0] and 0 <= c < L1.shape[1]: L1[r, c] = ring_vals_1_mod[idx]; modified_count1 += 1
        for idx, (r, c) in enumerate(coords_2):
             if idx < len(ring_vals_2_mod):
                  if 0 <= r < L2.shape[0] and 0 <= c < L2.shape[1]: L2[r, c] = ring_vals_2_mod[idx]; modified_count2 += 1
        logging.debug(f"Write back time: {time.time() - write_back_start:.4f}s. Pixels modified: {modified_count1}(F1), {modified_count2}(F2)")

        # 8. Обратный DTCWT
        inv_dtcwt_start = time.time()
        pyr1.lowpass = L1; pyr2.lowpass = L2
        Y1_mod = dtcwt_inverse(pyr1); Y2_mod = dtcwt_inverse(pyr2)
        if Y1_mod is None or Y2_mod is None: logging.error("Inverse DTCWT failed."); return frame1_bgr, frame2_bgr
        logging.debug(f"Inverse DTCWT time: {time.time() - inv_dtcwt_start:.4f}s.")

        # Проверка и ресайз Y_mod
        resize_needed = False
        if Y1_mod.shape != Y1_orig.shape: Y1_mod = cv2.resize(Y1_mod, (Y1_orig.shape[1], Y1_orig.shape[0]), cv2.INTER_LINEAR); resize_needed = True
        if Y2_mod.shape != Y2_orig.shape: Y2_mod = cv2.resize(Y2_mod, (Y2_orig.shape[1], Y2_orig.shape[0]), cv2.INTER_LINEAR); resize_needed = True
        if resize_needed: logging.warning("Resized Y_mod to match original Y dimensions!")

        # 9. Сборка BGR
        reconstruct_start = time.time()
        Y1_mod_scaled = np.clip(Y1_mod * 255.0, 0, 255).astype(np.uint8)
        Y2_mod_scaled = np.clip(Y2_mod * 255.0, 0, 255).astype(np.uint8)
        Cr1 = Cr1.astype(np.uint8); Cb1 = Cb1.astype(np.uint8); Cr2 = Cr2.astype(np.uint8); Cb2 = Cb2.astype(np.uint8)
        new_ycrcb1 = cv2.merge([Y1_mod_scaled, Cr1, Cb1]); new_ycrcb2 = cv2.merge([Y2_mod_scaled, Cr2, Cb2])
        frame1_mod_bgr = cv2.cvtColor(new_ycrcb1, cv2.COLOR_YCrCb2BGR)
        frame2_mod_bgr = cv2.cvtColor(new_ycrcb2, cv2.COLOR_YCrCb2BGR)
        logging.debug(f"Final BGR reconstruction time: {time.time() - reconstruct_start:.4f}s")

        total_pair_time = time.time() - func_start_time
        logging.info(f"--- Embed Finish: Pair {frame_number//2}. Total Time: {total_pair_time:.4f} sec ---")
        return frame1_mod_bgr, frame2_mod_bgr

    except Exception as e:
        logging.error(f"!!! UNHANDLED EXCEPTION in embed_frame_pair (Frame {frame_number}): {e}", exc_info=True)
        return frame1_bgr, frame2_bgr

# --- Обертка для потоков Embed ---
def _embed_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    idx1 = args['idx1']
    try:
        f1_mod, f2_mod = embed_frame_pair(
            args['frame1'], args['frame2'], args['bit'], args['n_rings'], args['adaptive_ring'],
            args['ring_selection_metric'], args['default_ring_index'], args['frame_number'],
            visualize_mask=args.get('visualize_mask', False)
        )
        if f1_mod is None or f2_mod is None: return idx1, None, None
        return idx1, f1_mod, f2_mod
    except Exception as e: logging.error(f"Exception in worker for pair {idx1//2}: {e}", exc_info=True); return idx1, None, None

# --- Основная функция Embed ---
def embed_watermark_in_video(
    frames: List[np.ndarray], watermark_bits: List[int], n_rings: int = N_RINGS,
    adaptive_ring_selection: bool = ADAPTIVE_RING_SELECTION,
    ring_selection_metric: str = RING_SELECTION_METRIC,
    default_ring_index: int = DEFAULT_RING_INDEX, fps: float = FPS,
    max_workers: Optional[int] = MAX_WORKERS, visualize_masks: bool = False
):
    logging.info(f"Starting embedding {len(watermark_bits)} bits. Visualize masks: {visualize_masks}")
    start_time = time.time()
    watermarked_frames = frames[:]
    num_frames = len(frames)
    pair_count = num_frames // 2
    bits_to_embed = len(watermark_bits)
    num_pairs_to_process = min(pair_count, bits_to_embed)
    if num_pairs_to_process == 0: logging.warning("No pairs to process."); return watermarked_frames

    tasks_args = []
    for pair_idx in range(num_pairs_to_process):
        idx1 = 2 * pair_idx; idx2 = idx1 + 1
        if idx2 >= num_frames: continue
        args = {
            'idx1': idx1, 'frame1': frames[idx1], 'frame2': frames[idx2],
            'bit': watermark_bits[pair_idx], 'n_rings': n_rings,
            'adaptive_ring': adaptive_ring_selection,
            'ring_selection_metric': ring_selection_metric,
            'default_ring_index': default_ring_index, 'frame_number': idx1,
            'visualize_mask': visualize_masks
        }
        tasks_args.append(args)

    results = {}
    processed_count = 0; error_count = 0
    try:
        logging.info(f"Submitting {len(tasks_args)} tasks to ThreadPoolExecutor (max_workers={max_workers})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_embed_frame_pair_worker, arg): arg['idx1'] for arg in tasks_args}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
                idx1 = future_to_idx[future]
                try:
                    _, f1_mod, f2_mod = future.result()
                    if f1_mod is not None and f2_mod is not None:
                        results[idx1] = (f1_mod, f2_mod); processed_count += 1
                        logging.debug(f"Pair {idx1//2} completed successfully ({i+1}/{len(tasks_args)}).")
                    else: error_count += 1; logging.error(f"Pair {idx1//2} failed (returned None).")
                except Exception as exc: error_count += 1; logging.error(f'Pair {idx1//2} generated exception: {exc}', exc_info=False)

        logging.info(f"ThreadPoolExecutor finished. Successful: {processed_count}, Failed: {error_count}.")
        update_count = 0
        for idx1, (f1_mod, f2_mod) in results.items():
             idx2 = idx1 + 1
             if idx1 < len(watermarked_frames): watermarked_frames[idx1] = f1_mod; update_count +=1
             if idx2 < len(watermarked_frames): watermarked_frames[idx2] = f2_mod; update_count +=1
        logging.info(f"Applied results to {len(results)} pairs ({update_count} frames updated).")
    except Exception as e:
        logging.critical(f"CRITICAL ERROR in ThreadPoolExecutor: {e}", exc_info=True); return frames[:]

    if visualize_masks: logging.info("Destroying visualization windows."); cv2.destroyAllWindows()
    end_time = time.time()
    logging.info(f"Embedding finished. Total time: {end_time - start_time:.2f} sec.")
    return watermarked_frames

# ============================================================
# --- ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ (Только Встраивание) ---
# ============================================================

def main():
    main_start_time = time.time()
    input_video = "input.mp4"
    output_video = "watermarked_video.mp4" # Имя выходного файла

    logging.info("--- Starting Embedding Main Process ---")
    frames, input_fps = read_video(input_video)
    if not frames: logging.critical("Failed to read input video. Exiting."); return
    if input_fps <= 0: logging.warning(f"Using default FPS={FPS} for writing."); input_fps = float(FPS)

    num_pairs = len(frames) // 2
    watermark_length = min(num_pairs, 64) # Длина ВЗ (можно изменить)
    # Генерируем ВЗ
    watermark_bits = [random.randint(0, 1) for _ in range(watermark_length)]
    # watermark_bits = [(i % 2) for i in range(watermark_length)] # Или простой 0101...
    logging.info(f"Using watermark ({watermark_length} bits): {''.join(map(str, watermark_bits))}")

    # Включаем визуализацию маски для первых кадров?
    enable_mask_visualization = False # Установите True, чтобы видеть маски
    logging.info(f"Mask visualization: {'ENABLED' if enable_mask_visualization else 'DISABLED'}")

    # Выполняем встраивание
    watermarked_frames = embed_watermark_in_video(
        frames=frames, watermark_bits=watermark_bits, n_rings=N_RINGS,
        adaptive_ring_selection=ADAPTIVE_RING_SELECTION, # Из констант
        ring_selection_metric=RING_SELECTION_METRIC,     # Из констант
        default_ring_index=DEFAULT_RING_INDEX,         # Из констант
        fps=input_fps, max_workers=MAX_WORKERS,
        visualize_masks=enable_mask_visualization
    )

    # Записываем результат
    if watermarked_frames: # Проверяем, что получили кадры
        write_video(watermarked_frames, output_video, fps=input_fps)
        logging.info(f"Watermarked video saved to: {output_video}")
    else:
        logging.error("Embedding resulted in no frames. Output video not saved.")

    logging.info("--- Embedding Main Process Finished ---")
    total_script_time = time.time() - main_start_time
    logging.info(f"--- Total Embedder Script Time: {total_script_time:.2f} sec ---")
    print(f"Embedding finished. Output: {output_video}. Check log: {LOG_FILENAME}")

# --- Запуск ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception in main (Embedder): {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}. Check the log file: {LOG_FILENAME}")