import cProfile
import concurrent
import gc
import math
import pstats
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import random
import logging
import time
import json
import os
import hashlib
# import imagehash

# from line_profiler import profile
# import cProfile
# import pstats
import torch
import torch.nn.functional as F
from galois import BCH
from line_profiler import profile

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    class DTCWTForward: pass
    class DTCWTInverse: pass
    logging.error("ОШИБКА: Библиотека pytorch_wavelets не найдена! Установите: pip install pytorch_wavelets")
try:
    import torch_dct as dct_torch # Импортируем под псевдонимом
    TORCH_DCT_AVAILABLE = True
except ImportError:
    TORCH_DCT_AVAILABLE = False
    logging.error("ОШИБКА: Библиотека torch-dct не найдена! Установите: pip install torch-dct")


from typing import List, Tuple, Optional, Dict, Any, Iterator
import uuid
from math import ceil
from collections import Counter
import sys

# --- Galois импорты ---
try:
    import av
    from av import FFmpegError, VideoFrame
    from av import EOFError as FFmpegEOFError
    from av import ValueError as FFmpegValueError
    PYAV_AVAILABLE = True
    logging.info("PyAV library imported successfully.")

except ImportError:
    PYAV_AVAILABLE = False
    logging.error("PyAV library not found! Install it: pip install av")
    class av_dummy:
        class VideoFrame: pass
        class AudioFrame: pass
        class Packet: pass
        class TimeBase: pass
        class container:
            class Container: pass
        FFmpegError = Exception
        EOFError = EOFError
        ValueError = ValueError
        NotFoundError = Exception
    av = av_dummy
    FFmpegError = Exception
    FFmpegEOFError = EOFError
    FFmpegValueError = ValueError

try:
    import galois
    BCH_TYPE = galois.BCH; GALOIS_IMPORTED = True; logging.info("galois library imported.")
except ImportError:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.info("galois library not found.")
except Exception as import_err:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.error(f"Galois import error: {import_err}", exc_info=True)

# --- Глобальные Параметры (СОГЛАСОВАНЫ с Embedder) ---
LAMBDA_PARAM: float = 0.06
ALPHA_MIN: float = 1.09
ALPHA_MAX: float = 1.31
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection'
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 9
FPS: int = 30
LOG_FILENAME: str = 'watermarking_extract_pytorch_COMPRESSED.log'
INPUT_EXTENSION: str = '.mp4'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS_EXTRACT: Optional[int] = 14
expect_hybrid_ecc_global = True
MAX_TOTAL_PACKETS_global = 18
CV2_AVAILABLE = True
BCH_CODE_OBJECT: Optional[BCH_TYPE] = None
GALOIS_AVAILABLE = False


if GALOIS_IMPORTED:
    _test_bch_ok = False; _test_decode_ok = False
    try:
        _test_m = BCH_M; _test_t = BCH_T; _test_n = (1 << _test_m) - 1; _test_d = 2 * _test_t + 1
        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)
        if _test_t == 5: expected_k = 215
        elif _test_t == 7: expected_k = 201
        elif _test_t == 9: expected_k = 187 #Для t=9
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

# --- Базовые Функции ---

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

# --- Функции, работающие с NumPy ---
def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    """
        Вычисляет шенноновскую энтропию и энтропию столкновений (Реньи 2-го порядка
        в вашей старой реализации) для одномерного NumPy массива значений пикселей кольца.

        Args:
            rv: Одномерный NumPy массив значений пикселей (предположительно нормализованных от 0 до 1).
            fn: Номер кадра (для логирования, опционально).
            ri: Индекс кольца (для логирования, опционально).

        Returns:
            Кортеж (float, float): (шенноновская_энтропия, энтропия_столкновений).
                                   Возвращает (0.0, 0.0), если массив пуст или содержит константу.
        """
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
    """
        Генерирует детерминированный псевдослучайный набор индексов колец
        на основе индекса пары кадров.

        Args:
            pi: Индекс пары кадров (используется как сид для PRNG).
            nr: Общее количество доступных колец (например, N_RINGS).
            ps: Размер пула кандидатов колец, который нужно сгенерировать
                (например, CANDIDATE_POOL_SIZE).

        Returns:
            List[int]: Список псевдослучайных индексов колец без повторений.
                       Длина списка равна `ps` (или `nr`, если `ps > nr`).
                       Пустой список, если `ps <= 0`.
        """

    if ps <= 0: return []
    if ps > nr: ps = nr
    seed_str = str(pi).encode('utf-8'); hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big'); prng = random.Random(seed_int)
    try: candidate_indices = prng.sample(range(nr), ps)
    except ValueError: candidate_indices = list(range(nr)); prng.shuffle(candidate_indices); candidate_indices = candidate_indices[:ps]
    logging.debug(f"[P:{pi}] Candidates: {candidate_indices}");
    return candidate_indices

def bits_to_bytes_strict(bit_list: List[int]) -> Optional[bytes]:
    """
    Конвертирует список бит (только 0 или 1) в байтовую строку.
    Ожидает, что длина списка кратна 8.
    """
    num_bits = len(bit_list)

    if not all(b in (0, 1) for b in bit_list):
        logging.error("bits_to_bytes_strict: список содержит невалидные биты (не 0 или 1).")
        return None

    if num_bits == 0:
        return b''

    if num_bits % 8 != 0:
        logging.error(f"bits_to_bytes_strict: длина списка бит ({num_bits}) не кратна 8.")
        return None

    byte_array = bytearray()
    for i in range(0, num_bits, 8):
        byte_chunk = bit_list[i:i+8]
        try:
            byte_val = int("".join(map(str, byte_chunk)), 2)
            byte_array.append(byte_val)
        except ValueError: # Маловероятно, если биты уже проверены
            logging.error(f"bits_to_bytes_strict: Не удалось конвертировать битовый чанк: {byte_chunk}")
            return None
    return bytes(byte_array)

def get_byte_from_bits(bits_payload: List[int], byte_index: int) -> Optional[int]:
    start_bit_index = byte_index * 8
    end_bit_index = start_bit_index + 8
    if start_bit_index < 0 or end_bit_index > len(bits_payload):
        return None
    byte_bits_str = "".join(map(str, bits_payload[start_bit_index:end_bit_index]))
    try:
        return int(byte_bits_str, 2)
    except ValueError:
        return None

def get_bits_from_byte_value(byte_value: int) -> List[int]:
    if not (0 <= byte_value <= 255):
        raise ValueError("Значение байта должно быть в диапазоне 0-255")
    return [int(bit) for bit in format(byte_value, '08b')]


def ultimate_voting_strategy(
        valid_packets_info: List[Dict[str, Any]],
        # {'bits': List[int], 'hex': str, 'type': str, 'corrected_errors': int}
        payload_len_bytes: int
) -> List[str]:  # Возвращает список из 1 или 2 HEX-кандидатов

    payload_len_bits = payload_len_bytes * 8
    num_valid_packets = len(valid_packets_info)

    if num_valid_packets == 0:
        logging.error("Ultimate Voting: Нет валидных пакетов для голосования.")
        print("ОШИБКА: Нет валидных пакетов для голосования.")
        return []

    logging.info(f"Ultimate Voting: Голосование по {num_valid_packets} валидным пакетам...")
    print("\n--- Этап 1: Побитовое Голосование (Ultimate) ---")
    print(f"{'Bit Pos':<8} | {'Votes 0':<8} | {'Votes 1':<8} | {'Bit Winner':<11}")
    print("-" * (8 + 9 + 9 + 12))

    bit_voting_results: List[Optional[int]] = [None] * payload_len_bits

    for j in range(payload_len_bits):
        votes_for_0 = 0
        votes_for_1 = 0
        for packet_info in valid_packets_info:
            # Предполагается, что packet_info['bits'] всегда корректной длины payload_len_bits
            # и не содержит None на этом этапе
            if packet_info['bits'][j] == 1:
                votes_for_1 += 1
            elif packet_info['bits'][j] == 0:
                votes_for_0 += 1
            else:  # На случай, если в 'bits' попал None или другое значение
                logging.error(
                    f"Bit {j}, Packet {valid_packets_info.index(packet_info)}: Невалидное значение бита {packet_info['bits'][j]}. Пропуск.")
                continue  # Пропускаем этот бит из этого пакета

        winner_this_bit: Optional[int] = None
        if votes_for_1 > votes_for_0:
            winner_this_bit = 1
        elif votes_for_0 > votes_for_1:
            winner_this_bit = 0
        else:
            winner_this_bit = None  # Ничья на битовом уровне
            logging.info(f"Bit {j}: Ничья на побитовом голосовании ({votes_for_0}v{votes_for_1}).")

        bit_voting_results[j] = winner_this_bit
        winner_str = str(winner_this_bit) if winner_this_bit is not None else "TIE"
        print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {winner_str:<11}")
    print("-" * (8 + 9 + 9 + 12))

    if not any(b is None for b in bit_voting_results):
        logging.info("Побитовое голосование дало однозначный результат для всех бит.")
        # Гарантируем, что None нет, для mypy и для bits_to_bytes_strict
        final_bits_definitively_int: List[int] = [b for b in bit_voting_results if b is not None]
        if len(final_bits_definitively_int) != payload_len_bits:  # Должно быть невозможно, если any(None) is False
            logging.error("Логическая ошибка: все биты определены, но длина не совпадает.")
            return []

        final_bytes = bits_to_bytes_strict(final_bits_definitively_int)
        if final_bytes and len(final_bytes) == payload_len_bytes:
            hex_candidate = final_bytes.hex()
            logging.info(f"Сформирован единственный кандидат ID: {hex_candidate}")
            print(f"\n--- Финальные Кандидаты ---\nКандидат 1: {hex_candidate}")
            return [hex_candidate]
        else:
            logging.error("Ошибка конвертации бит в байты после побитового голосования без ничьих.")
            return []

    logging.info(f"Обнаружены ничьи на битовом уровне. Переход к голосованию по байтам.")
    print(f"\n--- Этап 2: Голосование по Байтам для Разрешения Ничьих на Битах ---")

    candidate1_bits: List[Optional[int]] = list(bit_voting_results)
    byte_level_tie_resolution: Dict[int, List[int]] = {}

    for byte_idx in range(payload_len_bytes):
        has_tie_in_this_byte = False
        for k_check_tie in range(8):
            if candidate1_bits[byte_idx * 8 + k_check_tie] is None:
                has_tie_in_this_byte = True
                break

        if not has_tie_in_this_byte:
            continue

        logging.info(f"  Разрешение для байта {byte_idx} (биты {byte_idx * 8}-{byte_idx * 8 + 7})...")

        byte_values_from_packets: List[int] = []
        for packet_info in valid_packets_info:
            byte_val = get_byte_from_bits(packet_info['bits'], byte_idx)
            if byte_val is not None:
                byte_values_from_packets.append(byte_val)

        if not byte_values_from_packets:
            logging.error(f"Нет валидных байт для голосования по байту {byte_idx}. Невозможно разрешить битовые ничьи.")
            return []

        byte_counts = Counter(byte_values_from_packets)
        most_common_bytes_tuples = byte_counts.most_common()

        logging.debug(f"    Голоса за байт {byte_idx}: {most_common_bytes_tuples}")

        if not most_common_bytes_tuples:
            logging.error(f"Нет информации о голосах для байта {byte_idx}.")
            return []

        max_vote_count = most_common_bytes_tuples[0][1]
        winning_byte_values_for_pos = sorted([b_val for b_val, count in most_common_bytes_tuples if
                                              count == max_vote_count])  # Сортируем для консистентности

        byte_level_tie_resolution[byte_idx] = winning_byte_values_for_pos

        if len(winning_byte_values_for_pos) == 1:
            winner_byte_val = winning_byte_values_for_pos[0]
            winner_byte_bits = get_bits_from_byte_value(winner_byte_val)
            logging.info(f"    Байт {byte_idx}: Однозначный победитель на байтовом уровне: {hex(winner_byte_val)}.")
            for k_fill in range(8):
                bit_global_idx = byte_idx * 8 + k_fill
                if candidate1_bits[bit_global_idx] is None:
                    candidate1_bits[bit_global_idx] = winner_byte_bits[k_fill]
        else:
            logging.warning(
                f"    Байт {byte_idx}: Ничья на байтовом уровне! Кандидаты: {[hex(b) for b in winning_byte_values_for_pos]}.")

    final_candidate1_bits: List[int] = [0] * payload_len_bits
    ambiguous_byte_indices_for_cand2: List[int] = []

    for byte_idx_cand1 in range(payload_len_bytes):
        is_this_byte_resolved = True
        for k_cand1_check in range(8):
            if candidate1_bits[byte_idx_cand1 * 8 + k_cand1_check] is None:
                is_this_byte_resolved = False
                break

        if is_this_byte_resolved:
            for k_cand1_copy in range(8):
                final_candidate1_bits[byte_idx_cand1 * 8 + k_cand1_copy] = candidate1_bits[
                    byte_idx_cand1 * 8 + k_cand1_copy]  # type: ignore
        else:  # Биты для этого байта не были разрешены, значит, была ничья на байтовом уровне
            if byte_idx_cand1 not in byte_level_tie_resolution or not byte_level_tie_resolution[byte_idx_cand1]:
                logging.error(f"Крит. ошибка: байт {byte_idx_cand1} не разрешен, нет инфо о байт. голосовании.")
                return []

            primary_winner_byte_val = byte_level_tie_resolution[byte_idx_cand1][0]  # Берем первый из победителей
            primary_winner_byte_bits = get_bits_from_byte_value(primary_winner_byte_val)
            logging.info(
                f"  Для Кандидата 1, неразрешенный байт {byte_idx_cand1} устанавливается в {hex(primary_winner_byte_val)} (первый из байт. ничьей).")
            for k_fill_cand1 in range(8):
                final_candidate1_bits[byte_idx_cand1 * 8 + k_fill_cand1] = primary_winner_byte_bits[k_fill_cand1]

            if len(byte_level_tie_resolution[byte_idx_cand1]) > 1:
                ambiguous_byte_indices_for_cand2.append(byte_idx_cand1)

    if any(b is None for b in final_candidate1_bits):  # Этой ситуации быть не должно
        logging.error("Не все биты в Кандидате 1 были разрешены после всех этапов! Логическая ошибка.")
        return []

    candidate1_bytes_obj = bits_to_bytes_strict(final_candidate1_bits)
    if not candidate1_bytes_obj or len(candidate1_bytes_obj) != payload_len_bytes:
        logging.error("Не удалось корректно сформировать байты для Кандидата 1.")
        return []

    final_hex_candidates = [candidate1_bytes_obj.hex()]
    logging.info(f"Сформирован первый кандидат ID: {final_hex_candidates[0]}")
    print(f"\n--- Финальные Кандидаты ---")
    print(f"Кандидат 1: {final_hex_candidates[0]}")

    if len(ambiguous_byte_indices_for_cand2) == 1:
        the_ambiguous_byte_idx = ambiguous_byte_indices_for_cand2[0]
        options_for_ambiguous_byte = byte_level_tie_resolution[the_ambiguous_byte_idx]

        if len(options_for_ambiguous_byte) == 2:
            logging.info(f"Обнаружена одна неоднозначная байтовая позиция ({the_ambiguous_byte_idx}) "
                         f"с двумя вариантами: {[hex(b) for b in options_for_ambiguous_byte]}. Формирование второго кандидата.")

            final_candidate2_bits = list(final_candidate1_bits)
            secondary_winner_byte_val = options_for_ambiguous_byte[1]
            secondary_winner_byte_bits = get_bits_from_byte_value(secondary_winner_byte_val)

            logging.info(
                f"  Для Кандидата 2, байт {the_ambiguous_byte_idx} устанавливается в {hex(secondary_winner_byte_val)}.")
            for k_amb in range(8):
                bit_global_idx_amb = the_ambiguous_byte_idx * 8 + k_amb
                final_candidate2_bits[bit_global_idx_amb] = secondary_winner_byte_bits[k_amb]

            candidate2_bytes_obj = bits_to_bytes_strict(final_candidate2_bits)
            if candidate2_bytes_obj and len(candidate2_bytes_obj) == payload_len_bytes:
                if candidate2_bytes_obj.hex() != final_hex_candidates[0]:  # Убедимся, что он отличается
                    final_hex_candidates.append(candidate2_bytes_obj.hex())
                    print(f"Кандидат 2: {final_hex_candidates[1]}")
                else:
                    logging.info("Второй кандидат идентичен первому, не добавляем.")
            else:
                logging.warning("Не удалось сформировать байты для Кандидата 2.")
        elif len(options_for_ambiguous_byte) > 2:
            logging.warning(
                f"Байт {the_ambiguous_byte_idx} имеет >2 равновероятных вариантов. Выводим только первого кандидата.")
            print(
                f"  (Байт {the_ambiguous_byte_idx} имеет >2 равновероятных вариантов, представлен только первый кандидат)")
    elif len(ambiguous_byte_indices_for_cand2) > 1:
        logging.warning("Обнаружено несколько неоднозначных байтовых позиций. Выводим только первого кандидата.")
        print(f"  (Обнаружено несколько неоднозначных байтовых позиций, представлен только первый кандидат)")

    return final_hex_candidates


def decode_ecc(packet_bits_list: List[int], bch_code: Optional[BCH_TYPE], expected_data_len_bytes: int) -> Tuple[
    Optional[List[int]], int]:
    """
    Декодирует пакет бит с использованием предоставленного объекта BCH кода.
    Returns:
        Кортеж (Optional[List[int]], int):
            - Декодированная полезная нагрузка в виде списка бит, или None при ошибке.
            - Количество исправленных ошибок (int):
                0  - если ошибок не было или они не были обнаружены ECC (пакет прошел как есть).
                >0 - если ошибки были исправлены.
                -1 - если декодирование не удалось / ошибки неисправимы / ECC недоступен.
    """
    if not GALOIS_AVAILABLE or bch_code is None:
        # logging.error("ECC decode called but unavailable.") # Уже логируется на более высоком уровне
        return None, -1

    n_corrected = -1
    payload_len_bits = expected_data_len_bytes * 8

    try:
        # Проверка типов и атрибутов BCH_CODE_OBJECT перед использованием
        if not (hasattr(bch_code, 'n') and hasattr(bch_code, 'k') and hasattr(bch_code, 'field') and hasattr(bch_code,
                                                                                                             'decode')):
            logging.error("decode_ecc: Объект bch_code не имеет необходимых атрибутов (n, k, field, decode).")
            return None, -1

        n = bch_code.n
        k = bch_code.k

        if len(packet_bits_list) != n:
            logging.error(
                f"Decode ECC: Неверная длина входного пакета {len(packet_bits_list)} != {n} (ожидаемая длина кодового слова).")
            return None, -1
        if payload_len_bits > k:
            logging.error(
                f"Decode ECC: Ожидаемая длина полезной нагрузки ({payload_len_bits}) > информационной длины кода k ({k}).")
            return None, -1

        packet_bits_np = np.array(packet_bits_list, dtype=np.uint8)
        GF = bch_code.field
        rx_vec = GF(packet_bits_np)

        try:
            # errors=True заставляет decode возвращать (decoded_msg, num_errors_corrected)
            # или вызывать UncorrectableError
            corr_msg_vec, num_errors = bch_code.decode(rx_vec, errors=True)
            n_corrected = int(num_errors)  # num_errors может быть GF(0) или int
        except galois.errors.UncorrectableError:  # type: ignore
            logging.warning(f"Galois ECC: Неисправимые ошибки в пакете (длина {len(packet_bits_list)}).")
            return None, -1
        except Exception as e_dec:  # Другие возможные ошибки декодирования
            logging.error(f"Decode ECC: Ошибка во время bch_code.decode: {e_dec}", exc_info=True)
            return None, -1

        corr_k_bits_np = corr_msg_vec.view(np.ndarray).astype(np.uint8)

        if corr_k_bits_np.size < payload_len_bits:
            logging.error(
                f"Decode ECC: Длина декодированных информационных бит ({corr_k_bits_np.size}) < ожидаемой ({payload_len_bits}).")
            # Это может случиться, если k кода меньше payload_len_bits, но проверка выше должна была это поймать.
            # Или если что-то пошло не так с view().
            return None, n_corrected

        final_payload_bits = corr_k_bits_np[:payload_len_bits].tolist()

        logging.info(f"Galois ECC: Пакет успешно декодирован, исправлено ошибок: {n_corrected}.")
        return final_payload_bits, n_corrected

    except Exception as e:
        logging.error(f"Decode ECC: Неожиданная ошибка: {e}", exc_info=True)
        return None, -1


@profile
def extract_single_bit(L1_tensor: torch.Tensor, L2_tensor: torch.Tensor, ring_idx: int, n_rings: int, fn: int) -> Optional[int]:
    """
    Извлекает один бит (PyTorch DCT/SVD).
    (BN-PyTorch-Optimized-Corrected - V2 Log)
    """
    pair_index = fn // 2
    prefix = f"[BN P:{pair_index}, R:{ring_idx}]"

    try:
        # --- Шаг 1: Проверка входных данных ---
        if L1_tensor is None or L2_tensor is None or L1_tensor.shape != L2_tensor.shape \
           or not isinstance(L1_tensor, torch.Tensor) or not isinstance(L2_tensor, torch.Tensor) \
           or L1_tensor.ndim != 2 or L2_tensor.ndim != 2 \
           or not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
            logging.warning(f"{prefix} Invalid L1/L2 provided.")
            return None
        device = L1_tensor.device

        # --- Шаг 2: Кольцевое деление ---
        r1c = ring_division(L1_tensor, n_rings, fn)
        r2c = ring_division(L2_tensor, n_rings, fn + 1)
        if r1c is None or r2c is None \
           or not(0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
             logging.warning(f"{prefix} Invalid ring index or ring_division failed.")
             return None
        cd1_tensor = r1c[ring_idx]; cd2_tensor = r2c[ring_idx]
        min_ring_size = 10
        if cd1_tensor is None or cd2_tensor is None or cd1_tensor.shape[0] < min_ring_size or cd2_tensor.shape[0] < min_ring_size:
             logging.debug(f"{prefix} Ring coords None or ring too small (<{min_ring_size}).")
             return None

        # --- Шаг 3: Извлечение значений, DCT, SVD (НА ТЕНЗОРАХ) ---
        try:
            rows1, cols1 = cd1_tensor[:, 0], cd1_tensor[:, 1]
            rows2, cols2 = cd2_tensor[:, 0], cd2_tensor[:, 1]
            rv1_tensor = L1_tensor[rows1, cols1].to(dtype=torch.float32)
            rv2_tensor = L2_tensor[rows2, cols2].to(dtype=torch.float32)
            min_s = min(rv1_tensor.numel(), rv2_tensor.numel())
            if min_s == 0: return None
            if rv1_tensor.numel() != rv2_tensor.numel():
                rv1_tensor = rv1_tensor[:min_s]; rv2_tensor = rv2_tensor[:min_s]

            # *** ЛОГ: Статистика входных значений кольца ***
            logging.debug(f"{prefix} rv1 stats: size={rv1_tensor.numel()}, mean={rv1_tensor.mean():.6e}, std={rv1_tensor.std():.6e}")
            logging.debug(f"{prefix} rv2 stats: size={rv2_tensor.numel()}, mean={rv2_tensor.mean():.6e}, std={rv2_tensor.std():.6e}")

            # --- PyTorch DCT ---
            if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct not available")
            d1_tensor = dct1d_torch(rv1_tensor)
            d2_tensor = dct1d_torch(rv2_tensor)
            if not torch.isfinite(d1_tensor).all() or not torch.isfinite(d2_tensor).all(): return None
            # *** ЛОГ: Первые DCT коэффициенты ***
            logging.debug(f"{prefix} DCT done. d1[0]={d1_tensor[0]:.6e}, d2[0]={d2_tensor[0]:.6e}")


            # --- PyTorch SVD ---
            s1_tensor = svd_torch_s1(d1_tensor)
            s2_tensor = svd_torch_s1(d2_tensor)
            if s1_tensor is None or s2_tensor is None: return None
            # *** ЛОГ: Сингулярные числа (тензоры) с высокой точностью ***
            logging.debug(f"{prefix} SVD done. s1_tensor={s1_tensor.item():.8e}, s2_tensor={s2_tensor.item():.8e}")

            # Конвертируем в Python float для финального расчета и сравнения
            s1 = s1_tensor.item()
            s2 = s2_tensor.item()

        except RuntimeError as torch_err:
             logging.error(f"{prefix} PyTorch runtime error during Tensor DCT/SVD: {torch_err}", exc_info=True); return None
        except IndexError:
             logging.warning(f"{prefix} Index error getting ring tensor values."); return None
        except Exception as e:
             logging.error(f"{prefix} Error in Tensor DCT/SVD processing part: {e}", exc_info=True); return None

        # --- Шаг 4: Принятие решения ---
        eps = 1e-12; threshold = 1.0
        if abs(s2) < eps:
             logging.warning(f"{prefix} s2={s2:.2e} is close to zero. Unreliable ratio.")
             return None

        ratio = s1 / s2
        # --- Явное вычисление и логирование сравнения ---
        comparison_result = (ratio >= threshold)
        extracted_bit = 0 if comparison_result else 1

        # *** ЛОГ: Детальная информация для принятия решения ***
        logging.info(f"{prefix} Decision: s1={s1:.8e}, s2={s2:.8e}, ratio={ratio:.8f}, threshold={threshold}, comparison (ratio >= threshold)={comparison_result}, extracted_bit={extracted_bit}")

        return extracted_bit

    except Exception as e:
        logging.error(f"Unexpected error in extract_single_bit (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
        return None

def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
    """
    Обрабатывает батч задач извлечения: выполняет DTCWT один раз на пару,
    затем вызывает extract_single_bit для выбранных колец.
    """
    batch_results: Dict[int, List[Optional[int]]] = {}
    if not batch_args_list: return {}

    # Получаем общие параметры из первого аргумента
    args_example = batch_args_list[0]
    nr = args_example.get('n_rings', N_RINGS)
    nrtu = args_example.get('num_rings_to_use', NUM_RINGS_TO_USE)
    cps = args_example.get('candidate_pool_size', CANDIDATE_POOL_SIZE)
    ec = args_example.get('embed_component', EMBED_COMPONENT)
    device = args_example.get('device')
    dtcwt_fwd = args_example.get('dtcwt_fwd')

    # Проверяем наличие критически важных общих аргументов
    if device is None or dtcwt_fwd is None:
         logging.error("Device или DTCWTForward не переданы в _extract_batch_worker!")
         # Возвращаем пустой результат для всех пар в батче
         for args in batch_args_list:
              pair_idx = args.get('pair_idx', -1)
              if pair_idx != -1:
                   batch_results[pair_idx] = [None] * nrtu
         return batch_results

    # Итерируем по парам в батче
    for args in batch_args_list:
        pair_idx = args.get('pair_idx', -1)
        f1_bgr = args.get('frame1')
        f2_bgr = args.get('frame2')

        # Инициализация результата для текущей пары
        # Проверка, что nrtu валидно (на случай, если selected_rings будет короче)
        current_nrtu = args.get('num_rings_to_use', NUM_RINGS_TO_USE)
        extracted_bits_for_pair: List[Optional[int]] = [None] * current_nrtu

        # Проверяем индивидуальные аргументы пары
        if pair_idx == -1 or f1_bgr is None or f2_bgr is None:
            logging.error(f"Недостаточно аргументов для обработки pair_idx={pair_idx if pair_idx != -1 else 'unknown'}")
            batch_results[pair_idx] = extracted_bits_for_pair
            continue

        fn = 2 * pair_idx
        L1_tensor: Optional[torch.Tensor] = None
        L2_tensor: Optional[torch.Tensor] = None

        try:
            # --- Шаг 1: Преобразование цвета и DTCWT ---
            if not isinstance(f1_bgr, np.ndarray) or not isinstance(f2_bgr, np.ndarray):
                 logging.warning(f"[BN Worker P:{pair_idx}] Input frames not numpy arrays.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            # Конвертация BGR -> YCrCb -> Компонент -> Тензор [0,1]
            y1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
            y2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2YCrCb)
            # Используем .copy() для избежания проблем с read-only
            c1_np = y1[:, :, ec].copy().astype(np.float32) / 255.0
            c2_np = y2[:, :, ec].copy().astype(np.float32) / 255.0
            comp1_tensor = torch.from_numpy(c1_np).to(device=device)
            comp2_tensor = torch.from_numpy(c2_np).to(device=device)

            # Прямое DTCWT
            Yl_t, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
            Yl_t1, _ = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, fn + 1)

            if Yl_t is None or Yl_t1 is None:
                 logging.warning(f"[BN Worker P:{pair_idx}] DTCWT forward failed.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            # --- Извлекаем LL и проверяем ---
            if Yl_t.dim() > 2: L1_tensor = Yl_t.squeeze(0).squeeze(0)
            elif Yl_t.dim() == 2: L1_tensor = Yl_t
            else: raise ValueError(f"Invalid Yl_t dim: {Yl_t.dim()}")

            if Yl_t1.dim() > 2: L2_tensor = Yl_t1.squeeze(0).squeeze(0)
            elif Yl_t1.dim() == 2: L2_tensor = Yl_t1
            else: raise ValueError(f"Invalid Yl_t1 dim: {Yl_t1.dim()}")

            if not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
                 raise TypeError(f"L1/L2 not float! L1:{L1_tensor.dtype}, L2:{L2_tensor.dtype}")
            if L1_tensor.shape != L2_tensor.shape:
                 raise ValueError(f"L1/L2 shape mismatch! L1:{L1_tensor.shape}, L2:{L2_tensor.shape}")

            # --- Шаг 2: Выбор колец (используем L1_tensor) ---
            coords = ring_division(L1_tensor, nr, fn)
            if coords is None or len(coords) != nr:
                 logging.warning(f"[BN Worker P:{pair_idx}] Ring division failed.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
            # current_nrtu, определенное ранее
            if len(candidate_rings) < current_nrtu:
                logging.warning(f"[BN Worker P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{current_nrtu}.")
                current_nrtu = len(candidate_rings) # Обновление nrtu до числа кандидатов
            if current_nrtu == 0:
                logging.error(f"[BN Worker P:{pair_idx}] No candidates to select rings from.")
                batch_results[pair_idx] = []; continue # Возвращение пустого список бит

            # Выбор по энтропии
            entropies = []; min_pixels = 10
            L1_numpy_for_entropy = L1_tensor.cpu().numpy()
            for r_idx_cand in candidate_rings:
                entropy_val = -float('inf')
                if 0 <= r_idx_cand < len(coords) and isinstance(coords[r_idx_cand], torch.Tensor) and coords[r_idx_cand].shape[0] >= min_pixels:
                     c_tensor = coords[r_idx_cand]; rows_t, cols_t = c_tensor[:, 0], c_tensor[:, 1]
                     # Конвертируем индексы тензора в NumPy для индексации NumPy массива
                     rows_np, cols_np = rows_t.cpu().numpy(), cols_t.cpu().numpy()
                     try:
                         rv_np = L1_numpy_for_entropy[rows_np, cols_np]
                         shannon_entropy, _ = calculate_entropies(rv_np, fn, r_idx_cand)
                         if np.isfinite(shannon_entropy): entropy_val = shannon_entropy
                     except IndexError: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] IndexError entropy")
                     except Exception as e_entr: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] Entropy error: {e_entr}")
                entropies.append((entropy_val, r_idx_cand))

            entropies.sort(key=lambda x: x[0], reverse=True)
            # Выбираем не более current_nrtu колец
            selected_rings = [idx for e, idx in entropies if e > -float('inf')][:current_nrtu]

            # Fallback, если нужно
            if len(selected_rings) < current_nrtu:
                logging.warning(f"[BN Worker P:{pair_idx}] Fallback ring selection ({len(selected_rings)}<{current_nrtu}).")
                deterministic_fallback = candidate_rings[:current_nrtu]
                needed = current_nrtu - len(selected_rings)
                for ring in deterministic_fallback:
                    if needed == 0: break
                    if ring not in selected_rings:
                        selected_rings.append(ring)
                        needed -= 1
                # Если и после fallback не хватает
                if len(selected_rings) < current_nrtu:
                     logging.error(f"[BN Worker P:{pair_idx}] Fallback failed, not enough rings ({len(selected_rings)}<{current_nrtu}).")
                     # Установка nrtu равным числу фактически выбранных колец
                     current_nrtu = len(selected_rings)
                     extracted_bits_for_pair = [None] * current_nrtu

            # Логируем финально выбранные кольца
            logging.info(f"[BN Worker P:{pair_idx}] Selected {len(selected_rings)} rings for extraction: {selected_rings}")

            # --- Шаг 3: Извлечение бит из выбранных колец ---
            # размер списка бит соответствует числу выбранных колец
            extracted_bits_for_pair = [None] * len(selected_rings)
            for i, ring_idx_to_extract in enumerate(selected_rings):
                 # Передаем действительные L1_tensor, L2_tensor
                 extracted_bits_for_pair[i] = extract_single_bit(L1_tensor, L2_tensor, ring_idx_to_extract, nr, fn)

            # Если изначально ожидали nrtu бит, а извлекли меньше, дополняем None
            while len(extracted_bits_for_pair) < nrtu:
                 extracted_bits_for_pair.append(None)

            batch_results[pair_idx] = extracted_bits_for_pair[:nrtu] # Гарантируем нужный размер

        except cv2.error as cv_err:
             logging.error(f"OpenCV error P:{pair_idx} in BN worker: {cv_err}", exc_info=True)
             batch_results[pair_idx] = [None] * nrtu
        except RuntimeError as torch_err: # Ловим ошибки PyTorch отдельно
            logging.error(f"PyTorch runtime error P:{pair_idx} in BN worker: {torch_err}", exc_info=True)
            batch_results[pair_idx] = [None] * nrtu
        except Exception as e:
            logging.error(f"Unexpected error processing pair {pair_idx} in BN worker: {e}", exc_info=True)
            batch_results[pair_idx] = [None] * nrtu

    return batch_results


# --- Вспомогательная функция чтения кадров ---
def read_required_frames_opencv(video_path: str, num_frames_to_read: int) -> Optional[List[np.ndarray]]:
    """
    Читает ТОЛЬКО первые num_frames_to_read кадров с помощью OpenCV.
    """
    frames_opencv = []
    cap = None
    logging.info(f"[OpenCV Read Limited] Попытка открыть: '{video_path}' для чтения {num_frames_to_read} кадров")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"[OpenCV Read Limited] Не удалось открыть файл: {video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cv = cap.get(cv2.CAP_PROP_FPS)
        logging.debug(f"[OpenCV Read Limited] Видео: {width}x{height} @ {fps_cv:.2f} FPS")

        for i in range(num_frames_to_read):
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(
                    f"[OpenCV Read Limited] Не удалось прочитать кадр {i} или достигнут конец файла (запрошено {num_frames_to_read}, прочитано {len(frames_opencv)}).")
                break
            frames_opencv.append(frame)
            if (i + 1) % 200 == 0:  # Логируем реже для больших количеств
                logging.info(f"[OpenCV Read Limited] Прочитано кадров: {i + 1}/{num_frames_to_read}")

        logging.info(f"[OpenCV Read Limited] Чтение завершено. Получено кадров: {len(frames_opencv)}.")

        if len(frames_opencv) == 0 and num_frames_to_read > 0:  # Если ничего не прочитали, но должны были
            logging.error(f"[OpenCV Read Limited] Не удалось прочитать ни одного кадра из '{video_path}'.")
            return None

        return frames_opencv

    except Exception as e:
        logging.error(f"[OpenCV Read Limited] Ошибка при чтении файла '{video_path}': {e}", exc_info=True)
        return None
    finally:
        if cap:
            cap.release()
            logging.debug("[OpenCV Read Limited] VideoCapture освобожден.")


def generate_frame_pairs_opencv(video_path: str,
                                pairs_to_process: int,
                                # Параметры, которые просто передаются дальше в args
                                nr: int, nrtu: int, cps: int, ec: int,
                                device: Optional[torch.device],
                                dtcwt_fwd: Optional[DTCWTForward]
                                ) -> Iterator[Dict[str, Any]]:
    """
    Ленивый генератор, читающий видеофайл с помощью OpenCV (grab/retrieve)
    и выдающий словари с аргументами для обработки пар кадров.

    Args:
        video_path: Путь к видеофайлу.
        pairs_to_process: Максимальное количество пар для генерации.
        nr, nrtu, cps, ec, device, dtcwt_fwd: Параметры для _extract_batch_worker.

    Yields:
        Словарь с аргументами для _extract_batch_worker для каждой пары кадров.
    """
    cap = None
    frames_read_count = 0
    pairs_yielded_count = 0

    logging.info(f"[Генератор OpenCV] Инициализация для '{video_path}', макс. {pairs_to_process} пар.")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл OpenCV: {video_path}")

        frame_t: Optional[np.ndarray] = None
        frame_t_plus_1: Optional[np.ndarray] = None

        read_success = True

        # Предварительное чтение первого кадра (будущий frame_t для первой пары)
        if read_success:
            read_success = cap.grab()
            if read_success:
                ret_t, frame_t = cap.retrieve()
                read_success = ret_t and isinstance(frame_t, np.ndarray)
                if read_success:
                    frames_read_count += 1
                    logging.debug(f"[Генератор OpenCV] Успешно прочитан первый кадр (индекс 0).")
                else:
                    logging.warning("[Генератор OpenCV] Не удалось получить первый кадр после grab().")
            else:
                logging.warning("[Генератор OpenCV] Не удалось захватить первый кадр (grab failed).")

        # Основной цикл генерации пар
        # pair_idx здесь -- это индекс ПАРЫ, которую мы хотим сформировать и выдать
        for pair_idx in range(pairs_to_process):
            if not read_success:  # Если предыдущее чтение было неудачным
                logging.warning(f"[Генератор OpenCV] Предыдущее чтение не удалось, остановка на паре {pair_idx}.")
                break

            # Читаем следующий кадр (t+1)
            grab_success = cap.grab()
            if not grab_success:
                logging.warning(
                    f"[Генератор OpenCV] Не удалось захватить кадр {frames_read_count} для пары {pair_idx} (grab failed). Конец файла?")
                break

            retrieve_success, frame_t_plus_1 = cap.retrieve()
            if not retrieve_success or not isinstance(frame_t_plus_1, np.ndarray):
                logging.warning(
                    f"[Генератор OpenCV] Не удалось получить кадр {frames_read_count} для пары {pair_idx} после grab(). Конец файла или ошибка.")
                break

            frames_read_count += 1  # Успешно прочитали кадр t+1

            # У нас есть frame_t (из предыдущей итерации или первый) и frame_t_plus_1 (текущий)
            if frame_t is not None:  # frame_t должен быть не None после первой успешной итерации
                args = {
                    'pair_idx': pair_idx,  # Текущий индекс пары
                    'frame1': frame_t.copy(),
                    'frame2': frame_t_plus_1.copy(),
                    'n_rings': nr, 'num_rings_to_use': nrtu,
                    'candidate_pool_size': cps, 'embed_component': ec,
                    'device': device, 'dtcwt_fwd': dtcwt_fwd
                }
                yield args
                pairs_yielded_count += 1

                frame_t = frame_t_plus_1  # Готовимся к следующей итерации
                frame_t_plus_1 = None
            else:
                # Это может произойти только если первый кадр не удалось прочитать
                logging.error(
                    "[Генератор OpenCV] Ошибка логики: frame_t is None внутри основного цикла, пара не может быть сформирована.")
                break  # Прерываем генерацию, так как нет первого кадра для пары

        logging.info(
            f"Генератор OpenCV завершил работу. Выдано пар: {pairs_yielded_count}. Всего прочитано кадров: {frames_read_count}.")

    except IOError as e_io:
        logging.error(f"Ошибка открытия видеофайла в генераторе OpenCV: {e_io}", exc_info=False)
    except Exception as e_gen:
        logging.error(f"Неожиданная ошибка в генераторе OpenCV: {e_gen}", exc_info=True)
    finally:
        if cap and cap.isOpened():
            cap.release()
            logging.debug("Генератор OpenCV: VideoCapture освобожден.")
        final_yield_count = locals().get('pairs_yielded_count', 0)
        logging.debug(f"[Генератор OpenCV] Финальное количество выданных пар: {final_yield_count}")


# --- Основная функция извлечения (использует новый _extract_batch_worker) ---
# @profile
def extract_watermark_from_video(
        frames: List[np.ndarray],
        nr: int,  # N_RINGS
        nrtu: int,  # NUM_RINGS_TO_USE
        bp: int,  # BITS_PER_PAIR
        cps: int,  # CANDIDATE_POOL_SIZE
        ec: int,  # EMBED_COMPONENT
        expect_hybrid_ecc: bool,
        max_expected_packets: int,
        ue: bool,  # USE_ECC
        bch_code: Optional[BCH_TYPE],  # BCH_CODE_OBJECT
        device: Optional[torch.device],  # type: ignore
        dtcwt_fwd: Optional[DTCWTForward],
        plb: int,  # PAYLOAD_LEN_BYTES
        mw: Optional[int]  # MAX_WORKERS_EXTRACT
) -> List[str]:  # Возвращает список из 1 или 2 HEX-строк, или пустой список при ошибке
    """
    Основная функция извлечения ЦВЗ из предоставленного списка кадров.
    Использует ThreadPoolExecutor и новую стратегию голосования.
    """
    # Проверка доступности PyTorch компонентов (если они не проверены до вызова)
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:  # type: ignore
        logging.critical("Extract Watermark: Отсутствуют PyTorch Wavelets или Torch DCT!")
        return []

    if not frames:
        logging.error("Extract Watermark: Список кадров пуст! Нечего извлекать.")
        return []
    if device is None or dtcwt_fwd is None:
        logging.critical("Extract Watermark: Device или DTCWTForward не переданы!")
        return []

    if ue and expect_hybrid_ecc and not GALOIS_AVAILABLE:  # type: ignore
        logging.warning(
            "Extract Watermark: ECC требуется для гибридного режима, но Galois недоступен! Это повлияет на декодирование.")

    logging.info(f"--- Extract Watermark: Запуск Извлечения (список {len(frames)} кадров, Ultimate Voting) ---")
    logging.info(f"Параметры: Hybrid={expect_hybrid_ecc}, MaxPkts={max_expected_packets}, NRTU={nrtu}, BP={bp}")
    start_time = time.time()

    nf = len(frames)
    total_pairs_available = nf // 2
    if total_pairs_available == 0:
        logging.error("Extract Watermark: В предоставленном списке нет пар кадров для обработки.")
        return []

    payload_len_bits = plb * 8
    # Длина сообщения для ECC (информационная часть)
    message_len_for_ecc = payload_len_bits
    # Длина кодового слова, если ECC возможен и применяется
    actual_codeword_len_if_ecc = message_len_for_ecc
    # Длина сырого пакета (обычно равна длине сообщения)
    packet_len_if_raw = payload_len_bits
    ecc_possible_for_first = False

    if ue and GALOIS_AVAILABLE and bch_code is not None and isinstance(bch_code, BCH_TYPE):  # type: ignore
        try:
            if hasattr(bch_code, 'k') and hasattr(bch_code, 'n') and message_len_for_ecc <= bch_code.k:
                actual_codeword_len_if_ecc = bch_code.n  # Длина кодового слова BCH
                ecc_possible_for_first = True
                logging.info(
                    f"ECC проверка: Возможно для 1-го пакета (n={bch_code.n}, k={bch_code.k}, t={bch_code.t}).")
            else:
                k_val = bch_code.k if hasattr(bch_code, 'k') else 'N/A'
                logging.warning(
                    f"ECC проверка: Payload ({message_len_for_ecc}) > k ({k_val}). ECC не будет применен к первому пакету.")
        except Exception as e_galois_check:
            logging.error(f"ECC проверка: Ошибка параметров Galois: {e_galois_check}.")
    else:
        logging.info("ECC проверка: Либо USE_ECC=False, либо Galois недоступен/некорректен.")

    effective_expect_hybrid_ecc = expect_hybrid_ecc
    if effective_expect_hybrid_ecc and not ecc_possible_for_first:
        logging.warning(
            "Гибридный режим запрошен, но ECC для первого пакета невозможен/не настроен. Все пакеты будут обрабатываться как Raw (или как ECC, если ue=True и не гибридный).")
        effective_expect_hybrid_ecc = False  # Отключаем гибридный режим, если ECC для первого пакета невозможен

    max_possible_bits_to_extract = 0
    if effective_expect_hybrid_ecc:  # Первый пакет - ECC кодовое слово, остальные - RAW сообщения
        max_possible_bits_to_extract = actual_codeword_len_if_ecc + max(0, max_expected_packets - 1) * packet_len_if_raw
    else:  # Все пакеты или RAW, или ECC (если ue=True и ecc_possible_for_first=True, но не гибрид)
        # Если ue=True и ecc_possible_for_first, то каждый пакет должен быть длиной кодового слова
        len_of_each_packet = actual_codeword_len_if_ecc if ue and ecc_possible_for_first else packet_len_if_raw
        max_possible_bits_to_extract = max_expected_packets * len_of_each_packet

    if bp <= 0:
        logging.error(f"Bits per pair (bp) должен быть > 0, получено: {bp}")
        return []

    pairs_needed = ceil(max_possible_bits_to_extract / bp) if max_possible_bits_to_extract > 0 else 0
    pairs_to_process = min(total_pairs_available, pairs_needed)

    logging.info(f"Цель извлечения: до {max_expected_packets} пакетов (~{max_possible_bits_to_extract} бит).")
    logging.info(
        f"Пар кадров: Доступно={total_pairs_available}, Нужно={pairs_needed}, Будет обработано={pairs_to_process}")

    if pairs_to_process == 0:
        logging.warning("Нечего обрабатывать (pairs_to_process=0).")
        return []

    all_pairs_args: List[Dict[str, Any]] = []
    for pair_idx in range(pairs_to_process):
        i1, i2 = 2 * pair_idx, 2 * pair_idx + 1
        if i2 >= nf:
            logging.warning(
                f"Недостаточно кадров для формирования пары {pair_idx} (требуется {i2}, доступно {nf - 1}). Прерывание формирования задач.")
            break
        if frames[i1] is None or frames[i2] is None:
            logging.warning(f"Пропуск пары {pair_idx}: один из кадров None.")
            continue
        args = {'pair_idx': pair_idx, 'frame1': frames[i1].copy(), 'frame2': frames[i2].copy(),
                'n_rings': nr, 'num_rings_to_use': nrtu, 'candidate_pool_size': cps,
                'embed_component': ec, 'device': device, 'dtcwt_fwd': dtcwt_fwd}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if num_valid_tasks == 0:
        logging.error("Нет валидных задач для ThreadPoolExecutor после фильтрации пар.")
        return []
    if num_valid_tasks < pairs_to_process:
        logging.info(
            f"Количество валидных задач ({num_valid_tasks}) меньше, чем изначально планировалось ({pairs_to_process}).")
        pairs_to_process = num_valid_tasks  # Обновляем фактическое число обрабатываемых пар

    # Если после всех проверок num_valid_tasks (и, соответственно, pairs_to_process) стало 0
    if pairs_to_process == 0:
        logging.warning("Не осталось пар для обработки после всех проверок и фильтраций.")
        return []

    num_workers = mw if mw is not None and mw > 0 else (os.cpu_count() or 1)
    num_workers = min(num_workers, num_valid_tasks)
    batch_size_calc = num_valid_tasks / (num_workers * 2) if num_workers > 0 else num_valid_tasks
    batch_size = max(1, ceil(batch_size_calc))

    batched_args_list: List[List[Dict[str, Any]]] = []
    if all_pairs_args:  # Только если есть задачи
        batched_args_list = [all_pairs_args[i: i + batch_size] for i in range(0, num_valid_tasks, batch_size) if
                             all_pairs_args[i:i + batch_size]]

    extracted_bits_map: Dict[int, List[Optional[int]]] = {}
    if batched_args_list:
        logging.info(
            f"Запуск {len(batched_args_list)} батчей ({num_valid_tasks} пар) в ThreadPool (mw={num_workers}, batch_size≈{batch_size})...")
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_batch_info = {
                    executor.submit(_extract_batch_worker, batch_args): {  # type: ignore
                        'start_pair_idx': batch_args[0]['pair_idx'] if batch_args else -1,
                        'num_pairs_in_batch': len(batch_args)
                    } for batch_args in batched_args_list
                }
                for future in concurrent.futures.as_completed(future_to_batch_info):
                    batch_info = future_to_batch_info[future]
                    try:
                        batch_results_map = future.result()
                        if batch_results_map:
                            extracted_bits_map.update(batch_results_map)
                    except Exception as e_future:
                        logging.error(
                            f"Ошибка выполнения батча (начинающегося с пары ~{batch_info['start_pair_idx']}, {batch_info['num_pairs_in_batch']} пар): {e_future}",
                            exc_info=True)
        except Exception as e_executor:
            logging.critical(f"Критическая ошибка ThreadPoolExecutor: {e_executor}", exc_info=True)
            return []
    else:
        logging.warning("Нет батчей для обработки в ThreadPoolExecutor (возможно, num_valid_tasks=0).")

    if not extracted_bits_map and pairs_to_process > 0:
        logging.error("Ни одной пары не было успешно обработано (карта результатов извлеченных бит пуста).")
        return []

    extracted_bits_all: List[Optional[int]] = []
    for pair_idx_loop in range(pairs_to_process):  # Используем обновленное значение pairs_to_process
        bits_from_pair = extracted_bits_map.get(pair_idx_loop)
        if bits_from_pair and isinstance(bits_from_pair, list) and len(bits_from_pair) == bp:
            extracted_bits_all.extend(bits_from_pair)
        else:
            extracted_bits_all.extend([None] * bp)
            if bits_from_pair is None:
                logging.debug(f"Пара {pair_idx_loop}: нет результата в карте (None). Добавляем None * {bp}.")
            else:
                logging.warning(
                    f"Пара {pair_idx_loop}: неккоректный результат (len {len(bits_from_pair)} != {bp} or type {type(bits_from_pair)}). Добавляем None * {bp}.")

    valid_bits_for_decoding = [b for b in extracted_bits_all if b is not None and b in (0, 1)]
    logging.info(
        f"Сборка бит: Всего извлечено (с None): {len(extracted_bits_all)}, Валидных (0/1): {len(valid_bits_for_decoding)}.")
    if not valid_bits_for_decoding:
        logging.error("Нет валидных бит (0/1) для декодирования.")
        return []

    all_packets_details: List[Dict[str, Any]] = []
    num_processed_bits_for_packets = 0

    print("\n--- Попытки Декодирования Пакетов (для Ultimate Voting) ---")
    print(f"{'Pkt #':<6} | {'Type':<7} | {'Status':<28} | {'Corrected':<10} | {'Payload HEX':<18}")
    print("-" * (6 + 8 + 29 + 11 + 19))

    for i in range(max_expected_packets):
        is_first_packet = (i == 0)
        use_ecc_for_this_packet = is_first_packet and effective_expect_hybrid_ecc and ecc_possible_for_first and ue

        packet_type_str = "ECC" if use_ecc_for_this_packet else "RAW"
        len_bits_to_take_for_this_packet = actual_codeword_len_if_ecc if use_ecc_for_this_packet else packet_len_if_raw

        if num_processed_bits_for_packets + len_bits_to_take_for_this_packet > len(valid_bits_for_decoding):
            logging.warning(
                f"Недостаточно валидных бит ({len(valid_bits_for_decoding) - num_processed_bits_for_packets}) "
                f"для пакета {i + 1} (требуется {len_bits_to_take_for_this_packet}). Прерывание декодирования пакетов.")
            break

        current_packet_input_bits = valid_bits_for_decoding[
                                    num_processed_bits_for_packets: num_processed_bits_for_packets + len_bits_to_take_for_this_packet]
        num_processed_bits_for_packets += len_bits_to_take_for_this_packet

        packet_info: Dict[str, Any] = {'bits': None, 'hex': "N/A", 'type': packet_type_str, 'corrected_errors': 0}
        status_msg = "Processing error"

        if use_ecc_for_this_packet:
            # decode_ecc ожидает List[int] кодового слова, возвращает List[int] сообщения и кол-во ошибок
            decoded_payload_bits, corrected_count = decode_ecc(current_packet_input_bits, bch_code, plb)
            packet_info['corrected_errors'] = corrected_count  # -1 если ошибка, 0 если чистый, >0 если исправлен

            if decoded_payload_bits is not None and len(decoded_payload_bits) == payload_len_bits:
                packet_info['bits'] = decoded_payload_bits
                temp_bytes = bits_to_bytes_strict(decoded_payload_bits)
                if temp_bytes:
                    packet_info['hex'] = temp_bytes.hex()
                status_msg = f"OK (ECC, {corrected_count if corrected_count != -1 else 'N/A'} fixed)"
            else:
                if decoded_payload_bits is None:  # Ошибка декодирования ECC
                    status_msg = f"Fail (ECC Uncorrectable)" if corrected_count == -1 else f"Fail (ECC Decode Err, {corrected_count})"
                else:
                    status_msg = f"Fail (ECC Decoded Len Mismatch)"
                    logging.error(
                        f"Пакет {i + 1} (ECC): ошибка длины после декодирования {len(decoded_payload_bits)}!={payload_len_bits}")
        else:  # RAW packet
            if len(current_packet_input_bits) >= payload_len_bits:
                raw_payload_bits = current_packet_input_bits[:payload_len_bits]
                packet_info['bits'] = raw_payload_bits
                # Для RAW corrected_errors остается 0 (неприменимо)
                temp_bytes_raw = bits_to_bytes_strict(raw_payload_bits)
                if temp_bytes_raw:
                    packet_info['hex'] = temp_bytes_raw.hex()
                status_msg = "OK (RAW)"
            else:  # Не хватило бит даже для RAW сообщения
                status_msg = f"Fail (RAW too short: got {len(current_packet_input_bits)}, need {payload_len_bits})"

        all_packets_details.append(packet_info)
        corrected_str_display = str(packet_info['corrected_errors']) if packet_info['type'] == 'ECC' and packet_info[
            'corrected_errors'] != -1 else "-"
        print(
            f"{i + 1:<6} | {packet_info['type']:<7} | {status_msg:<28} | {corrected_str_display:<10} | {packet_info['hex']:<18}")

    print("-" * (6 + 8 + 29 + 11 + 19))

    valid_packets_for_voting = [
        p_info for p_info in all_packets_details
        if p_info['bits'] is not None and len(p_info['bits']) == payload_len_bits
    ]

    if not valid_packets_for_voting:
        logging.error("Нет валидных декодированных пакетов для финального голосования.")
        return []

    final_hex_candidates = ultimate_voting_strategy(
        valid_packets_info=valid_packets_for_voting,
        payload_len_bytes=plb
    )

    end_time = time.time()
    processing_duration = end_time - start_time
    logging.info(f"Извлечение ЦВЗ завершено. Общее время: {processing_duration:.2f} сек.")

    if not final_hex_candidates:
        logging.error("Ultimate voting не вернуло кандидатов.")
    elif len(final_hex_candidates) == 1:
        logging.info(f"Финальный извлеченный ID (1 кандидат): {final_hex_candidates[0]}")
    else:
        logging.warning(f"Извлечено два возможных кандидата ID: {final_hex_candidates[0]} и {final_hex_candidates[1]}")

    return final_hex_candidates

# --- Функция main ---
def main() -> int:
    main_start_time = time.time()
    logging.info(f"--- Запуск Основного Процесса Извлечения (с чтением XMP и Ultimate Voting) ---")

    # --- Проверки доступности библиотек ---
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        print("ERROR: PyTorch libraries required.")
        logging.critical("Критические PyTorch библиотеки не найдены.")
        return 1

    # --- Настройка current_expect_hybrid_ecc ---
    current_expect_hybrid_ecc = globals().get('expect_hybrid_ecc_global', True)
    if USE_ECC and current_expect_hybrid_ecc and not GALOIS_AVAILABLE:
        print("\nWARNING: ECC requested for hybrid mode but galois unavailable.")
        logging.warning("Galois недоступен, ECC для гибридного режима не будет работать (влияет на расчет числа пар).")

    # --- Настройка PyTorch device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device=device)  # Пробное создание тензора
            logging.info(f"Используется CUDA: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e_cuda_init:
            logging.error(f"Ошибка CUDA: {e_cuda_init}. Переключение на CPU.")
            device = torch.device("cpu")
    else:
        logging.info("Используется CPU.")

    # --- Инициализация DTCWTForward ---
    dtcwt_fwd: Optional[DTCWTForward] = None
    if PYTORCH_WAVELETS_AVAILABLE:
        try:
            dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)  # type: ignore
            logging.info("PyTorch DTCWTForward instance created.")
        except Exception as e_dtcwt:
            logging.critical(f"Failed to init DTCWTForward: {e_dtcwt}", exc_info=True)
            print(f"ОШИБКА: Failed to init DTCWTForward: {e_dtcwt}")
            return 1
    else:  # Сюда не должны попасть, если проверка выше отработала
        logging.critical("PyTorch Wavelets недоступен! (Ошибка в логике проверки)")
        print("ОШИБКА: PyTorch Wavelets недоступен!")
        return 1

    # --- Инициализация BCH_CODE_OBJECT (если еще не сделано глобально) ---
    # Обычно это делается один раз при старте модуля, но для полноты:
    global BCH_CODE_OBJECT  # Если BCH_CODE_OBJECT - глобальная переменная
    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is None:
        # Ваша логика инициализации BCH_CODE_OBJECT, как в начале файла extractor.py
        try:
            # Глобальные BCH_M, BCH_T должны быть определены
            _test_m = globals().get('BCH_M', 8)
            _test_t = globals().get('BCH_T', 9)
            _test_n = (1 << _test_m) - 1
            _test_d = 2 * _test_t + 1
            BCH_CODE_OBJECT = galois.BCH(_test_n, d=_test_d)  # type: ignore
            logging.info(
                f"BCH_CODE_OBJECT инициализирован в main: n={BCH_CODE_OBJECT.n}, k={BCH_CODE_OBJECT.k}, t={BCH_CODE_OBJECT.t}")  # type: ignore
        except Exception as e_bch_main_init:
            logging.error(f"Ошибка инициализации BCH_CODE_OBJECT в main: {e_bch_main_init}")

    # --- Определение имени входного файла ---
    input_extension_val = globals().get('INPUT_EXTENSION', ".mp4")
    bch_t_from_global = globals().get('BCH_T', "X")
    input_base_name = f"watermarked_ffmpeg_t9"
    input_video = input_base_name + input_extension_val

    logging.info(f"--- Начало извлечения из файла: '{input_video}' ---")
    if not os.path.exists(input_video):
        logging.critical(f"Входной файл не найден: '{input_video}'.")
        print(f"ОШИБКА: Файл не найден: '{input_video}'")
        return 1

    # --- Чтение эталонного хеша из XMP ---
    original_id_hash_from_xmp: Optional[str] = None
    exiftool_direct_path = r"C:\exiftool-13.29_64\exiftool.exe"
    exiftool_path_to_use = None
    if os.path.isfile(exiftool_direct_path):
        exiftool_path_to_use = exiftool_direct_path
    else:
        exiftool_path_to_use = shutil.which("exiftool.exe") or shutil.which("exiftool")

    if exiftool_path_to_use:
        # tag_to_read_for_exiftool = "XMP-xmp:MediaDataHash" # Используйте ваш актуальный тег
        tag_to_read_for_exiftool = globals().get('XMP_TAG_NAME_FOR_HASH', "XMP-xmp:TrackMetaHash")  # Пример

        cmd_exiftool_read = [exiftool_path_to_use, "-s3", f"-{tag_to_read_for_exiftool}", input_video]
        logging.info(f"Чтение хеша ID из XMP: {' '.join(cmd_exiftool_read)}")
        try:
            result_exiftool_read = subprocess.run(cmd_exiftool_read, check=False, capture_output=True, text=True,
                                                  encoding='utf-8', errors='replace')
            if result_exiftool_read.returncode == 0:
                original_id_hash_from_xmp = result_exiftool_read.stdout.strip()
                if original_id_hash_from_xmp:
                    if len(original_id_hash_from_xmp) == 64:  # SHA256
                        logging.info(f"Эталонный хеш ID из XMP: {original_id_hash_from_xmp}")
                        print(f"  Эталонный хеш из XMP: {original_id_hash_from_xmp}")
                    else:
                        logging.warning(
                            f"XMP хеш имеет неверную длину: '{original_id_hash_from_xmp}' ({len(original_id_hash_from_xmp)}).")
                        original_id_hash_from_xmp = None
                else:
                    logging.info(f"Тег XMP '{tag_to_read_for_exiftool}' не найден или пуст в '{input_video}'.")
            else:  # returncode != 0
                logging.warning(
                    f"ExifTool не смог прочитать тег '{tag_to_read_for_exiftool}' (код {result_exiftool_read.returncode}). Stderr: {result_exiftool_read.stderr.strip()}")
        except FileNotFoundError:
            logging.error(f"ExifTool не найден по пути: '{exiftool_path_to_use}'.")
        except Exception as e_exif_read_general:
            logging.error(f"Общая ошибка при чтении XMP с ExifTool: {e_exif_read_general}", exc_info=True)
    else:
        logging.warning("ExifTool не найден. Невозможно прочитать эталонный хеш из XMP.")
        print("ПРЕДУПРЕЖДЕНИЕ: ExifTool не найден.")

    # --- Расчет необходимого числа кадров для чтения ---
    # Эта логика идентична той, что в extract_watermark_from_video, используется для определения num_frames_to_read
    # Для краткости, предположим, что PAYLOAD_LEN_BYTES, USE_ECC, MAX_TOTAL_PACKETS_global, BITS_PER_PAIR определены глобально
    payload_len_bits_calc = PAYLOAD_LEN_BYTES * 8
    packet_len_if_ecc_calc = payload_len_bits_calc
    packet_len_if_raw_calc = payload_len_bits_calc
    ecc_possible_for_first_calc = False
    actual_codeword_len_if_ecc_calc = payload_len_bits_calc

    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is not None and isinstance(BCH_CODE_OBJECT, BCH_TYPE):
        try:
            if hasattr(BCH_CODE_OBJECT, 'k') and hasattr(BCH_CODE_OBJECT,
                                                         'n') and payload_len_bits_calc <= BCH_CODE_OBJECT.k:  # type: ignore
                actual_codeword_len_if_ecc_calc = BCH_CODE_OBJECT.n  # type: ignore
                ecc_possible_for_first_calc = True
        except Exception:
            pass  # Ошибки уже логировались при инициализации BCH_CODE_OBJECT

    expect_hybrid_for_calc = current_expect_hybrid_ecc
    if expect_hybrid_for_calc and not ecc_possible_for_first_calc:
        expect_hybrid_for_calc = False

    max_possible_bits_to_extract_calc = 0
    if expect_hybrid_for_calc:
        max_possible_bits_to_extract_calc = actual_codeword_len_if_ecc_calc + max(0,
                                                                                  MAX_TOTAL_PACKETS_global - 1) * packet_len_if_raw_calc
    else:
        len_for_each_pkt_calc = actual_codeword_len_if_ecc_calc if USE_ECC and ecc_possible_for_first_calc and not expect_hybrid_for_calc else packet_len_if_raw_calc
        max_possible_bits_to_extract_calc = MAX_TOTAL_PACKETS_global * len_for_each_pkt_calc

    if BITS_PER_PAIR <= 0: logging.critical(f"BITS_PER_PAIR ({BITS_PER_PAIR}) <= 0!"); return 1
    pairs_needed_for_extract = ceil(
        max_possible_bits_to_extract_calc / BITS_PER_PAIR) if max_possible_bits_to_extract_calc > 0 else 0

    if pairs_needed_for_extract == 0:
        logging.error("Не требуется обрабатывать ни одной пары (согласно расчетам).")
        return 1
    num_frames_to_read = pairs_needed_for_extract * 2
    logging.info(
        f"Требуется обработать {pairs_needed_for_extract} пар, необходимо прочитать {num_frames_to_read} кадров.")

    # --- Чтение кадров ---
    # Предполагается, что функция read_required_frames_opencv доступна
    if not CV2_AVAILABLE:  # Проверка, если cv2 не импортирован
        logging.critical("OpenCV (cv2) недоступен, не удается прочитать кадры.")
        return 1

    read_start_time = time.time()
    frames_for_extraction = read_required_frames_opencv(input_video, num_frames_to_read)  # type: ignore
    read_time = time.time() - read_start_time

    if frames_for_extraction is None:
        logging.critical("Критическая ошибка при чтении кадров. Прерывание.")
        return 1
    logging.info(f"Прочитано {len(frames_for_extraction)} кадров для извлечения за {read_time:.2f} сек.")
    if len(frames_for_extraction) < 2:
        logging.error(f"Прочитано менее 2 кадров ({len(frames_for_extraction)}). Невозможно извлечь ЦВЗ.")
        return 1

    # --- Вызов основной функции извлечения ---
    # Передаем глобальные константы явно или убеждаемся, что они доступны в extract_watermark_from_video
    final_hex_candidates = extract_watermark_from_video(
        frames=frames_for_extraction,
        nr=N_RINGS,
        nrtu=NUM_RINGS_TO_USE,
        bp=BITS_PER_PAIR,
        cps=CANDIDATE_POOL_SIZE,
        ec=EMBED_COMPONENT,
        expect_hybrid_ecc=current_expect_hybrid_ecc,
        max_expected_packets=MAX_TOTAL_PACKETS_global,
        ue=USE_ECC,
        bch_code=BCH_CODE_OBJECT,
        device=device,
        dtcwt_fwd=dtcwt_fwd,
        plb=PAYLOAD_LEN_BYTES,
        mw=MAX_WORKERS_EXTRACT
    )
    if frames_for_extraction:  # Очистка памяти
        del frames_for_extraction
        gc.collect()

    # --- Вывод и сравнение результатов ---
    print(f"\n--- Результаты Извлечения (main) ---")
    final_match_status = False

    if not final_hex_candidates:
        print(f"  Извлечение НЕ УДАЛОСЬ (нет кандидатов).")
        logging.error("Извлечение не удалось/нет кандидатов от extract_watermark_from_video.")
        if original_id_hash_from_xmp:
            print(f"  (Эталонный хеш из XMP был: {original_id_hash_from_xmp})")
    else:
        print(f"  Получено кандидатов: {len(final_hex_candidates)}")
        for idx, extracted_hex in enumerate(final_hex_candidates):
            print(f"  Кандидат {idx + 1} (Hex): {extracted_hex}")
            calculated_hash = hashlib.sha256(bytes.fromhex(extracted_hex)).hexdigest()
            print(f"    Хеш кандидата {idx + 1} : {calculated_hash}")
            if original_id_hash_from_xmp:
                if calculated_hash == original_id_hash_from_xmp:
                    print(f"    >>> ХЕШ КАНДИДАТА {idx + 1} СОВПАЛ С XMP (ID MATCH) <<<")
                    logging.info(
                        f"ХЕШ КАНДИДАТА {idx + 1} ({extracted_hex}) СОВПАЛ С XMP ({original_id_hash_from_xmp}).")
                    final_match_status = True  # Считаем успехом, если хотя бы один кандидат совпал
                else:
                    print(f"    !!! ХЕШ КАНДИДАТА {idx + 1} НЕ СОВПАЛ С XMP !!!")
                    logging.warning(
                        f"ХЕШ КАНДИДАТА {idx + 1} ({extracted_hex}) НЕ СОВПАЛ С XMP ({original_id_hash_from_xmp}).")
            else:
                print(f"    Эталонный хеш из XMP недоступен для сравнения с кандидатом {idx + 1}.")

        if not original_id_hash_from_xmp and final_hex_candidates:
            logging.info(f"Хеш XMP не найден, но извлечены кандидаты: {', '.join(final_hex_candidates)}")

    # Загрузка ID из txt файла для дополнительной информации (если есть)
    original_id_from_txt: Optional[str] = None
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r", encoding='utf-8') as f_id:
                original_id_from_txt = f_id.read().strip()
            if original_id_from_txt:
                print(f"  Оригинальный ID из '{ORIGINAL_WATERMARK_FILE}': {original_id_from_txt}")
        except Exception as e_read_txt_id:
            logging.warning(f"Не удалось прочитать ID из TXT файла: {e_read_txt_id}")

    if final_match_status:
        print("\n  >>>ID MATCH<<<")
    else:
        print("\n  Отсутствие XMP либо неудача.")

    logging.info("--- Основной Процесс Извлечения Завершен ---")
    total_main_time = time.time() - main_start_time
    logging.info(f"--- Общее Время Работы Экстрактора: {total_main_time:.2f} сек ---")

    log_filename_val = LOG_FILENAME
    print(f"\nИзвлечение завершено. Лог: {log_filename_val}")

    return 0 if final_match_status else 1


# --- Блок if __name__ == "__main__": ---
if __name__ == "__main__":
    # --- Настройка логирования ---
    # Убедимся, что LOG_FILENAME определена глобально перед этим блоком
    # LOG_FILENAME = "watermarking_extract_ultimate.log" # Пример, если не определена выше
    if not logging.getLogger().handlers:  # Проверяем, есть ли уже настроенные хендлеры
        # Если нет, настраиваем. Если есть, предполагаем, что они настроены корректно выше.
        # Это предотвращает дублирование сообщений, если скрипт импортируется или запускается несколько раз с настройкой.
        logging.basicConfig(filename=LOG_FILENAME,
                            filemode='w',  # 'a' для добавления, 'w' для перезаписи
                            level=logging.INFO,  # Уровень по умолчанию
                            format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    # Устанавливаем уровень для корневого логгера (можно сделать более гранулярно для конкретных модулей)
    logging.getLogger().setLevel(logging.DEBUG)  # Для отладки можно поставить DEBUG
    # logging.getLogger().setLevel(logging.INFO) # Для обычного использования

    # --- Проверка зависимостей ---
    # (Ваш код проверки зависимостей)
    missing_libs_critical = []
    if not CV2_AVAILABLE: missing_libs_critical.append("OpenCV (cv2)")
    if not PYTORCH_WAVELETS_AVAILABLE: missing_libs_critical.append("pytorch_wavelets")
    if not TORCH_DCT_AVAILABLE: missing_libs_critical.append("torch-dct")
    try:
        import numpy  # Проверка numpy
    except ImportError:
        missing_libs_critical.append("NumPy")
    try:
        import torch  # Проверка torch
    except ImportError:
        missing_libs_critical.append("PyTorch")

    # Проверка shutil и subprocess (встроенные, но для полноты)
    try:
        import shutil
    except ImportError:
        missing_libs_critical.append("shutil (стандартная библиотека)")
    try:
        import subprocess
    except ImportError:
        missing_libs_critical.append("subprocess (стандартная библиотека)")

    if missing_libs_critical:
        error_msg = f"ОШИБКА: Отсутствуют КРИТИЧЕСКИ важные библиотеки: {', '.join(missing_libs_critical)}."
        print(error_msg);
        logging.critical(error_msg);
        sys.exit(1)  # type: ignore

    if globals().get('USE_ECC', False) and not globals().get('GALOIS_AVAILABLE', False):
        warning_msg = "\nПРЕДУПРЕЖДЕНИЕ: USE_ECC=True, но библиотека 'galois' не найдена/не работает. ECC будет недоступен."
        print(warning_msg);
        logging.warning(warning_msg.strip())

    # --- Инициализация BCH_CODE_OBJECT (если используется ECC) ---
    # Этот блок должен быть здесь, если BCH_CODE_OBJECT не инициализируется в main() или глобально выше
    # и если GALOIS_AVAILABLE.
    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is None:
        try:
            # Глобальные BCH_M, BCH_T должны быть определены
            _m = globals().get('BCH_M', 8)  # Дефолтные значения, если не найдены
            _t = globals().get('BCH_T', 9)
            _n = (1 << _m) - 1
            _d = 2 * _t + 1

            # Определение k на основе t (ваша логика)
            expected_k = -1
            if _t == 5:
                expected_k = 215
            elif _t == 7:
                expected_k = 201
            elif _t == 9:
                expected_k = 187
            elif _t == 11:
                expected_k = 173
            elif _t == 15:
                expected_k = 131

            if expected_k == -1:
                logging.error(f"Неизвестное ожидаемое k для t={_t}. BCH объект не будет создан.")
            else:
                BCH_CODE_OBJECT = galois.BCH(_n, d=_d)  # type: ignore
                if BCH_CODE_OBJECT.k == expected_k and BCH_CODE_OBJECT.t == _t:  # type: ignore
                    logging.info(
                        f"BCH код инициализирован: n={BCH_CODE_OBJECT.n}, k={BCH_CODE_OBJECT.k}, t={BCH_CODE_OBJECT.t}")  # type: ignore
                else:
                    logging.error(
                        f"Ошибка инициализации BCH: ожидалось k={expected_k}, t={_t}, получено k={BCH_CODE_OBJECT.k}, t={BCH_CODE_OBJECT.t}. Сбрасываю BCH_CODE_OBJECT.")  # type: ignore
                    BCH_CODE_OBJECT = None
        except Exception as e_bch_init_main:
            logging.error(f"Ошибка при инициализации BCH_CODE_OBJECT в __main__: {e_bch_init_main}", exc_info=True)
            BCH_CODE_OBJECT = None  # Сбрасываем при ошибке

    # --- Профилирование ---
    DO_PROFILING = False  # Установите в True для профилирования
    profiler_instance = None
    if DO_PROFILING:
        try:
            import cProfile
            import pstats

            if 'KERNPROF_VAR' not in os.environ and 'profile' not in globals():
                profiler_instance = cProfile.Profile()
                profiler_instance.enable()
                print("cProfile профилирование включено.")
                logging.info("cProfile профилирование включено.")
            elif 'profile' in globals() and callable(globals()['profile']):  # type: ignore
                print("line_profiler активен (через декоратор @profile). cProfile не будет запущен.")
                logging.info("line_profiler активен. cProfile не запущен.")
        except ImportError:
            logging.warning("Модули cProfile или pstats не найдены. Профилирование отключено.")
            DO_PROFILING = False

    # --- Запуск main ---
    final_exit_code = 1  # По умолчанию - ошибка
    try:
        final_exit_code = main()
    except FileNotFoundError as e_fnf_main_exc:
        print(f"\nОШИБКА: Файл не найден во время выполнения main(): {e_fnf_main_exc}")
        logging.critical(f"FileNotFoundError в __main__ -> main(): {e_fnf_main_exc}", exc_info=True)
    except torch.cuda.OutOfMemoryError as e_oom_main_exc:  # type: ignore
        print(f"\nОШИБКА: Недостаточно памяти CUDA: {e_oom_main_exc}")
        logging.critical(f"torch.cuda.OutOfMemoryError в __main__ -> main(): {e_oom_main_exc}", exc_info=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()  # type: ignore
    except Exception as e_global_main_exc:
        print(f"\nКРИТИЧЕСКАЯ НЕОБРАБОТАННАЯ ОШИБКА в __main__ -> main(): {e_global_main_exc}")
        logging.critical(f"Необработанная ошибка в __main__ -> main(): {e_global_main_exc}", exc_info=True)
    finally:
        if DO_PROFILING and profiler_instance is not None:
            profiler_instance.disable()
            logging.info("cProfile профилирование выключено.")
            try:
                stats_obj = pstats.Stats(profiler_instance).strip_dirs().sort_stats("cumulative")
                print("\n--- Статистика Профилирования (cProfile, Top 30) ---");
                stats_obj.print_stats(30)
                # Глобальная BCH_T или дефолт "X"
                bch_t_for_profile_name = globals().get('BCH_T', "X")
                profile_prof_file = f"profile_extract_main_t{bch_t_for_profile_name}.prof"
                profile_txt_file = f"profile_extract_main_t{bch_t_for_profile_name}.txt"
                stats_obj.dump_stats(profile_prof_file)
                with open(profile_txt_file, 'w', encoding='utf-8') as f_pstats:
                    # Перенаправляем вывод pstats в файл
                    ps = pstats.Stats(profiler_instance, stream=f_pstats).strip_dirs().sort_stats('cumulative')
                    ps.print_stats()
                print(f"Статистика профилирования сохранена: {profile_prof_file}, {profile_txt_file}")
                logging.info(f"Статистика профилирования сохранена: {profile_prof_file}, {profile_txt_file}")
            except Exception as e_pstats_save_exc:
                logging.error(f"Ошибка сохранения статистики профилирования: {e_pstats_save_exc}")

        logging.info(f"Скрипт watermark_extractor.py завершен с кодом выхода {final_exit_code}.")
        print(f"\nСкрипт завершен с кодом выхода {final_exit_code}.")
        # sys.exit(final_exit_code) # Закомментировано, чтобы не прерывать выполнение в интерактивных средах
