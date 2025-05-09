from typing import Dict, List
from math import ceil

import pytest
import time
import os
import numpy as np
import logging
import cv2
from concurrent.futures import ProcessPoolExecutor, wait, as_completed

try:
    from watermark_embedder import (
        _embed_frame_pair_worker,
        embed_frame_pair,
        dtcwt_transform,
        ring_division,
        calculate_entropies,
        get_fixed_pseudo_random_rings,
        compute_adaptive_alpha_entropy,
        calculate_perceptual_mask,
        dct_1d, idct_1d, svd,
        dtcwt_inverse,
        N_RINGS,
        NUM_RINGS_TO_USE,
        CANDIDATE_POOL_SIZE,
        EMBED_COMPONENT,
        USE_PERCEPTUAL_MASKING,
        PAYLOAD_LEN_BYTES,
        BITS_PER_PAIR,
        ALPHA_MIN, ALPHA_MAX, LAMBDA_PARAM,
        LOG_FILENAME # Можно использовать для настройки логгера теста
    )
    EMBEDDER_IMPORTED = True
    print("Модуль watermark_embedder успешно импортирован.")
except ImportError as e:
    print(f"Не удалось импортировать watermark_embedder: {e}. Пропуск тестов, требующих его.")
    EMBEDDER_IMPORTED = False
    # Определяем заглушки, чтобы код ниже не падал с NameError
    _embed_frame_pair_worker = None
    dtcwt_transform = None
    ring_division = None
    calculate_perceptual_mask = None
    get_fixed_pseudo_random_rings = None
    EMBED_COMPONENT = 2
    NUM_RINGS_TO_USE = 2
    N_RINGS = 8
    CANDIDATE_POOL_SIZE = 4
    USE_PERCEPTUAL_MASKING = True

@pytest.fixture(scope="module")
def worker_test_data_original():
    """Готовит данные для ОРИГИНАЛЬНОГО воркера _embed_frame_pair_worker."""
    if not EMBEDDER_IMPORTED:
        pytest.skip("Модуль embedder не импортирован, пропуск оригинального теста.")

    # Используем количество ядер CPU как основу для числа задач
    num_tasks = (os.cpu_count() or 4) * 2 # Задач больше чем ядер, чтобы увидеть эффект
    print(f"\nПодготовка данных для {num_tasks} ОРИГИНАЛЬНЫХ тестовых задач...")
    height, width = 120, 160 # Маленький размер для ускорения
    dummy_frame1 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    dummy_frame2 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    dummy_bits = [i % 2 for i in range(NUM_RINGS_TO_USE)]

    args_list = []
    for i in range(num_tasks):
        args = {
            'pair_idx': i,
            'frame1': dummy_frame1.copy(),
            'frame2': dummy_frame2.copy(),
            'bits': list(dummy_bits),
            'n_rings': N_RINGS,
            'num_rings_to_use': NUM_RINGS_TO_USE,
            'candidate_pool_size': CANDIDATE_POOL_SIZE,
            'frame_number': 2 * i,
            'use_perceptual_masking': USE_PERCEPTUAL_MASKING,
            'embed_component': EMBED_COMPONENT
        }
        args_list.append(args)
    print(f"Оригинальные данные подготовлены ({num_tasks} задач).")
    return args_list

@pytest.fixture(scope="module")
def worker_test_data_simplified():
    """Готовит данные для УПРОЩЕННОГО воркера."""
    if not EMBEDDER_IMPORTED:
        pytest.skip("Модуль embedder не импортирован, пропуск упрощенного теста.")

    num_tasks = (os.cpu_count() or 4) * 2
    print(f"\nПодготовка УПРОЩЕННЫХ данных для {num_tasks} тестовых задач...")
    height, width = 120, 160
    dummy_frame1_bgr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    dummy_frame2_bgr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    dummy_bits = [i % 2 for i in range(NUM_RINGS_TO_USE)]

    try:
        f1_ycrcb = cv2.cvtColor(dummy_frame1_bgr, cv2.COLOR_BGR2YCrCb)
        f2_ycrcb = cv2.cvtColor(dummy_frame2_bgr, cv2.COLOR_BGR2YCrCb)
        comp1 = f1_ycrcb[:, :, EMBED_COMPONENT].astype(np.float32) / 255.0
        comp2 = f2_ycrcb[:, :, EMBED_COMPONENT].astype(np.float32) / 255.0
    except Exception as e:
        pytest.fail(f"Ошибка подготовки компонентов: {e}")

    dummy_selected_rings = get_fixed_pseudo_random_rings(0, N_RINGS, NUM_RINGS_TO_USE)

    args_list = []
    for i in range(num_tasks):
        args = {
            'pair_idx': i,
            'comp1': comp1.copy(),
            'comp2': comp2.copy(),
            'bits': list(dummy_bits),
            'selected_rings': list(dummy_selected_rings),
            'n_rings': N_RINGS,
            'use_perceptual_masking': USE_PERCEPTUAL_MASKING,
            'embed_component': EMBED_COMPONENT
        }
        args_list.append(args)
    print(f"Упрощенные данные подготовлены ({num_tasks} задач).")
    return args_list

@pytest.fixture(scope="module")
def worker_test_data_minimal():
    """Готовит минимальные данные."""
    num_tasks = (os.cpu_count() or 4) * 2
    print(f"\nПодготовка МИНИМАЛЬНЫХ данных для {num_tasks} тестовых задач...")
    args_list = [{'pair_idx': i} for i in range(num_tasks)]
    print(f"Минимальные данные подготовлены ({num_tasks} задач).")
    return args_list

@pytest.fixture(scope="module")
def worker_test_data_batched(worker_test_data_original):
    """Готовит список батчей аргументов."""
    if not EMBEDDER_IMPORTED:
        pytest.skip("Модуль embedder не импортирован, пропуск теста батчей.")

    original_args_list = worker_test_data_original
    num_total_tasks = len(original_args_list)

    batch_size = max(1, (os.cpu_count() or 4) // 2)
    num_batches = ceil(num_total_tasks / batch_size)

    print(f"\nПодготовка {num_batches} батчей по ~{batch_size} задач в каждом (всего {num_total_tasks} задач)...")

    batched_args_list = []
    for i in range(0, num_total_tasks, batch_size):
        batch = original_args_list[i : i + batch_size]
        if batch:
            batched_args_list.append(batch)

    print(f"Данные разделены на {len(batched_args_list)} батчей.")
    return batched_args_list


def set_thread_limits():
    """Устанавливает переменные окружения для ограничения потоков библиотек."""
    print("Установка переменных окружения: *_NUM_THREADS=1")
    limits = {'1': ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                   'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']}
    for limit, keys in limits.items():
        for key in keys:
            os.environ[key] = limit

def run_and_measure(worker_func, args_list, num_workers, run_parallel=True, test_name=""):
    """Запускает задачи (последовательно или параллельно) и измеряет время."""
    num_tasks = len(args_list)
    results = []
    mode = "ПАРАЛЛЕЛЬНО" if run_parallel else "ПОСЛЕДОВАТЕЛЬНО"
    worker_count_str = f"{num_workers} воркер{'ов' if num_workers != 1 else ''}" if run_parallel else "1 воркер"
    print(f"\n--- Запуск {test_name} ({num_tasks} задач) {mode} ({worker_count_str}) ---")

    start_time = time.perf_counter()
    try:
        if run_parallel:
            futures = []
            actual_workers = max(1, num_workers)
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                for args in args_list:
                    futures.append(executor.submit(worker_func, args))
                temp_results = {} # Словарь для сбора результатов по индексу/ключу
                for i, future in enumerate(as_completed(futures)):
                    try:
                         res = future.result()
                         if isinstance(args_list[0], dict) and 'pair_idx' in args_list[0]:
                              key = args_list[i]['pair_idx']
                         elif isinstance(args_list[0], list):
                              key = i
                         else:
                              key = i
                         temp_results[key] = res
                    except Exception as e:
                         print(f"  !!! Ошибка при получении результата задачи {i}: {e}")
                         key = i # Используем индекс как ключ для ошибки
                         temp_results[key] = f"ERROR: {e}"


            # Это больше для проверки, чем для измерения времени
            if all(isinstance(k, int) for k in temp_results.keys()):
                 results = [temp_results.get(i) for i in range(num_tasks)]
            else:
                 results = list(temp_results.values())


            if len(results) != num_tasks:
                 print(f"ПРЕДУПРЕЖДЕНИЕ: Получено {len(results)} результатов, ожидалось {num_tasks}")

        else:
            for args in args_list:
                results.append(worker_func(args))
    except Exception as e:
        pytest.fail(f"Критическая ошибка во время выполнения {mode} {test_name}: {e}", pytrace=True)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"-> Общее время выполнения {mode}: {elapsed_time:.4f} секунд.")
    return elapsed_time, results


def _test_worker_simplified(args_simplified: Dict):
    """Воркер для теста, принимает компоненты, возвращает статус."""
    pair_idx = args_simplified['pair_idx']
    frame_number = 2 * pair_idx
    comp1 = args_simplified['comp1']
    comp2 = args_simplified['comp2']
    selected_rings_indices = args_simplified['selected_rings']
    n_rings_arg = args_simplified['n_rings']
    use_perceptual_masking_arg = args_simplified['use_perceptual_masking']

    try:
        pyramid1 = dtcwt_transform(comp1, frame_number)
        if use_perceptual_masking_arg:
            calculate_perceptual_mask(comp1, frame_number)
        if pyramid1 is None or pyramid1.lowpass is None: return pair_idx, False

        time.sleep(0.005)
        return pair_idx, True
    except Exception:
        return pair_idx, False

def _test_worker_minimal(args_minimal: Dict):
    """Воркер, который почти ничего не делает."""
    pair_idx = args_minimal['pair_idx']
    time.sleep(0.005)
    return pair_idx, True

def _test_worker_batch(batch_args_list: List[Dict]):
    """
    Обрабатывает батч задач последовательно внутри одного процесса.
    Использует ОРИГИНАЛЬНЫЙ воркер _embed_frame_pair_worker.
    """
    if not EMBEDDER_IMPORTED or _embed_frame_pair_worker is None:
        return []

    batch_results = []
    for args in batch_args_list:
        try:
            result = _embed_frame_pair_worker(args)
            batch_results.append(result)
        except Exception as e:
            pair_idx = args.get('pair_idx', -1)
            print(f"!!! Ошибка в батч-воркере (pair_idx={pair_idx}): {e}")
            batch_results.append(None) # Маркер ошибки
    return batch_results


# --- Основная функция теста масштабируемости ---
def run_scaling_test(worker_func, test_data, test_name, is_batch_test=False):
    """Проводит тест масштабируемости для заданной функции воркера и данных."""
    set_thread_limits()

    num_tasks_or_batches = len(test_data)
    if is_batch_test:
         num_total_tasks = sum(len(batch) for batch in test_data)
         num_workers = min(num_tasks_or_batches, os.cpu_count() or 4)
         print(f"Тест батчей: {num_tasks_or_batches} батчей, {num_total_tasks} всего задач.")
    else:
         num_total_tasks = num_tasks_or_batches
         num_workers = min(num_total_tasks, os.cpu_count() or 4)

    sequential_time, _ = run_and_measure(worker_func, test_data, 1, run_parallel=False, test_name=test_name)

    concurrent_time, _ = run_and_measure(worker_func, test_data, num_workers, run_parallel=True, test_name=test_name)

    print(f"\n--- Оценка Масштабирования ({test_name}) ---")
    if num_workers <= 0:
        print("Невозможно оценить масштабирование с 0 воркерами.")
        return

    seq_time_per_task = sequential_time / num_total_tasks if num_total_tasks > 0 else 0
    conc_time_per_task = concurrent_time / num_total_tasks if num_total_tasks > 0 else 0

    ideal_concurrent_time = sequential_time / num_workers if num_workers > 0 else sequential_time
    overhead_tolerance_factor = 5.0 if test_name == "Minimal" else 3.0 # Увеличим допуск
    max_allowed_concurrent_time = ideal_concurrent_time * overhead_tolerance_factor

    print(f"   Последовательное время: {sequential_time:.4f}с ({seq_time_per_task*1000:.2f} мс/задачу)")
    print(f"   Параллельное время ({num_workers} воркеров): {concurrent_time:.4f}с ({conc_time_per_task*1000:.2f} мс/задачу)")
    print(f"   Идеальное параллельное время: {ideal_concurrent_time:.4f}с")
    print(f"   Макс. допустимое (Ideal * {overhead_tolerance_factor:.1f}): {max_allowed_concurrent_time:.4f}с")

    scaling_ratio = sequential_time / concurrent_time if concurrent_time > 0 else float('inf')
    print(f"   Ускорение (Seq / Conc): {scaling_ratio:.2f}x (Идеал: {num_workers:.2f}x)")

    assert concurrent_time > 0, "Параллельное время не может быть нулевым или отрицательным."

    if num_workers > 1 and num_total_tasks > 1:
        assert concurrent_time < max_allowed_concurrent_time, \
            f"Плохое масштабирование ({test_name})! Параллельное время ({concurrent_time:.4f}с) " \
            f"значительно превышает ожидаемое ({max_allowed_concurrent_time:.4f}с). " \
            f"Причины: блокировки, высокие накладные расходы (IPC, запуск процессов, импорт) или нехватка ресурсов."

        assert concurrent_time < sequential_time * 1.1, \
            f"Параллельное выполнение ({test_name}) ({concurrent_time:.4f}с) " \
            f"НЕ БЫСТРЕЕ последовательного ({sequential_time:.4f}с). " \
            f"Явная проблема с параллелизмом или огромные накладные расходы."

    print(f"--- Тест Масштабируемости ({test_name}) ПРОЙДЕН ---")



@pytest.mark.original # Метка для запуска только этого теста: pytest -m original
@pytest.mark.skipif(not EMBEDDER_IMPORTED, reason="Модуль embedder не импортирован")
def test_scaling_original(worker_test_data_original):
    """Тест масштабируемости для ОРИГИНАЛЬНОГО воркера."""
    print("\n\n=== ЗАПУСК ТЕСТА ОРИГИНАЛЬНОГО ВОРКЕРА ===")
    run_scaling_test(_embed_frame_pair_worker, worker_test_data_original, "Original Worker")

@pytest.mark.simplified # Метка: pytest -m simplified
@pytest.mark.skipif(not EMBEDDER_IMPORTED, reason="Модуль embedder не импортирован")
def test_scaling_simplified(worker_test_data_simplified):
    """Тест масштабируемости для УПРОЩЕННОГО воркера."""
    print("\n\n=== ЗАПУСК ТЕСТА УПРОЩЕННОГО ВОРКЕРА ===")
    run_scaling_test(_test_worker_simplified, worker_test_data_simplified, "Simplified Worker")

@pytest.mark.minimal # Метка: pytest -m minimal
def test_scaling_minimal(worker_test_data_minimal):
    """Тест масштабируемости для МИНИМАЛЬНОГО воркера."""
    print("\n\n=== ЗАПУСК ТЕСТА МИНИМАЛЬНОГО ВОРКЕРА ===")
    run_scaling_test(_test_worker_minimal, worker_test_data_minimal, "Minimal Worker")

@pytest.mark.batched # Метка: pytest -m batched
@pytest.mark.skipif(not EMBEDDER_IMPORTED, reason="Модуль embedder не импортирован")
def test_scaling_batched(worker_test_data_batched):
    """Тест масштабируемости для БАТЧЕЙ задач."""
    print("\n\n=== ЗАПУСК ТЕСТА БАТЧЕЙ ===")
    run_scaling_test(_test_worker_batch, worker_test_data_batched, "Batched Worker", is_batch_test=True)