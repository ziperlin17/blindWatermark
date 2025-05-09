import time
import numpy as np
import logging
import sys

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("--- Проверка PyOpenCL и OpenCL Runtime ---")

try:
    import pyopencl as cl
    logging.info("PyOpenCL импортирован успешно.")
    PYOPENCL_AVAILABLE = True
except ImportError:
    logging.error("PyOpenCL не найден. Установите его: pip install pyopencl")
    PYOPENCL_AVAILABLE = False
except Exception as e:
    logging.error(f"Ошибка при импорте PyOpenCL: {e}")
    PYOPENCL_AVAILABLE = False

platforms = []
devices = []
if PYOPENCL_AVAILABLE:
    try:
        platforms = cl.get_platforms()
        if not platforms:
            logging.warning("Не найдено ни одной платформы OpenCL.")
            logging.warning("Убедитесь, что установлены драйверы OpenCL для вашего CPU/GPU.")
        else:
            logging.info(f"Найдено платформ OpenCL: {len(platforms)}")
            for i, p in enumerate(platforms):
                logging.info(f"  Платформа {i}: {p.name} ({p.vendor})")
                try:
                    platform_devices = p.get_devices()
                    if not platform_devices:
                         logging.warning(f"    На платформе '{p.name}' не найдено устройств OpenCL.")
                    else:
                         logging.info(f"    Найдено устройств на платформе: {len(platform_devices)}")
                         for j, d in enumerate(platform_devices):
                              logging.info(f"      Устройство {j}: {d.name}")
                              logging.info(f"        Тип: {cl.device_type.to_string(d.type)}")
                              logging.info(f"        Версия OpenCL: {d.opencl_c_version}")
                              logging.info(f"        Память (Global): {d.global_mem_size / (1024**2):.0f} MB")
                              logging.info(f"        Макс. рабочая группа: {d.max_work_group_size}")
                              devices.append(d) # Добавляем устройство в общий список
                except cl.LogicError as le:
                    # Иногда get_devices может вернуть ошибку, если драйвер неполный
                    logging.warning(f"    Не удалось получить устройства для платформы '{p.name}': {le}")
                except Exception as e_dev:
                    logging.error(f"    Ошибка при получении устройств для платформы '{p.name}': {e_dev}", exc_info=True)

    except Exception as e_plat:
        logging.error(f"Ошибка при получении платформ OpenCL: {e_plat}")
        logging.error("Это может указывать на проблемы с установкой OpenCL ICD Loader или драйверов.")
        PYOPENCL_AVAILABLE = False # Считаем OpenCL неработоспособным

context = None
queue = None
if devices:
    logging.info("--- Попытка создать OpenCL контекст и очередь команд ---")
    # Попробуем использовать первое найденное устройство
    selected_device = devices[0]
    logging.info(f"Используем первое найденное устройство: {selected_device.name}")
    try:
        context = cl.Context([selected_device])
        queue = cl.CommandQueue(context)
        logging.info("Контекст и очередь команд OpenCL успешно созданы.")
        OPENCL_CONTEXT_OK = True
    except Exception as e_ctx:
        logging.error(f"Ошибка при создании контекста/очереди OpenCL: {e_ctx}", exc_info=True)
        OPENCL_CONTEXT_OK = False
else:
    if PYOPENCL_AVAILABLE:
         logging.warning("Устройства OpenCL не найдены, пропускаем создание контекста.")
    OPENCL_CONTEXT_OK = False

computation_ok = False
if context and queue:
    logging.info("--- Попытка выполнить тестовое вычисление OpenCL (сложение векторов) ---")
    try:
        vector_size = 1024 * 1024
        host_a = np.random.rand(vector_size).astype(np.float32)
        host_b = np.random.rand(vector_size).astype(np.float32)
        host_result = np.empty_like(host_a)

        mf = cl.mem_flags
        device_a = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_a)
        device_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_b)
        device_result = cl.Buffer(context, mf.WRITE_ONLY, host_result.nbytes)

        kernel_code = """
        __kernel void vector_add(__global const float *a,
                                 __global const float *b,
                                 __global float *result)
        {
            int gid = get_global_id(0);
            result[gid] = a[gid] + b[gid];
        }
        """
        program = cl.Program(context, kernel_code).build()

        logging.info("OpenCL ядро скомпилировано. Запуск вычисления...")
        start_time = time.perf_counter()
        kernel_event = program.vector_add(queue, (vector_size,), None, device_a, device_b, device_result)
        kernel_event.wait() # Ждем завершения ядра
        end_time = time.perf_counter()
        logging.info(f"Вычисление на OpenCL устройстве заняло: {end_time - start_time:.4f} сек.")

        cl.enqueue_copy(queue, host_result, device_result).wait()

        expected_result = host_a + host_b
        if np.allclose(host_result, expected_result):
            logging.info("Результат вычисления на OpenCL совпадает с ожидаемым (NumPy).")
            computation_ok = True
        else:
            logging.error("Результат вычисления на OpenCL НЕ совпадает с ожидаемым!")
            computation_ok = False

    except Exception as e_comp:
        logging.error(f"Ошибка во время выполнения вычисления OpenCL: {e_comp}", exc_info=True)
        computation_ok = False
else:
     if PYOPENCL_AVAILABLE:
          logging.warning("Контекст/очередь OpenCL не созданы, пропускаем вычисление.")

logging.info("--- Итоговый Отчет Проверки OpenCL ---")
logging.info(f"PyOpenCL установлен и импортируется: {'ДА' if PYOPENCL_AVAILABLE else 'НЕТ'}")
if PYOPENCL_AVAILABLE:
    logging.info(f"Найдены платформы OpenCL: {'ДА' if platforms else 'НЕТ'}")
    logging.info(f"Найдены устройства OpenCL: {'ДА' if devices else 'НЕТ'}")
    if devices:
         logging.info(f"Контекст и очередь команд созданы: {'ДА' if OPENCL_CONTEXT_OK else 'НЕТ'}")
         if OPENCL_CONTEXT_OK:
              logging.info(f"Тестовое вычисление выполнено успешно: {'ДА' if computation_ok else 'НЕТ'}")

print("\nПроверка OpenCL завершена. Смотрите лог выше.")

if not PYOPENCL_AVAILABLE or not devices or not OPENCL_CONTEXT_OK or not computation_ok:
    print("Обнаружены проблемы с OpenCL!")
else:
    print("Базовая проверка OpenCL пройдена успешно!")
