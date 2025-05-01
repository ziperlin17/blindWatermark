import torch
import platform
import sys
import os
import cpuinfo # Дополнительная библиотека для информации о CPU
import logging

# Установка cpuinfo, если ее нет: pip install py-cpuinfo
try:
    import cpuinfo
except ImportError:
    logging.warning("Библиотека py-cpuinfo не установлена. Информация о CPU будет неполной.")
    logging.warning("Установите ее: pip install py-cpuinfo")
    cpuinfo = None # Чтобы избежать ошибок ниже

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("\n--- Информация о Системе и PyTorch ---")

# Информация об ОС
print(f"\nОперационная система:")
print(f"  Platform: {platform.platform()}")
print(f"  System:   {platform.system()}")
print(f"  Release:  {platform.release()}")
print(f"  Version:  {platform.version()}")
print(f"  Machine:  {platform.machine()}")
print(f"  Processor:{platform.processor()}") # Может быть не очень информативно

# Информация о Python
print(f"\nPython:")
print(f"  Version: {sys.version}")
print(f"  Executable: {sys.executable}")

# Информация о CPU
print(f"\nCPU:")
if cpuinfo:
    try:
        info = cpuinfo.get_cpu_info()
        print(f"  Brand:    {info.get('brand_raw', 'N/A')}")
        print(f"  Arch:     {info.get('arch_string_raw', 'N/A')}")
        print(f"  Hz Advertised: {info.get('hz_advertised_friendly', 'N/A')}")
        print(f"  Hz Actual:     {info.get('hz_actual_friendly', 'N/A')}")
        print(f"  Cores (Physical): {info.get('count_cores', 'N/A')}")
        print(f"  Cores (Logical):  {info.get('count', 'N/A')}")
        # print(f"  L2 Cache: {info.get('l2_cache_size', 'N/A')}") # Размер кэша может потребовать доп. обработки
        # print(f"  L3 Cache: {info.get('l3_cache_size', 'N/A')}")
        # print(f"  Flags:    {info.get('flags', [])}") # Очень много флагов
    except Exception as e:
        print(f"  Не удалось получить детальную информацию о CPU: {e}")
else:
    print("  Библиотека py-cpuinfo не установлена для детальной информации.")
print(f"  CPU Count (os): {os.cpu_count()}") # Количество логических процессоров по версии OS

# Информация о PyTorch
print(f"\nPyTorch:")
print(f"  Version: {torch.__version__}")
print(f"  Build Configuration:")
# Выводим детальную информацию о сборке, включая версии CUDA/cuDNN, с которыми он скомпилирован
try:
    print(torch.__config__.show())
    # или для более старых версий: print(torch._C._show_config())
except AttributeError:
     try:
       print(torch._C._show_config())
     except AttributeError:
       print("    Не удалось получить детальную конфигурацию сборки PyTorch.")


# Информация о CUDA и GPU (если доступно)
print(f"\nCUDA / GPU:")
cuda_available = torch.cuda.is_available()
print(f"  CUDA доступна: {'Да' if cuda_available else 'Нет'}")

if cuda_available:
    print(f"  Версия CUDA (скомпилировано с): {torch.version.cuda}")
    # Получаем версию CUDA драйвера (может отличаться от версии Toolkit)
    try:
        driver_version = torch.cuda.get_driver_version()
        print(f"  Версия CUDA драйвера: {driver_version // 1000}.{ (driver_version % 1000) // 10 }")
    except Exception:
        print("  Не удалось определить версию CUDA драйвера.")

    num_gpus = torch.cuda.device_count()
    print(f"  Количество GPU: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}:")
        try:
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            print(f"    Имя: {props.name}")
            print(f"    Общая память: {props.total_memory / (1024**3):.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Многопроцессоров: {props.multi_processor_count}")
            # print(f"    Частота ядра (max): {props.clock_rate / 1000:.2f} MHz") # Может быть не всегда доступно
            # print(f"    Частота памяти: {props.memory_clock_rate / 1000:.2f} MHz")
        except Exception as e:
             print(f"    Не удалось получить свойства для GPU {i}: {e}")

    # Проверка cuDNN
    cudnn_available = torch.backends.cudnn.is_available()
    print(f"\n  cuDNN доступна: {'Да' if cudnn_available else 'Нет'}")
    if cudnn_available:
        print(f"  Версия cuDNN: {torch.backends.cudnn.version()}")
        print(f"  cuDNN включена: {torch.backends.cudnn.enabled}")
else:
    print("  (GPU информация недоступна, так как CUDA не найдена PyTorch)")

print("\n--- Проверка завершена ---")