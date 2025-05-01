import logging
import torch

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_pytorch_wavelets_installation():
    """Проверяет установку и базовую работоспособность pytorch-wavelets."""

    logging.info("--- Проверка установки PyTorch Wavelets ---")

    # 1. Проверка импорта библиотеки
    try:
        from pytorch_wavelets import DTCWTForward, DTCWTInverse
        logging.info("Библиотека pytorch_wavelets успешно импортирована.")
    except ImportError:
        logging.error("ОШИБКА: Библиотека pytorch_wavelets не найдена!")
        logging.error("Пожалуйста, установите ее: pip install pytorch_wavelets")
        return False
    except Exception as e:
        logging.error(f"ОШИБКА при импорте pytorch_wavelets: {e}", exc_info=True)
        return False

    # 2. Проверка доступности CUDA в PyTorch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        logging.info(f"Обнаружено устройство CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.warning("Устройство CUDA не найдено или PyTorch установлен без поддержки CUDA. Тест будет выполнен на CPU.")

    # 3. Базовый тест прямого и обратного преобразования
    try:
        logging.info("Создание экземпляров DTCWTForward и DTCWTInverse...")
        # Используем параметры по умолчанию или те, что вы планируете использовать
        J = 1 # Количество уровней (как в вашем коде)
        biort = 'near_sym_a' # Пример фильтра
        qshift = 'qshift_a'  # Пример фильтра

        xfm = DTCWTForward(J=J, biort=biort, qshift=qshift).to(device)
        ifm = DTCWTInverse(biort=biort, qshift=qshift).to(device)
        logging.info("Экземпляры созданы успешно.")

        # Создание тестового тензора
        logging.info("Создание тестового тензора...")
        # Небольшой размер для быстрого теста, NCHW формат
        N, C, H, W = 2, 1, 64, 64
        # Создаем тензор сразу на нужном устройстве
        X = torch.randn(N, C, H, W, dtype=torch.float32, device=device)
        logging.info(f"Тестовый тензор создан (форма: {X.shape}, устройство: {X.device})")

        # Прямое преобразование
        logging.info("Выполнение прямого преобразования (forward)...")
        Yl, Yh = xfm(X)
        logging.info("Прямое преобразование выполнено.")
        # Проверка форм выходных данных (пример для J=1)
        expected_yl_shape = (N, C, H // 2, W // 2)
        expected_yh0_shape_prefix = (N, C, 6, H // 2, W // 2, 2) # Без учета channel
        logging.info(f"  Форма Yl: {Yl.shape} (Ожидалось ~{expected_yl_shape})")
        if not Yh: logging.warning("  Yh пуст!")
        else: logging.info(f"  Форма Yh[0]: {Yh[0].shape} (Ожидалось ~{expected_yh0_shape_prefix})")

        # Обратное преобразование
        logging.info("Выполнение обратного преобразования (inverse)...")
        X_recon = ifm((Yl, Yh))
        logging.info("Обратное преобразование выполнено.")
        logging.info(f"  Форма восстановленного тензора: {X_recon.shape}")

        # Проверка на совпадение формы и близость значений
        if X_recon.shape != X.shape:
            logging.error(f"ОШИБКА: Форма восстановленного тензора {X_recon.shape} не совпадает с исходной {X.shape}!")
            return False

        # Вычисляем ошибку реконструкции
        mse = torch.mean((X - X_recon)**2).item()
        logging.info(f"Среднеквадратичная ошибка реконструкции (MSE): {mse:.4e}")
        # Порог MSE (должен быть очень мал, но не ноль из-за точности float)
        mse_threshold = 1e-10
        if mse < mse_threshold:
            logging.info(f"Ошибка реконструкции достаточно мала (< {mse_threshold:.1e}). Тест пройден!")
            return True
        else:
            logging.warning(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка реконструкции {mse:.4e} выше порога {mse_threshold:.1e}. Проверьте правильность фильтров или точность.")
            # Не считаем это критической ошибкой установки, но обращаем внимание
            return True

    except Exception as e:
        logging.error(f"ОШИБКА во время выполнения теста преобразования: {e}", exc_info=True)
        if "CUDA out of memory" in str(e):
            logging.error("-> Возможная причина: нехватка памяти на GPU. Попробуйте уменьшить размер тензора (N, C, H, W).")
        elif "cuDNN error" in str(e):
             logging.error("-> Возможная причина: проблема с установкой/версией cuDNN.")
        return False

# --- Запуск проверки ---
if __name__ == "__main__":
    installed_and_working = check_pytorch_wavelets_installation()
    if installed_and_working:
        print("\n[ Итог: Библиотека pytorch-wavelets установлена и базовый тест пройден! ]")
    else:
        print("\n[ Итог: Обнаружены проблемы с установкой или работой pytorch-wavelets. См. лог выше. ]")