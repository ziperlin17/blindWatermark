import cv2
import numpy as np
import math # Понадобится для ceil

def visualize_rings(ll_subband_shape: tuple, n_rings: int):
    """
    Создает изображение, визуализирующее концентрические кольца.

    Args:
        ll_subband_shape: Кортеж (height, width) LL-подполосы.
        n_rings: Количество колец.

    Returns:
        Изображение (NumPy BGR array) с раскрашенными кольцами.
    """
    H, W = ll_subband_shape
    if H < 2 or W < 2:
        print("Ошибка: Слишком маленький размер подполосы для визуализации.")
        return None

    # --- Логика расчета индексов колец (аналогично _ring_division_internal) ---
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    rr, cc = np.indices((H, W), dtype=np.float32)
    distances = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
    min_dist, max_dist = 0.0, np.max(distances) # Кольца начинаются от центра

    if max_dist < 1e-6: # Если все точки в центре (очень маленькое изображение)
        ring_bins = np.array([0.0, 1.0])
        n_rings_eff = 1
    else:
        # Добавляем малое значение к max_dist для включения граничных пикселей
        ring_bins = np.linspace(min_dist, max_dist + 1e-6, n_rings + 1)
        n_rings_eff = n_rings

    if len(ring_bins) < 2:
        print("Ошибка: Не удалось создать границы колец.")
        return None

    # Присваиваем каждому пикселю индекс кольца (0, 1, ..., n_rings-1)
    ring_indices = np.digitize(distances, ring_bins) - 1
    # Убедимся, что центральные пиксели попадают в нулевое кольцо
    ring_indices[distances < ring_bins[1]] = 0
    # Ограничиваем индексы сверху
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)
    # --- Конец логики расчета индексов ---

    # --- Создание цветного изображения для визуализации ---
    vis_image = np.zeros((*ll_subband_shape, 3), dtype=np.uint8) # Черный фон, 3 канала (BGR)

    # Определяем цвета для каждого кольца (можно выбрать другие)
    # BGR формат: (Blue, Green, Red)
    colors = [
        (255, 0, 0),      # Синий    (Кольцо 0 - центр)
        (0, 255, 0),      # Зеленый  (Кольцо 1)
        (0, 0, 255),      # Красный  (Кольцо 2)
        (255, 255, 0),    # Голубой  (Кольцо 3)
        (255, 0, 255),    # Пурпурный(Кольцо 4)
        (0, 255, 255),    # Желтый   (Кольцо 5)
        (255, 255, 255),  # Белый    (Кольцо 6)
        (128, 128, 128),  # Серый    (Кольцо 7)
        # Добавьте больше цветов, если N_RINGS > 8
        (0, 128, 255),    # Оранжевый
        (128, 0, 128),    # Фиолетовый
    ]

    # Раскрашиваем пиксели каждого кольца
    for ring_idx in range(n_rings_eff):
        # Находим маску пикселей, принадлежащих текущему кольцу
        mask = (ring_indices == ring_idx)
        # Выбираем цвет (циклически, если колец больше, чем цветов в списке)
        color = colors[ring_idx % len(colors)]
        # Присваиваем цвет этим пикселям на визуализации
        vis_image[mask] = color

    return vis_image

# --- Пример использования ---
if __name__ == "__main__":
    ll_height = 1080
    ll_width = 1920
    num_rings = 16

    print(f"Генерация визуализации для LL-подполосы размером {ll_width}x{ll_height} с {num_rings} кольцами...")
    ring_visualization = visualize_rings((ll_height, ll_width), num_rings)

    if ring_visualization is not None:
        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        # Определяем максимальный размер окна (например, 800x600)
        max_display_width = 1200
        max_display_height = 800
        # Рассчитываем коэффициент масштабирования, чтобы вписать в окно
        scale_w = max_display_width / ll_width
        scale_h = max_display_height / ll_height
        scale = min(scale_w, scale_h) # Используем минимальный масштаб, чтобы сохранить пропорции

        # Новый размер для отображения
        display_width = int(ll_width * scale)
        display_height = int(ll_height * scale)
        display_dim = (display_width, display_height)

        # Масштабируем до нового размера
        display_vis = cv2.resize(ring_visualization, display_dim, interpolation = cv2.INTER_NEAREST)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        cv2.imshow(f"Ring Visualization ({num_rings} rings) - Scaled", display_vis) # Отображаем уменьшенное
        print("\nОкно с визуализацией колец создано.")
        # ... (остальные print) ...
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Не удалось создать визуализацию.")