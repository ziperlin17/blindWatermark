import unittest
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.linalg import svd
import dtcwt
from dtcwt import Transform2d
import logging
from typing import List, Optional, Tuple

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Константы, соответствующие extractor.py ---
# Убедитесь, что они совпадают с вашим extractor.py
N_RINGS = 8
EMBED_COMPONENT = 0 # 0:Y, 1:Cr, 2:Cb - ИЗМЕНИТЕ ЕСЛИ НУЖНО

class TestWatermarkingStagesVisualization(unittest.TestCase):

    def setUp(self):
        """Подготовка тестовых данных и параметров."""
        self.n_rings = N_RINGS
        self.embed_component = EMBED_COMPONENT
        self.visualize = True
        self.video_path = '../input1080.mp4'  # <--- УКАЖИТЕ ПУТЬ К ВИДЕО
        self.frame_num_t = 270  # <--- УКАЖИТЕ НОМЕР ПЕРВОГО КАДРА
        self.frame_num_t_plus_1 = self.frame_num_t + 1

        # Загрузка кадров (теперь с извлечением нужного компонента)
        self.frame_t_comp, self.frame_t_plus_1_comp = self.load_two_frames_component(
            self.video_path, self.frame_num_t, self.frame_num_t_plus_1, self.embed_component
        )
        if self.frame_t_comp is None or self.frame_t_plus_1_comp is None:
            self.fail(f"Не удалось загрузить компоненты кадров {self.frame_num_t} и {self.frame_num_t_plus_1}")

        # Преобразуем в float32 [0, 1]
        self.frame_t_float = self.frame_t_comp.astype(np.float32) / 255.0
        self.frame_t_plus_1_float = self.frame_t_plus_1_comp.astype(np.float32) / 255.0

        self.dtcwt_transform = dtcwt.Transform2d()
        logging.info(f"Setup complete for frames {self.frame_num_t} & {self.frame_num_t_plus_1}, component {self.embed_component}")

    # ИСПРАВЛЕНО: Загружает кадры и извлекает нужный компонент
    def load_two_frames_component(self, video_path: str, frame_n: int, frame_n_plus_1: int, component_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Загружает два кадра из видео и извлекает указанный компонент YCrCb."""
        cap = cv2.VideoCapture(video_path)
        frame1_comp, frame2_comp = None, None
        if not cap.isOpened():
            logging.error(f"Не удалось открыть видео: {video_path}")
            cap.release()
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_n_plus_1 >= total_frames:
            logging.error(f"Запрошен кадр {frame_n_plus_1}, но в видео всего {total_frames} кадров.")
            cap.release()
            return None, None

        # Чтение первого кадра
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
        ret1, frame_bgr1 = cap.read()
        if ret1 and frame_bgr1 is not None:
            try:
                frame_ycrcb1 = cv2.cvtColor(frame_bgr1, cv2.COLOR_BGR2YCrCb)
                frame1_comp = frame_ycrcb1[:, :, component_idx].copy() # Копируем, чтобы избежать проблем
                logging.info(f"Загружен кадр {frame_n}, извлечен компонент {component_idx}, размер: {frame1_comp.shape}")
            except IndexError:
                logging.error(f"Неверный индекс компонента {component_idx} для кадра {frame_n}")
            except cv2.error as e:
                logging.error(f"Ошибка OpenCV при обработке кадра {frame_n}: {e}")
        else:
            logging.error(f"Не удалось прочитать кадр {frame_n}")

        # Чтение второго кадра
        # Переустанавливаем позицию, т.к. read() мог сдвинуть указатель
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n_plus_1)
        ret2, frame_bgr2 = cap.read()
        if ret2 and frame_bgr2 is not None:
             try:
                frame_ycrcb2 = cv2.cvtColor(frame_bgr2, cv2.COLOR_BGR2YCrCb)
                frame2_comp = frame_ycrcb2[:, :, component_idx].copy()
                logging.info(f"Загружен кадр {frame_n_plus_1}, извлечен компонент {component_idx}, размер: {frame2_comp.shape}")
             except IndexError:
                 logging.error(f"Неверный индекс компонента {component_idx} для кадра {frame_n_plus_1}")
             except cv2.error as e:
                 logging.error(f"Ошибка OpenCV при обработке кадра {frame_n_plus_1}: {e}")
        else:
            logging.error(f"Не удалось прочитать кадр {frame_n_plus_1}")

        cap.release()

        # Проверка на совпадение размеров, если оба компонента загружены
        if frame1_comp is not None and frame2_comp is not None and frame1_comp.shape != frame2_comp.shape:
            logging.error("Размеры компонентов загруженных кадров не совпадают!")
            return None, None
        elif frame1_comp is None or frame2_comp is None:
             logging.error("Не удалось загрузить оба компонента кадров.")
             return None, None


        return frame1_comp, frame2_comp


    # --- Вспомогательные функции (остаются без изменений) ---
    def ring_division(self, lp: np.ndarray, nr: int) -> List[Optional[np.ndarray]]:
        """Разбивает 2D массив на N концентрических колец."""
        if lp is None or lp.ndim != 2:
             logging.error("Invalid input to ring_division")
             return [None] * nr
        H, W = lp.shape
        if H < 2 or W < 2: return [None] * nr
        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
        rr, cc = np.indices((H, W), dtype=np.float32)
        distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2)
        min_dist, max_dist = 0.0, np.max(distances)
        if max_dist < 1e-6: ring_bins = np.array([0.0, max_dist+1e-6]*(nr + 1))[:nr+1]
        else: ring_bins = np.linspace(min_dist, max_dist + 1e-6, nr + 1)

        n_rings_eff = len(ring_bins)-1
        if n_rings_eff <= 0: return [None] * nr

        ring_indices = np.digitize(distances, ring_bins) - 1
        ring_indices[distances < ring_bins[1]] = 0
        ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)

        rings: List[Optional[np.ndarray]] = [None] * nr
        for rdx in range(n_rings_eff):
            coords = np.argwhere(ring_indices == rdx)
            if coords.shape[0] > 0: rings[rdx] = coords
        return rings

    def dct_1d(self, s: np.ndarray) -> np.ndarray:
        """1D Дискретное Косинусное Преобразование."""
        return dct(s, type=2, norm='ortho')

    def get_s1_for_ring(self, frame_component_float: np.ndarray, ring_idx: int) -> Optional[float]:
        """Вычисляет s1 для указанного кольца в одном компоненте кадра."""
        if frame_component_float is None:
            logging.error("Получен пустой компонент кадра в get_s1_for_ring")
            return None

        # 1. DTCWT
        try:
            pyramid = self.dtcwt_transform.forward(frame_component_float, nlevels=1)
            if pyramid is None or pyramid.lowpass is None:
                logging.error("DTCWT failed for frame component in get_s1_for_ring")
                return None
            ll = pyramid.lowpass # Комплексный
        except Exception as e:
             logging.error(f"Ошибка DTCWT в get_s1_for_ring: {e}", exc_info=True)
             return None

        # 2. Разделение на кольца (по магнитуде)
        ll_magnitude = np.abs(ll)
        rings = self.ring_division(ll_magnitude, self.n_rings)
        if not (0 <= ring_idx < len(rings)):
            logging.error(f"Неверный индекс кольца {ring_idx}")
            return None
        coords = rings[ring_idx]
        if coords is None or coords.shape[0] < 10: # Пропускаем пустые или маленькие кольца
            # Логируем как DEBUG, так как это может быть нормальной ситуацией
            logging.debug(f"Кольцо {ring_idx} пустое или слишком маленькое")
            return None

        # 3. Извлечение значений (магнитуда) и DCT
        try:
            # Используем ll (комплексный) для извлечения значений
            ring_values_complex = ll[coords[:, 0], coords[:, 1]].astype(np.complex64)
            # Берем магнитуду перед DCT
            ring_values_magnitude = np.abs(ring_values_complex).astype(np.float32)
            dct_coeffs = self.dct_1d(ring_values_magnitude)
            if not np.all(np.isfinite(dct_coeffs)):
                logging.error(f"DCT coeffs contain NaN/inf for ring {ring_idx}")
                return None
        except IndexError as e:
             logging.error(f"IndexError при извлечении значений кольца {ring_idx}: {e}")
             return None
        except Exception as e:
             logging.error(f"Ошибка при извлечении/DCT кольца {ring_idx}: {e}", exc_info=True)
             return None

        # 4. SVD
        try:
            U, s_values, Vt = svd(dct_coeffs.reshape(-1, 1), full_matrices=False)
            if s_values is None or not np.all(np.isfinite(s_values)):
                logging.error(f"SVD values contain NaN/inf or None for ring {ring_idx}")
                return None
            if s_values.size == 0:
                logging.warning(f"SVD returned 0 values for ring {ring_idx}")
                return None # Возвращаем None, если нет значений

            s1 = s_values[0]
            # Проверка на очень маленькое значение s1
            if abs(s1) < 1e-9:
                 logging.warning(f"s1 for ring {ring_idx} is very close to zero ({s1:.2e})")
                 # Вернем 0.0, чтобы избежать ошибок деления, но это плохой знак
                 # Или можно вернуть None, чтобы явно указать на проблему
                 return 0.0 # Оставляем 0.0 пока
            return float(s1) # Убедимся, что возвращается стандартный float
        except np.linalg.LinAlgError as e:
             logging.error(f"SVD failed for ring {ring_idx}: {e}")
             return None
        except Exception as e:
             logging.error(f"Unexpected error during SVD for ring {ring_idx}: {e}", exc_info=True)
             return None


    # --- Тесты ---

    def test_01_visualize_input_components(self):
        """Визуализирует два загруженных компонента кадра."""
        logging.info("Running test_01_visualize_input_components...")
        if self.visualize:
            plt.figure(figsize=(12, 6))
            plt.suptitle(f'Компонент {self.embed_component} для кадров {self.frame_num_t} и {self.frame_num_t_plus_1}')

            plt.subplot(1, 2, 1)
            im1 = plt.imshow(self.frame_t_comp, cmap='gray') # Используем gray для одного канала
            plt.title(f'Кадр {self.frame_num_t}')
            plt.colorbar(im1, label=f'Component {self.embed_component} Value')

            plt.subplot(1, 2, 2)
            im2 = plt.imshow(self.frame_t_plus_1_comp, cmap='gray')
            plt.title(f'Кадр {self.frame_num_t_plus_1}')
            plt.colorbar(im2, label=f'Component {self.embed_component} Value')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
        self.assertIsNotNone(self.frame_t_comp)
        self.assertIsNotNone(self.frame_t_plus_1_comp)


    def test_02_compare_s1_values(self):
        """Сравнивает значения s1 для всех колец между двумя кадрами."""
        logging.info("Running test_02_compare_s1_values...")
        s1_values_t = []
        s1_values_t_plus_1 = []
        ratios = []
        ring_indices_analyzed = []

        print("\n--- Сравнение s1 для Колец ---")
        print(f"{'Ring':<5} | {'s1(t)':<15} | {'s1(t+1)':<15} | {'Ratio':<15} | {'Implied Bit':<12}")
        print("-" * 68)

        for ring_idx in range(self.n_rings):
            # Используем float компоненты кадров
            s1_t = self.get_s1_for_ring(self.frame_t_float, ring_idx)
            s1_t1 = self.get_s1_for_ring(self.frame_t_plus_1_float, ring_idx)

            if s1_t is not None and s1_t1 is not None:
                ring_indices_analyzed.append(ring_idx)
                s1_values_t.append(s1_t)
                s1_values_t_plus_1.append(s1_t1)

                ratio_str = "N/A (s1(t+1)≈0)"
                bit_str = "N/A"
                if abs(s1_t1) > 1e-12:
                    ratio = s1_t / s1_t1
                    ratios.append(ratio)
                    ratio_str = f"{ratio:.6f}"
                    extracted_bit = 0 if ratio >= 1.0 else 1
                    bit_str = str(extracted_bit)
                else:
                    ratios.append(np.nan)
                    logging.warning(f"s1(t+1) для кольца {ring_idx} слишком близок к нулю ({s1_t1:.2e}) для вычисления отношения.")


                print(f"{ring_idx:<5} | {s1_t:<15.6e} | {s1_t1:<15.6e} | {ratio_str:<15} | {bit_str:<12}")
            else:
                reason_t = "OK" if s1_t is not None else "Failed"
                reason_t1 = "OK" if s1_t1 is not None else "Failed"
                print(f"{ring_idx:<5} | {reason_t:<15} | {reason_t1:<15} | {'N/A':<15} | {'N/A':<12}")

        print("-" * 68)

        self.assertGreater(len(ring_indices_analyzed), 0, "Не удалось вычислить s1 ни для одного кольца в обоих кадрах.")

        s1_values_t = np.array(s1_values_t)
        s1_values_t_plus_1 = np.array(s1_values_t_plus_1)
        ratios = np.array(ratios)
        ring_indices_analyzed = np.array(ring_indices_analyzed)

        relative_diff = np.abs(s1_values_t - s1_values_t_plus_1) / (np.abs(s1_values_t) + 1e-9)
        mean_relative_diff = np.mean(relative_diff)
        logging.info(f"Средняя относительная разница |s1(t)-s1(t+1)|/|s1(t)|: {mean_relative_diff:.4f}")

        mean_ratio = np.nanmean(ratios)
        std_ratio = np.nanstd(ratios)
        logging.info(f"Статистика отношений s1(t)/s1(t+1) (NaN ignored): mean={mean_ratio:.4f}, std={std_ratio:.4f}")

        if self.visualize and len(ring_indices_analyzed) > 0:
            plt.figure(figsize=(14, 6))
            plt.suptitle(f'Сравнение s1 и Отношения для Колец ({self.frame_num_t} vs {self.frame_num_t_plus_1}, Комп. {self.embed_component})')

            plt.subplot(1, 2, 1)
            plt.plot(ring_indices_analyzed, s1_values_t, 'o-', label=f's1 (Кадр {self.frame_num_t})')
            plt.plot(ring_indices_analyzed, s1_values_t_plus_1, 's--', label=f's1 (Кадр {self.frame_num_t_plus_1})')
            plt.xlabel('Индекс кольца')
            plt.ylabel('Значение s1')
            plt.title('Сравнение значений s1')
            plt.xticks(ring_indices_analyzed)
            plt.yscale('log')
            plt.grid(True, which="both", ls="--")
            plt.legend()

            plt.subplot(1, 2, 2)
            valid_ratios_mask = ~np.isnan(ratios)
            if np.any(valid_ratios_mask):
                plt.plot(ring_indices_analyzed[valid_ratios_mask], ratios[valid_ratios_mask], 'o-', label='s1(t) / s1(t+1)')
            else:
                plt.text(0.5, 0.5, 'Нет валидных отношений для отображения', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

            plt.axhline(1.0, color='r', linestyle='--', label='Порог = 1.0') # Пороговая линия
            plt.xlabel('Индекс кольца')
            plt.ylabel('Отношение')
            plt.title('Отношение s1(t) / s1(t+1)')
            if len(ring_indices_analyzed)>0: plt.xticks(ring_indices_analyzed)
            plt.grid(True)
            plt.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestWatermarkingStagesVisualization))
    runner = unittest.TextTestRunner(verbosity=2) # verbosity=2 для более детального вывода
    runner.run(suite)
    if plt.get_fignums():
        print("\nЗакройте окна с графиками для завершения.")
        plt.show()