"""
Скрипт для управления цветокоррекцией камеры в реальном времени
Позволяет настраивать экспозицию, контраст, точки черного/белого и другие параметры
"""
# python Minicpm_hand\color_correction.py --camera 0   

import cv2
import numpy as np
from typing import Dict, Any


class ColorCorrectionTool:
    def __init__(self, camera_id: int = 0):
        """
        Инициализация инструмента цветокоррекции
        
        Args:
            camera_id: ID камеры (по умолчанию 0)
        """
        self.camera_id = camera_id
        self.cap = None
        self.window_name = "Color Correction Tool"
        
        # Параметры цветокоррекции (значения по умолчанию)
        self.params = {
            'brightness': 50,      # Яркость (0-100, 50 = норма)
            'contrast': 50,        # Контраст (0-100, 50 = норма)
            'exposure': 50,        # Экспозиция (0-100, 50 = авто)
            'saturation': 50,      # Насыщенность (0-100, 50 = норма)
            'hue': 0,              # Оттенок (-180 до +180, 0 = норма)
            'gamma': 50,           # Гамма (1-300, 50 = 1.0)
            'black_point': 0,      # Точка черного (0-255)
            'white_point': 255,    # Точка белого (0-255)
            'sharpness': 50,       # Резкость (0-100, 50 = норма)
            'temperature': 50,     # Цветовая температура (0-100, 50 = норма)
        }
        
        self.running = False
        
    def initialize_camera(self) -> bool:
        """Инициализация камеры"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Ошибка: Не удалось открыть камеру {self.camera_id}")
            return False
        
        # Настройка параметров камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Попытка установить автоэкспозицию
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto mode
        
        print(f"Камера {self.camera_id} успешно инициализирована")
        print(f"Разрешение: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
        
        return True
    
    def create_trackbars(self):
        """Создание trackbar'ов для управления параметрами"""
        cv2.namedWindow(self.window_name)
        
        # Создаем trackbar для каждого параметра
        cv2.createTrackbar('Brightness', self.window_name, 
                          self.params['brightness'], 100, self.on_trackbar)
        cv2.createTrackbar('Contrast', self.window_name, 
                          self.params['contrast'], 100, self.on_trackbar)
        cv2.createTrackbar('Exposure', self.window_name, 
                          self.params['exposure'], 100, self.on_trackbar)
        cv2.createTrackbar('Saturation', self.window_name, 
                          self.params['saturation'], 100, self.on_trackbar)
        cv2.createTrackbar('Hue', self.window_name, 
                          180, 360, self.on_trackbar)  # -180 to +180
        cv2.createTrackbar('Gamma', self.window_name, 
                          self.params['gamma'], 300, self.on_trackbar)
        cv2.createTrackbar('Black Point', self.window_name, 
                          self.params['black_point'], 255, self.on_trackbar)
        cv2.createTrackbar('White Point', self.window_name, 
                          self.params['white_point'], 255, self.on_trackbar)
        cv2.createTrackbar('Sharpness', self.window_name, 
                          self.params['sharpness'], 100, self.on_trackbar)
        cv2.createTrackbar('Temperature', self.window_name, 
                          self.params['temperature'], 100, self.on_trackbar)
    
    def on_trackbar(self, value):
        """Callback для trackbar'ов (не используется, но требуется OpenCV)"""
        pass
    
    def get_trackbar_values(self):
        """Получение текущих значений с trackbar'ов"""
        self.params['brightness'] = cv2.getTrackbarPos('Brightness', self.window_name)
        self.params['contrast'] = cv2.getTrackbarPos('Contrast', self.window_name)
        self.params['exposure'] = cv2.getTrackbarPos('Exposure', self.window_name)
        self.params['saturation'] = cv2.getTrackbarPos('Saturation', self.window_name)
        self.params['hue'] = cv2.getTrackbarPos('Hue', self.window_name) - 180
        self.params['gamma'] = cv2.getTrackbarPos('Gamma', self.window_name)
        self.params['black_point'] = cv2.getTrackbarPos('Black Point', self.window_name)
        self.params['white_point'] = cv2.getTrackbarPos('White Point', self.window_name)
        self.params['sharpness'] = cv2.getTrackbarPos('Sharpness', self.window_name)
        self.params['temperature'] = cv2.getTrackbarPos('Temperature', self.window_name)
    
    def apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Применение яркости и контраста
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        brightness = (self.params['brightness'] - 50) * 2  # -100 to +100
        contrast = self.params['contrast'] / 50.0  # 0 to 2
        
        # Применяем контраст и яркость
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        return image
    
    def apply_exposure(self, image: np.ndarray) -> np.ndarray:
        """
        Применение экспозиции
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        exposure = self.params['exposure'] / 50.0  # 0 to 2
        
        # Попытка установить экспозицию камеры
        if self.params['exposure'] == 50:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, (exposure - 1) * 10)
        
        # Программная коррекция экспозиции
        image = np.clip(image * exposure, 0, 255).astype(np.uint8)
        
        return image
    
    def apply_saturation_hue(self, image: np.ndarray) -> np.ndarray:
        """
        Применение насыщенности и оттенка
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        # Конвертируем в HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Применяем оттенок
        hue_shift = self.params['hue']
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Применяем насыщенность
        saturation = self.params['saturation'] / 50.0  # 0 to 2
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        # Конвертируем обратно в BGR
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image
    
    def apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        Применение гамма-коррекции
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        gamma = self.params['gamma'] / 50.0  # 0.02 to 6
        
        if gamma == 1.0:
            return image
        
        # Создаем lookup table для гамма-коррекции
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        # Применяем гамма-коррекцию
        image = cv2.LUT(image, table)
        
        return image
    
    def apply_black_white_points(self, image: np.ndarray) -> np.ndarray:
        """
        Применение точек черного и белого
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        black_point = self.params['black_point']
        white_point = max(self.params['white_point'], black_point + 1)
        
        # Применяем уровни
        image = image.astype(np.float32)
        image = np.clip((image - black_point) * (255.0 / (white_point - black_point)), 0, 255)
        image = image.astype(np.uint8)
        
        return image
    
    def apply_sharpness(self, image: np.ndarray) -> np.ndarray:
        """
        Применение резкости
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        sharpness = (self.params['sharpness'] - 50) / 50.0  # -1 to 1
        
        if abs(sharpness) < 0.01:
            return image
        
        # Создаем kernel для резкости
        if sharpness > 0:
            # Увеличение резкости
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * sharpness
            kernel[1, 1] = 1 + 8 * sharpness
        else:
            # Размытие
            kernel_size = int(abs(sharpness) * 10) * 2 + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return image
        
        # Применяем фильтр резкости
        image = cv2.filter2D(image, -1, kernel)
        
        return image
    
    def apply_temperature(self, image: np.ndarray) -> np.ndarray:
        """
        Применение цветовой температуры
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        temperature = (self.params['temperature'] - 50) / 50.0  # -1 to 1
        
        if abs(temperature) < 0.01:
            return image
        
        # Конвертируем в float для точных вычислений
        image = image.astype(np.float32)
        
        if temperature > 0:
            # Теплые тона (больше красного/желтого)
            image[:, :, 2] = np.clip(image[:, :, 2] * (1 + temperature * 0.3), 0, 255)  # Red
            image[:, :, 1] = np.clip(image[:, :, 1] * (1 + temperature * 0.1), 0, 255)  # Green
            image[:, :, 0] = np.clip(image[:, :, 0] * (1 - temperature * 0.2), 0, 255)  # Blue
        else:
            # Холодные тона (больше синего)
            image[:, :, 0] = np.clip(image[:, :, 0] * (1 - temperature * 0.3), 0, 255)  # Blue
            image[:, :, 1] = np.clip(image[:, :, 1] * (1 - temperature * 0.1), 0, 255)  # Green
            image[:, :, 2] = np.clip(image[:, :, 2] * (1 + temperature * 0.2), 0, 255)  # Red
        
        return image.astype(np.uint8)
    
    def apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Применение всех эффектов цветокоррекции
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        # Применяем эффекты в определенном порядке
        image = self.apply_exposure(image)
        image = self.apply_black_white_points(image)
        image = self.apply_gamma(image)
        image = self.apply_brightness_contrast(image)
        image = self.apply_saturation_hue(image)
        image = self.apply_temperature(image)
        image = self.apply_sharpness(image)
        
        return image
    
    def draw_info(self, image: np.ndarray) -> np.ndarray:
        """
        Отрисовка информации о параметрах на изображении
        
        Args:
            image: Исходное изображение
            
        Returns:
            Изображение с информацией
        """
        info_lines = [
            f"Brightness: {self.params['brightness']}",
            f"Contrast: {self.params['contrast']}",
            f"Exposure: {self.params['exposure']}",
            f"Saturation: {self.params['saturation']}",
            f"Hue: {self.params['hue']}",
            f"Gamma: {self.params['gamma'] / 50.0:.2f}",
            f"Black Point: {self.params['black_point']}",
            f"White Point: {self.params['white_point']}",
            f"Sharpness: {self.params['sharpness']}",
            f"Temperature: {self.params['temperature']}",
        ]
        
        # Создаем полупрозрачный фон для текста
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 270), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        
        # Отрисовываем текст
        y_offset = 30
        for line in info_lines:
            cv2.putText(image, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Добавляем подсказки по управлению
        cv2.putText(image, "Q - Выход | R - Сброс | S - Сохранить настройки", 
                   (20, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return image
    
    def reset_parameters(self):
        """Сброс параметров к значениям по умолчанию"""
        self.params = {
            'brightness': 50,
            'contrast': 50,
            'exposure': 50,
            'saturation': 50,
            'hue': 0,
            'gamma': 50,
            'black_point': 0,
            'white_point': 255,
            'sharpness': 50,
            'temperature': 50,
        }
        
        # Обновляем trackbar'ы
        cv2.setTrackbarPos('Brightness', self.window_name, 50)
        cv2.setTrackbarPos('Contrast', self.window_name, 50)
        cv2.setTrackbarPos('Exposure', self.window_name, 50)
        cv2.setTrackbarPos('Saturation', self.window_name, 50)
        cv2.setTrackbarPos('Hue', self.window_name, 180)
        cv2.setTrackbarPos('Gamma', self.window_name, 50)
        cv2.setTrackbarPos('Black Point', self.window_name, 0)
        cv2.setTrackbarPos('White Point', self.window_name, 255)
        cv2.setTrackbarPos('Sharpness', self.window_name, 50)
        cv2.setTrackbarPos('Temperature', self.window_name, 50)
        
        print("Параметры сброшены к значениям по умолчанию")
    
    def save_parameters(self, filename: str = "color_correction_params.txt"):
        """
        Сохранение параметров в файл
        
        Args:
            filename: Имя файла для сохранения
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Параметры цветокоррекции\n")
            for key, value in self.params.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Параметры сохранены в {filename}")
    
    def run(self):
        """Основной цикл работы"""
        if not self.initialize_camera():
            return
        
        self.create_trackbars()
        self.running = True
        
        print("\n=== Инструмент цветокоррекции ===")
        print("Управление:")
        print("  Q - Выход")
        print("  R - Сброс параметров")
        print("  S - Сохранить параметры")
        print("\nИспользуйте ползунки для настройки параметров")
        print("=====================================\n")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Ошибка: Не удалось получить кадр с камеры")
                break
            
            # Получаем текущие значения параметров
            self.get_trackbar_values()
            
            # Применяем цветокоррекцию
            corrected_frame = self.apply_color_correction(frame.copy())
            
            # Создаем изображение для отображения (оригинал + обработанное)
            # Уменьшаем размер для удобства отображения
            display_height = 480
            scale = display_height / frame.shape[0]
            display_width = int(frame.shape[1] * scale)
            
            original_resized = cv2.resize(frame, (display_width, display_height))
            corrected_resized = cv2.resize(corrected_frame, (display_width, display_height))
            
            # Добавляем подписи
            cv2.putText(original_resized, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(corrected_resized, "Corrected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Объединяем изображения горизонтально
            combined = np.hstack([original_resized, corrected_resized])
            
            # Добавляем информацию о параметрах
            combined = self.draw_info(combined)
            
            # Отображаем результат
            cv2.imshow(self.window_name, combined)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q или ESC
                self.running = False
            elif key == ord('r') or key == ord('R'):  # R - сброс
                self.reset_parameters()
            elif key == ord('s') or key == ord('S'):  # S - сохранить
                self.save_parameters()
        
        # Освобождаем ресурсы
        self.cleanup()
    
    def cleanup(self):
        """Освобождение ресурсов"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Инструмент цветокоррекции камеры')
    parser.add_argument('--camera', type=int, default=0, 
                       help='ID камеры (по умолчанию 0)')
    
    args = parser.parse_args()
    
    # Создаем и запускаем инструмент
    tool = ColorCorrectionTool(camera_id=args.camera)
    tool.run()


if __name__ == "__main__":
    main()
