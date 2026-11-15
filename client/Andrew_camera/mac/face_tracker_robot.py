import json
import logging
import time
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Deque
from collections import deque

from feetech_servo import FeetechServo
from calibration import CalibrationManager
from pid_controller import PIDController

logger = logging.getLogger(__name__)


class FaceTrackerRobot:
    def __init__(
        self,
        pan_servo_id: int,
        tilt_servo_id: int,
        port: str,
        camera_index: int = 0,
        shape_predictor_path: Optional[str] = None,
        mock_servos: bool = False,
        calibration_file: str = "calibration.json",
        use_cnn: bool = False,
        camera_calibration_file: str = "camera_calibration.json",
        force_fisheye: bool = False,
    ):
        self.mock_servos = mock_servos
        self.calibration_manager = CalibrationManager(calibration_file)
        self.calibration_manager.load()
        
        self.camera_calibration_file = Path(camera_calibration_file)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.is_fisheye: bool = False
        self.undistort_map1: Optional[np.ndarray] = None
        self.undistort_map2: Optional[np.ndarray] = None
        self.calibrated_image_size: Optional[Tuple[int, int]] = None
        self.force_fisheye: bool = force_fisheye
        
        if not mock_servos:
            pan_calib = self.calibration_manager.get_calibration("pan")
            tilt_calib = self.calibration_manager.get_calibration("tilt")
            
            self.pan_servo = FeetechServo(
                pan_servo_id,
                port,
                min_position=pan_calib.min_position if pan_calib else None,
                max_position=pan_calib.max_position if pan_calib else None,
                center_position=pan_calib.center_position if pan_calib else None,
            )
            self.tilt_servo = FeetechServo(
                tilt_servo_id,
                port,
                min_position=tilt_calib.min_position if tilt_calib else None,
                max_position=tilt_calib.max_position if tilt_calib else None,
                center_position=tilt_calib.center_position if tilt_calib else None,
            )
        else:
            self.pan_servo = None
            self.tilt_servo = None
        
        self.cap = None
        self.camera_index = camera_index
        self.use_cnn = use_cnn
        self.detection_model = "cnn" if use_cnn else "hog"
            
        self.image_center = None
        
        self.pan_position = 0.0
        self.tilt_position = 0.0
        self.invert_pan = False
        self.invert_tilt = False
        
        self.pan_pid = PIDController(kp=0.15, ki=0.00, kd=0.0, min_output=-1.0, max_output=1.0, integral_limit=0.15, deadband=0.003)
        self.tilt_pid = PIDController(kp=0.15, ki=0.00, kd=0.0, min_output=-1.0, max_output=1.0, integral_limit=0.15, deadband=0.003)
        
        self.max_step_size = 0.05
        
        self.last_normalized_error_x = 0.0
        self.last_normalized_error_y = 0.0
        self.last_pan_output = 0.0
        self.last_tilt_output = 0.0
        
        self.upsample_count = 0
        self.show_preprocessed = False
        self.use_heavy_denoising = False
        
        self.detection_scale = 0.5
        self.frame_skip_interval = 1
        
        self.enable_color_correction = True
        self.green_tint_reduction = 0.3
        self.white_balance_strength = 0.7
        self.rotate_180 = False
        
        self.face_history: Deque[Tuple[int, int]] = deque(maxlen=5)
        self.detection_threshold = 0.3
        self.min_face_size = 15
        self.max_face_size = 2000
        self.show_all_detections = False
        self.last_detections: list[Tuple[int, int, int, int]] = []
        self.enable_detection_fallback = True
        self.max_fallback_upsamples = 2
        
        self._load_camera_calibration()
        
    def _load_camera_calibration(self) -> bool:
        if not self.camera_calibration_file.exists():
            logger.info(f"Camera calibration file {self.camera_calibration_file} not found, skipping undistortion")
            return False
            
        try:
            with open(self.camera_calibration_file, 'r') as f:
                data = json.load(f)
            
            camera_matrix_list = data.get('camera_matrix', [])
            dist_coeffs_list = data.get('dist_coeffs', [])
            image_size = data.get('image_size', [])
            model = data.get('model', 'pinhole')
            
            if not camera_matrix_list or not dist_coeffs_list:
                logger.warning("Camera calibration file missing required parameters")
                return False
            
            self.camera_matrix = np.array(camera_matrix_list, dtype=np.float64)
            self.dist_coeffs = np.array(dist_coeffs_list, dtype=np.float64)
            
            if self.force_fisheye:
                self.is_fisheye = True
            else:
                self.is_fisheye = (model == 'fisheye')
                if not self.is_fisheye and len(dist_coeffs_list) == 4:
                    self.is_fisheye = True
            
            if self.is_fisheye:
                if len(self.dist_coeffs) > 4:
                    self.dist_coeffs = self.dist_coeffs[:4].reshape(4, 1)
                elif len(self.dist_coeffs.shape) == 1:
                    self.dist_coeffs = self.dist_coeffs.reshape(-1, 1)
                if self.dist_coeffs.shape[0] != 4:
                    logger.warning(f"Fisheye model expects 4 distortion coefficients, got {self.dist_coeffs.shape[0]}")
                    self.dist_coeffs = self.dist_coeffs[:4].reshape(4, 1)
            
            self.calibrated_image_size = tuple(image_size) if image_size else None
            
            logger.info(f"Loaded camera calibration: {'fisheye' if self.is_fisheye else 'pinhole'} model")
            return True
        except Exception as e:
            logger.error(f"Failed to load camera calibration: {e}")
            return False
    
    def _init_undistort_maps(self, image_size: Tuple[int, int]) -> bool:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False
        
        try:
            if self.is_fisheye:
                new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.camera_matrix,
                    self.dist_coeffs,
                    image_size,
                    np.eye(3),
                    balance=0.0
                )
                self.undistort_map1, self.undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.camera_matrix,
                    self.dist_coeffs,
                    np.eye(3),
                    new_camera_matrix,
                    image_size,
                    cv2.CV_16SC2
                )
            else:
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix,
                    self.dist_coeffs,
                    image_size,
                    1.0,
                    image_size
                )
                self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    new_camera_matrix,
                    image_size,
                    cv2.CV_16SC2
                )
            logger.info(f"Initialized undistortion maps for image size {image_size}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize undistortion maps: {e}")
            return False
    
    def undistort_image(self, frame: np.ndarray) -> np.ndarray:
        if self.undistort_map1 is None or self.undistort_map2 is None:
            return frame
        
        h, w = frame.shape[:2]
        if self.undistort_map1.shape[:2] != (h, w):
            if not self._init_undistort_maps((w, h)):
                return frame
        
        return cv2.remap(
            frame,
            self.undistort_map1,
            self.undistort_map2,
            cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT
        )
        
    def connect(self) -> bool:
        if not self.mock_servos:
            if not self.pan_servo.connect():
                return False
            if not self.tilt_servo.connect():
                self.pan_servo.disconnect()
                return False
        
        if self.cap is None:
            logger.info(f"Opening camera {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
                self.cap.set(cv2.CAP_PROP_SATURATION, 1.2)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 1.1)
            except Exception as e:
                logger.debug(f"Could not set some camera properties: {e}")
            
            time.sleep(0.5)
            
        for attempt in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                break
            logger.warning(f"Camera read attempt {attempt + 1}/5 failed, retrying...")
            time.sleep(0.2)
        
        if not ret or frame is None:
            logger.error(f"Failed to read from camera {self.camera_index} after 5 attempts")
            logger.error("Make sure the camera is connected and not being used by another application")
            return False
            
        h, w = frame.shape[:2]
        self.image_center = (w // 2, h // 2)
        logger.info(f"Camera initialized: {w}x{h}")
        
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            if not self._init_undistort_maps((w, h)):
                logger.warning("Failed to initialize undistortion maps")
        
        if not self.mock_servos:
            pan_current = self.pan_servo.get_position_normalized()
            tilt_current = self.tilt_servo.get_position_normalized()
            
            if pan_current is not None:
                self.pan_position = pan_current
                logger.info(f"Pan servo current position: {pan_current:.3f} (normalized)")
            else:
                logger.warning("Could not read pan servo position, starting at center (0.0)")
                self.pan_position = 0.0
                
            if tilt_current is not None:
                self.tilt_position = tilt_current
                logger.info(f"Tilt servo current position: {tilt_current:.3f} (normalized)")
            else:
                logger.warning("Could not read tilt servo position, starting at center (0.0)")
                self.tilt_position = 0.0
            
            if abs(self.pan_position) > 0.01 or abs(self.tilt_position) > 0.01:
                logger.info("Moving servos to maintain current position...")
                self.pan_servo.set_position_normalized(self.pan_position)
                self.tilt_servo.set_position_normalized(self.tilt_position)
                time.sleep(0.5)
            else:
                logger.info("Servos already at center position, no movement needed")
                self.pan_servo.set_position_normalized(0.0)
                self.tilt_servo.set_position_normalized(0.0)
            
            self.pan_pid.reset()
            self.tilt_pid.reset()
            
        calib_status = "calibrated" if (
            self.calibration_manager.has_calibration("pan") and
            self.calibration_manager.has_calibration("tilt")
        ) else "uncalibrated"
        logger.info(
            f"Face tracker robot connected (servos: {'enabled' if not self.mock_servos else 'mock mode'}, "
            f"calibration: {calib_status})"
        )
        return True
        
    def disconnect(self):
        if not self.mock_servos:
            self.pan_servo.disable_torque()
            self.tilt_servo.disable_torque()
            self.pan_servo.disconnect()
            self.tilt_servo.disconnect()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 70:
            brightened = cv2.convertScaleAbs(gray, alpha=1.5, beta=50)
        elif mean_brightness < 90:
            brightened = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)
        elif mean_brightness < 120:
            brightened = cv2.convertScaleAbs(gray, alpha=1.1, beta=15)
        else:
            brightened = gray
            
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(brightened)
        
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        sharpened = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)
        
        if self.use_heavy_denoising:
            denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
            result = denoised
        else:
            result = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel * 0.5 + np.eye(3) * 0.5)
        
        result = np.ascontiguousarray(result, dtype=np.uint8)
        return result
    
    def correct_color(self, frame: np.ndarray) -> np.ndarray:
        if not self.enable_color_correction:
            return frame
        
        corrected = frame.copy().astype(np.float32)
        
        b, g, r = cv2.split(corrected)
        
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)
        gray_value = (mean_b + mean_g + mean_r) / 3.0
        
        if gray_value > 1.0:
            gain_b = gray_value / mean_b if mean_b > 1.0 else 1.0
            gain_g = gray_value / mean_g if mean_g > 1.0 else 1.0
            gain_r = gray_value / mean_r if mean_r > 1.0 else 1.0
            
            max_gain = max(gain_b, gain_g, gain_r)
            if max_gain > 3.0:
                gain_b = 1.0 + (gain_b - 1.0) * (2.0 / max_gain)
                gain_g = 1.0 + (gain_g - 1.0) * (2.0 / max_gain)
                gain_r = 1.0 + (gain_r - 1.0) * (2.0 / max_gain)
            
            white_balance_strength = self.white_balance_strength
            gain_b = 1.0 + (gain_b - 1.0) * white_balance_strength
            gain_g = 1.0 + (gain_g - 1.0) * white_balance_strength
            gain_r = 1.0 + (gain_r - 1.0) * white_balance_strength
            
            b = np.clip(b * gain_b, 0, 255)
            g = np.clip(g * gain_g, 0, 255)
            r = np.clip(r * gain_r, 0, 255)
        
        if self.green_tint_reduction > 0:
            green_reduction = self.green_tint_reduction
            g = np.clip(g * (1.0 - green_reduction * 0.3), 0, 255)
            r = np.clip(r * (1.0 + green_reduction * 0.15), 0, 255)
            b = np.clip(b * (1.0 + green_reduction * 0.15), 0, 255)
        
        corrected = cv2.merge([b, g, r])
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        corrected = cv2.merge([l_channel, a_channel, b_channel])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
        
        return corrected
        
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        def detect_once(scale: float, upsample: int):
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(
                rgb,
                model=self.detection_model,
                number_of_times_to_upsample=upsample
            )
            valid = []
            all_rects = []
            for loc in locs:
                top, right, bottom, left = loc
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
                w = right - left
                h = bottom - top
                area = w * h
                if self.min_face_size <= min(w, h) <= self.max_face_size:
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
                    if aspect_ratio < 2.5:
                        valid.append(((top, right, bottom, left), area))
                        all_rects.append((left, top, w, h))
            return valid, all_rects
        
        base_scale = self.detection_scale
        base_ups = self.upsample_count
        scales = [base_scale]
        if self.enable_detection_fallback:
            if base_scale < 0.75:
                scales.append(0.75)
            if 1.0 not in scales:
                scales.append(1.0)
        scales = list(dict.fromkeys(scales))
        ups_list = [base_ups]
        if self.enable_detection_fallback:
            for extra in range(1, self.max_fallback_upsamples + 1):
                ups_list.append(min(2, base_ups + extra))
        ups_list = list(dict.fromkeys(ups_list))
        
        best = None
        all_last = []
        for scale in scales:
            for ups in ups_list:
                valid_faces, all_faces_rects = detect_once(scale, ups)
                if all_faces_rects:
                    all_last = all_faces_rects
                if valid_faces:
                    best = max(valid_faces, key=lambda x: x[1])[0]
                    self.last_detections = all_faces_rects
                    break
            if best is not None:
                break
        
        if best is None:
            self.last_detections = all_last
            return None
        
        top, right, bottom, left = best
        x = left
        y = top
        w = right - left
        h = bottom - top
        
        current_center = (x + w // 2, y + h // 2)
        if len(self.face_history) > 0:
            last_center = self.face_history[-1]
            distance = np.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
            max_distance = max(frame.shape[1], frame.shape[0]) * 0.3
            if distance <= max_distance:
                smoothed_center = self._smooth_face_position(current_center, w, h)
                x = int(smoothed_center[0] - w // 2)
                y = int(smoothed_center[1] - h // 2)
            else:
                self.face_history.clear()
                self.face_history.append(current_center)
        else:
            self.face_history.append(current_center)
        
        return (x, y, w, h)
    
    def _smooth_face_position(self, position: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
        if len(self.face_history) == 0:
            self.face_history.append(position)
            return position
        
        last_pos = self.face_history[-1]
        alpha = 0.6
        
        smoothed_x = int(alpha * position[0] + (1 - alpha) * last_pos[0])
        smoothed_y = int(alpha * position[1] + (1 - alpha) * last_pos[1])
        
        smoothed = (smoothed_x, smoothed_y)
        self.face_history.append(smoothed)
        
        return smoothed
        
    def calculate_face_center(self, face_rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = face_rect
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        return (face_center_x, face_center_y)
        
    def update_servo_positions(self, face_center: Tuple[int, int]):
        if self.image_center is None:
            return
            
        center_x, center_y = self.image_center
        face_x, face_y = face_center
        
        error_x = face_x - center_x
        error_y = face_y - center_y
        
        if self.image_center:
            frame_width, frame_height = self.image_center[0] * 2, self.image_center[1] * 2
        else:
            frame_width, frame_height = 640, 480
        half_w = frame_width / 2.0 if frame_width > 0 else 1.0
        half_h = frame_height / 2.0 if frame_height > 0 else 1.0
        
        normalized_error_x = error_x / half_w
        normalized_error_y = error_y / half_h
        
        self.last_normalized_error_x = normalized_error_x
        self.last_normalized_error_y = normalized_error_y
        
        pan_output = self.pan_pid.update(normalized_error_x)
        tilt_output = self.tilt_pid.update(normalized_error_y)
        
        pan_sign = -1 if self.rotate_180 else 1
        tilt_sign = -1 if self.rotate_180 else 1
        if self.invert_pan:
            pan_sign *= -1
        if self.invert_tilt:
            tilt_sign *= -1
        pan_output *= pan_sign
        tilt_output *= tilt_sign
        
        self.last_pan_output = pan_output
        self.last_tilt_output = tilt_output
        
        pan_delta = np.clip(pan_output, -self.max_step_size, self.max_step_size)
        tilt_delta = np.clip(tilt_output, -self.max_step_size, self.max_step_size)
        
        self.pan_position = np.clip(self.pan_position + pan_delta, -1.0, 1.0)
        self.tilt_position = np.clip(self.tilt_position + tilt_delta, -1.0, 1.0)
        
        if not self.mock_servos:
            success_pan = self.pan_servo.set_position_normalized(self.pan_position)
            success_tilt = self.tilt_servo.set_position_normalized(self.tilt_position)
            
            if not success_pan:
                logger.warning(f"Failed to set pan servo position: {self.pan_position:.3f}")
            if not success_tilt:
                logger.warning(f"Failed to set tilt servo position: {self.tilt_position:.3f}")
        
    def run(self):
        if not self.connect():
            logger.error("Failed to connect robot")
            return
            
        try:
            logger.info("Starting face tracking. Press 'q' to quit.")
            
            fps_counter = 0
            fps_start_time = time.time()
            fps = 0
            frame_skip_counter = 0
            
            while True:
                if self.cap is None or not self.cap.isOpened():
                    logger.error("Camera not available")
                    break
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    break
                
                frame_skip_counter += 1
                if frame_skip_counter < self.frame_skip_interval:
                    continue
                frame_skip_counter = 0
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                frame = self.undistort_image(frame)
                frame = self.correct_color(frame)
                    
                face_rect = self.detect_face(frame)
                
                if face_rect is not None:
                    x, y, w, h = face_rect
                    if self.show_all_detections and self.last_detections:
                        for dx, dy, dw, dh in self.last_detections:
                            cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (0, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(
                        frame,
                        "Face Detected",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    face_center = self.calculate_face_center(face_rect)
                    cv2.circle(frame, face_center, 8, (0, 0, 255), -1)
                    cv2.circle(frame, face_center, 12, (0, 0, 255), 2)
                    
                    if self.image_center is not None:
                        center_x, center_y = self.image_center
                        error_x = face_center[0] - center_x
                        error_y = face_center[1] - center_y
                        
                        cv2.line(
                            frame,
                            self.image_center,
                            face_center,
                            (255, 255, 0),
                            2
                        )
                        
                        cv2.putText(
                            frame,
                            f"Error: X={error_x} Y={error_y}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2
                        )
                    
                    self.update_servo_positions(face_center)
                    
                    cv2.putText(
                        frame,
                        f"Pan: {self.pan_position:.3f}, Tilt: {self.tilt_position:.3f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Norm Err: X={self.last_normalized_error_x:.3f} Y={self.last_normalized_error_y:.3f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"PID Out: Pan={self.last_pan_output:.3f} Tilt={self.last_tilt_output:.3f}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )
                    if self.show_all_detections:
                        cv2.putText(
                            frame,
                            f"Faces: {len(self.last_detections)}",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 200, 255),
                            2
                        )
                else:
                    self.face_history.clear()
                    self.pan_pid.reset()
                    self.tilt_pid.reset()
                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                
                if self.image_center is not None:
                    cv2.circle(frame, self.image_center, 15, (255, 0, 0), 2)
                    cv2.circle(frame, self.image_center, 3, (255, 0, 0), -1)
                    cv2.putText(
                        frame,
                        "Target",
                        (self.image_center[0] + 20, self.image_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )
                
                cv2.putText(
                    frame,
                    f"FPS: {fps}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                
                help_text = "Press 'q' to quit | 'p' to toggle preprocessed view"
                cv2.putText(
                    frame,
                    help_text,
                    (frame.shape[1] - 350, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )
                
                cv2.imshow("Face Tracker - Debug View", frame)
                
                if self.show_preprocessed:
                    preprocessed = self.preprocess_image(frame)
                    cv2.imshow("Preprocessed Image", preprocessed)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.show_preprocessed = not self.show_preprocessed
                    logger.info(f"Preprocessed view: {'ON' if self.show_preprocessed else 'OFF'}")
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.disconnect()
            logger.info("Robot disconnected")

