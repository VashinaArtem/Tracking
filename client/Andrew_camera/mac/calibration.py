import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ServoCalibration:
    servo_id: int
    min_position: int
    max_position: int
    center_position: int
    
    def __post_init__(self):
        if self.min_position >= self.max_position:
            raise ValueError(f"min_position ({self.min_position}) must be less than max_position ({self.max_position})")
        if not (self.min_position <= self.center_position <= self.max_position):
            raise ValueError(
                f"center_position ({self.center_position}) must be between "
                f"min ({self.min_position}) and max ({self.max_position})"
            )


class CalibrationManager:
    def __init__(self, calibration_file: str = "calibration.json"):
        self.calibration_file = Path(calibration_file)
        self.calibrations: Dict[str, ServoCalibration] = {}
        
    def load(self) -> bool:
        if not self.calibration_file.exists():
            logger.info(f"Calibration file {self.calibration_file} not found")
            return False
            
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                
            self.calibrations = {}
            for motor_name, calib_data in data.items():
                self.calibrations[motor_name] = ServoCalibration(**calib_data)
                
            logger.info(f"Loaded calibration from {self.calibration_file}")
            logger.info(f"Calibrated motors: {list(self.calibrations.keys())}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
            
    def save(self) -> bool:
        try:
            data = {
                motor_name: asdict(calib)
                for motor_name, calib in self.calibrations.items()
            }
            
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved calibration to {self.calibration_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
            
    def set_calibration(self, motor_name: str, calibration: ServoCalibration):
        self.calibrations[motor_name] = calibration
        
    def get_calibration(self, motor_name: str) -> Optional[ServoCalibration]:
        return self.calibrations.get(motor_name)
        
    def has_calibration(self, motor_name: str) -> bool:
        return motor_name in self.calibrations
        
    def normalize_position(self, motor_name: str, raw_position: int) -> float:
        calib = self.get_calibration(motor_name)
        if calib is None:
            raise ValueError(f"No calibration found for motor '{motor_name}'")
            
        if raw_position <= calib.center_position:
            range_to_min = calib.center_position - calib.min_position
            if range_to_min > 0:
                normalized = float(raw_position - calib.center_position) / float(range_to_min)
            else:
                normalized = 0.0
        else:
            range_to_max = calib.max_position - calib.center_position
            if range_to_max > 0:
                normalized = float(raw_position - calib.center_position) / float(range_to_max)
            else:
                normalized = 0.0
            
        return max(-1.0, min(1.0, normalized))
        
    def denormalize_position(self, motor_name: str, normalized_position: float) -> int:
        calib = self.get_calibration(motor_name)
        if calib is None:
            raise ValueError(f"No calibration found for motor '{motor_name}'")
            
        normalized_position = max(-1.0, min(1.0, normalized_position))
        
        if normalized_position <= 0.0:
            range_to_min = calib.center_position - calib.min_position
            if range_to_min > 0:
                raw_position = int(calib.center_position + normalized_position * range_to_min)
            else:
                raw_position = calib.center_position
        else:
            range_to_max = calib.max_position - calib.center_position
            if range_to_max > 0:
                raw_position = int(calib.center_position + normalized_position * range_to_max)
            else:
                raw_position = calib.center_position
            
        return max(calib.min_position, min(calib.max_position, raw_position))

