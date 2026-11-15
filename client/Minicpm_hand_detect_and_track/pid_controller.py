import time
from typing import Optional


class PIDController:
    def __init__(
        self,
        kp: float = 0.001,
        ki: float = 0.0,
        kd: float = 0.0,
        min_output: float = -1.0,
        max_output: float = 1.0,
        integral_limit: Optional[float] = None,
        deadband: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.integral_limit = integral_limit
        self.deadband = deadband
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        
    def update(self, error: float, dt: Optional[float] = None) -> float:
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            self.last_error = error
            return 0.0
            
        if dt is None:
            dt = current_time - self.last_time
        else:
            self.last_time = current_time
            
        if dt <= 0:
            dt = 0.01
            
        if abs(error) < self.deadband:
            error = 0.0
            
        proportional = self.kp * error
        
        self.integral += error * dt
        
        if self.integral_limit is not None:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
            
        integral_term = self.ki * self.integral
        
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        derivative_term = self.kd * derivative
        
        output = proportional + integral_term + derivative_term
        
        self.last_error = error
        self.last_time = current_time
        
        return max(self.min_output, min(self.max_output, output))
