import logging
from typing import Optional

try:
    import scservo_sdk as scs
except ImportError:
    raise ImportError(
        "scservo_sdk is required. Install with: pip install scservo-sdk"
    )

logger = logging.getLogger(__name__)

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

CONTROL_TABLE = {
    "Torque_Enable": (40, 1),
    "Goal_Position": (42, 2),
    "Present_Position": (56, 2),
    "Operating_Mode": (33, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
}

POSITION_MODE = 0


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    """
    HACK: This patches the PortHandler behavior to set the correct packet timeouts.

    It fixes https://gitee.com/ftservo/SCServoSDK/issues/IBY2S6
    The bug is fixed on the official Feetech SDK repo (https://gitee.com/ftservo/FTServo_Python)
    but because that version is not published on PyPI, we rely on the (unofficial) on that is, which needs
    patching.
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


class FeetechServo:
    def __init__(
        self,
        servo_id: int,
        port: str,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
        baudrate: int = DEFAULT_BAUDRATE,
        min_position: Optional[int] = None,
        max_position: Optional[int] = None,
        center_position: Optional[int] = None,
        port_handler: Optional["scs.PortHandler"] = None,
    ):
        self.servo_id = servo_id
        self.port = port
        self.protocol_version = protocol_version
        self.baudrate = baudrate
        
        # Allow sharing a single PortHandler (and underlying OS handle) between multiple servos
        self._owns_port_handler = port_handler is None
        self.port_handler = port_handler or scs.PortHandler(port)
        # self.port_handler.openPort()
        self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
            self.port_handler, scs.PortHandler
        )
        self.packet_handler = scs.PacketHandler(protocol_version)
        self._comm_success = scs.COMM_SUCCESS
        self._no_error = 0x00
        
        self._connected = False
        
        self._min_position = min_position if min_position is not None else 0
        self._max_position = max_position if max_position is not None else 4095
        self._center_position = center_position if center_position is not None else 2048
        
    def connect(self) -> bool:
        if self._connected:
            return True
            
        # Open the serial port only if not already open (shared handler may already be open)
        if not getattr(self.port_handler, "is_open", False):
            if not self.port_handler.openPort():
                logger.error(f"Failed to open port {self.port}")
                return False
            
        # Avoid resetting the port if it's already configured at the desired baudrate.
        try:
            current_baud = self.port_handler.getBaudRate()
        except Exception:
            current_baud = None
        if current_baud != self.baudrate:
            if not self.port_handler.setBaudRate(self.baudrate):
                logger.error(f"Failed to set baudrate {self.baudrate}")
                return False
            
        dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(self.port_handler, self.servo_id)
        if not self._check_result(dxl_comm_result, dxl_error, "ping"):
            # self.port_handler.closePort()
            return False
        if dxl_model_number == 0:
            logger.warning(f"Ping succeeded but model number is 0 for servo {self.servo_id}")
            
        self._write("Operating_Mode", POSITION_MODE)
        self._write("Torque_Enable", 1)
        
        self._connected = True
        logger.info(f"Servo {self.servo_id} connected successfully")
        return True
        
    def disconnect(self):
        if not self._connected:
            return
            
        self.disable_torque()
        # Close only if we created the handler ourselves. If shared, let the owner manage lifecycle.
        # if self._owns_port_handler and self.port_handler.is_open:
        #     self.port_handler.closePort()
        self._connected = False
        logger.info(f"Servo {self.servo_id} disconnected")
        
    def _check_result(self, comm_result: int, error: int, operation: str) -> bool:
        if comm_result != self._comm_success:
            logger.error(
                f"{operation} failed on servo {self.servo_id}: "
                f"{self.packet_handler.getTxRxResult(comm_result)}"
            )
            return False
            
        if error != 0:
            logger.error(
                f"{operation} error on servo {self.servo_id}: "
                f"{self.packet_handler.getRxPacketError(error)}"
            )
            return False
            
        return True
        
    def _read(self, register_name: str) -> Optional[int]:
        addr, length = CONTROL_TABLE[register_name]
        return self._read_raw(addr, length, register_name)
            
    def _write(self, register_name: str, value: int) -> bool:
        addr, length = CONTROL_TABLE[register_name]
        return self._write_raw(addr, length, value, register_name)
        
    def _write_raw(self, addr: int, length: int, value: int, name: str = "register") -> bool:
        if length == 1:
            param_data = [value]
        elif length == 2:
            param_data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
        elif length == 4:
            param_data = [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
            ]
        else:
            logger.error(f"Unsupported write length {length} for {name}")
            return False
            
        dxl_comm_result, dxl_error = self.packet_handler.writeTxRx(
            self.port_handler, self.servo_id, addr, length, param_data
        )
        
        return self._check_result(dxl_comm_result, dxl_error, f"write {name}")
        
    def _read_raw(self, addr: int, length: int, name: str = "register") -> Optional[int]:
        if length == 1:
            read_fn = self.packet_handler.read1ByteTxRx
        elif length == 2:
            read_fn = self.packet_handler.read2ByteTxRx
        elif length == 4:
            read_fn = self.packet_handler.read4ByteTxRx
        else:
            logger.error(f"Unsupported read length {length} for {name}")
            return None
        
        value, dxl_comm_result, dxl_error = read_fn(self.port_handler, self.servo_id, addr)
        
        if not self._check_result(dxl_comm_result, dxl_error, f"read {name}"):
            return None
            
        return value
        
    def enable_torque(self) -> bool:
        return self._write("Torque_Enable", 1)
        
    def disable_torque(self) -> bool:
        return self._write("Torque_Enable", 0)
        
    def set_position(self, position: int) -> bool:
        clamped = max(self._min_position, min(self._max_position, position))
        if clamped != position:
            logger.debug(f"Position {position} clamped to {clamped} for servo {self.servo_id}")
        return self._write("Goal_Position", clamped)
        
    def get_position(self) -> Optional[int]:
        return self._read("Present_Position")
        
    def set_position_normalized(self, normalized_position: float) -> bool:
        normalized_position = max(-1.0, min(1.0, normalized_position))
        
        if abs(normalized_position) < 1e-9:
            position = self._center_position
        elif normalized_position < 0.0:
            range_to_min = self._center_position - self._min_position
            if range_to_min > 0:
                position = int(self._center_position + normalized_position * range_to_min)
            else:
                position = self._center_position
        else:
            range_to_max = self._max_position - self._center_position
            if range_to_max > 0:
                position = int(self._center_position + normalized_position * range_to_max)
            else:
                position = self._center_position
            
        return self.set_position(position)
        
    def get_position_normalized(self) -> Optional[float]:
        raw = self.get_position()
        if raw is None:
            return None
            
        if raw <= self._center_position:
            range_to_min = self._center_position - self._min_position
            if range_to_min > 0:
                normalized = float(raw - self._center_position) / float(range_to_min)
            else:
                normalized = 0.0
        else:
            range_to_max = self._max_position - self._center_position
            if range_to_max > 0:
                normalized = float(raw - self._center_position) / float(range_to_max)
            else:
                normalized = 0.0
            
        return max(-1.0, min(1.0, normalized))
        
    def set_calibration(self, min_position: int, max_position: int, center_position: int):
        if min_position >= max_position:
            raise ValueError(f"min_position ({min_position}) must be less than max_position ({max_position})")
        if not (min_position <= center_position <= max_position):
            raise ValueError(
                f"center_position ({center_position}) must be between "
                f"min ({min_position}) and max ({max_position})"
            )
            
        self._min_position = min_position
        self._max_position = max_position
        self._center_position = center_position
        logger.info(
            f"Servo {self.servo_id} calibration updated: "
            f"min={min_position}, max={max_position}, center={center_position}"
        )
        
    def get_min_position(self) -> int:
        return self._min_position
        
    def get_max_position(self) -> int:
        return self._max_position
        
    def get_center_position(self) -> int:
        return self._center_position
        
    @property
    def is_connected(self) -> bool:
        return self._connected

