#!/usr/bin/env python3

import logging
import argparse
from feetech_servo import FeetechServo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CONTROL_TABLE_STS_SERIES = {
    "Firmware_Major_Version": (0, 1),
    "Firmware_Minor_Version": (1, 1),
    "Model_Number": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Velocity_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    "Goal_Position_2": (71, 2),
    "Moving_Velocity": (80, 1),
    "Moving_Velocity_Threshold": (80, 1),
    "DTs": (81, 1),
    "Velocity_Unit_factor": (82, 1),
    "Hts": (83, 1),
    "Maximum_Velocity_Limit": (84, 1),
    "Maximum_Acceleration": (85, 1),
    "Acceleration_Multiplier": (86, 1),
}


def read_register(servo: FeetechServo, name: str, addr: int, length: int):
    try:
        if not servo.is_connected:
            return None, "Servo not connected"
        
        value = servo._read_raw(addr, length, name)
        if value is None:
            return None, "Read failed"
        return value, None
        
    except Exception as e:
        return None, f"Exception: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Test script to read all control table registers from a Feetech servo')
    parser.add_argument(
        '--port',
        type=str,
        default='/dev/ttyUSB0',
        help='Serial port for servo communication (default: /dev/ttyUSB0)'
    )
    parser.add_argument(
        '--servo-id',
        type=int,
        default=1,
        help='Servo ID to test (default: 1)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format (default: table)'
    )
    
    args = parser.parse_args()
    
    servo = FeetechServo(args.servo_id, args.port)
    
    print(f"\n{'='*80}")
    print(f"Testing Servo ID {args.servo_id} on port {args.port}")
    print(f"{'='*80}\n")
    
    if not servo.connect():
        print(f"❌ Failed to connect to servo {args.servo_id}")
        return
        
    print(f"✓ Connected to servo {args.servo_id}\n")
    
    results = {}
    failed_registers = []
    
    print("Reading control table registers...\n")
    
    for name, (addr, length) in sorted(CONTROL_TABLE_STS_SERIES.items(), key=lambda x: x[1][0]):
        value, error = read_register(servo, name, addr, length)
        
        if error:
            failed_registers.append((name, error))
            results[name] = {"address": addr, "length": length, "value": None, "error": error}
        else:
            results[name] = {"address": addr, "length": length, "value": value, "error": None}
    
    servo.disconnect()
    
    if args.format == 'json':
        import json
        print(json.dumps(results, indent=2))
        
    elif args.format == 'csv':
        print("Register Name,Address,Length,Value,Error")
        for name, data in sorted(results.items(), key=lambda x: x[1]["address"]):
            value_str = str(data["value"]) if data["value"] is not None else "N/A"
            error_str = data["error"] if data["error"] else ""
            print(f"{name},{data['address']},{data['length']},{value_str},{error_str}")
            
    else:
        print("\n" + "="*80)
        print("CONTROL TABLE READOUT")
        print("="*80 + "\n")
        
        for name, data in sorted(results.items(), key=lambda x: x[1]["address"]):
            addr = data["address"]
            length = data["length"]
            value = data["value"]
            error = data["error"]
            
            if error:
                print(f"  {name:45} [0x{addr:02X}, {length}B] ❌ {error}")
            else:
                if length == 2:
                    print(f"  {name:45} [0x{addr:02X}, {length}B] = {value:6d} (0x{value:04X})")
                else:
                    print(f"  {name:45} [0x{addr:02X}, {length}B] = {value:6d} (0x{value:02X})")
        
        print("\n" + "="*80)
        
        if failed_registers:
            print(f"\n⚠️  Failed to read {len(failed_registers)} registers:")
            for name, error in failed_registers:
                print(f"   - {name}: {error}")
        else:
            print("\n✓ Successfully read all registers")
        
        print(f"\nTotal registers read: {len(results)}")
        print(f"Successful: {len(results) - len(failed_registers)}")
        print(f"Failed: {len(failed_registers)}")
        
        print("\n" + "="*80)
        print("KEY REGISTERS SUMMARY")
        print("="*80 + "\n")
        
        key_registers = [
            ("Model_Number", "Model Number"),
            ("Firmware_Major_Version", "Firmware Major"),
            ("Firmware_Minor_Version", "Firmware Minor"),
            ("ID", "Servo ID"),
            ("Present_Position", "Current Position"),
            ("Present_Velocity", "Current Velocity"),
            ("Present_Load", "Current Load"),
            ("Present_Voltage", "Voltage (V)"),
            ("Present_Temperature", "Temperature (°C)"),
            ("Present_Current", "Current"),
            ("Moving", "Is Moving"),
            ("Torque_Enable", "Torque Enabled"),
            ("Operating_Mode", "Operating Mode"),
            ("Min_Position_Limit", "Min Position"),
            ("Max_Position_Limit", "Max Position"),
            ("Goal_Position", "Goal Position"),
        ]
        
        for reg_name, display_name in key_registers:
            if reg_name in results:
                data = results[reg_name]
                if data["value"] is not None:
                    value = data["value"]
                    if reg_name == "Present_Voltage":
                        value_str = f"{value / 10.0:.1f}V"
                    elif reg_name == "Present_Temperature":
                        value_str = f"{value}°C"
                    elif reg_name == "Present_Load":
                        value_str = f"{value} (raw)"
                    elif reg_name == "Present_Velocity":
                        value_str = f"{value} (raw)"
                    elif reg_name == "Moving" or reg_name == "Torque_Enable":
                        value_str = "Yes" if value else "No"
                    elif reg_name == "Operating_Mode":
                        modes = {0: "Position", 1: "Velocity", 2: "PWM", 3: "Step"}
                        value_str = f"{value} ({modes.get(value, 'Unknown')})"
                    else:
                        value_str = str(value)
                    print(f"  {display_name:25} = {value_str}")
                else:
                    print(f"  {display_name:25} = N/A ({data['error']})")


if __name__ == '__main__':
    main()

