#!/usr/bin/env python3
"""
Check available USB-2 ports and connected USB devices on macOS.
Uses system_profiler and ioreg — no third-party dependencies required.
"""

import json
import subprocess
import sys


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def divider(title: str = "", width: int = 60) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


# ── USB device tree via system_profiler ──────────────────────────────────────

def show_usb_device_tree() -> None:
    """Parse system_profiler JSON output and display USB-2 devices."""
    divider("USB Device Tree (system_profiler)")

    raw = run(["system_profiler", "SPUSBDataType", "-json"])
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("  Could not parse system_profiler output.")
        return

    usb_items = data.get("SPUSBDataType", [])
    if not usb_items:
        print("  No USB data returned.")
        return

    usb2_count = 0
    all_count  = 0

    def walk(devices: list, indent: int = 0) -> None:
        nonlocal usb2_count, all_count
        prefix = "  " * indent
        for dev in devices:
            name        = dev.get("_name", "Unknown Device")
            speed       = dev.get("device_speed", "")
            vendor_id   = dev.get("vendor_id", "")
            product_id  = dev.get("product_id", "")
            serial      = dev.get("serial_num", "")
            manufacturer= dev.get("manufacturer", "")
            location    = dev.get("location_id", "")
            bcd_usb     = dev.get("bcd_usb", "")

            is_usb2 = "480" in speed or "12" in speed or "1.5" in speed or "2.0" in bcd_usb

            all_count += 1
            if is_usb2:
                usb2_count += 1

            speed_label = f"[USB 2.0 – {speed}]" if is_usb2 else f"[{speed}]"
            flag = " ◀ USB 2" if is_usb2 else ""

            print(f"{prefix}• {name}{flag}")
            if manufacturer:
                print(f"{prefix}  Manufacturer : {manufacturer}")
            if vendor_id or product_id:
                print(f"{prefix}  VID/PID      : {vendor_id} / {product_id}")
            if serial:
                print(f"{prefix}  Serial       : {serial}")
            if speed:
                print(f"{prefix}  Speed        : {speed_label}")
            if location:
                print(f"{prefix}  Location ID  : {location}")

            # Recurse into hubs
            children = dev.get("_items", [])
            if children:
                walk(children, indent + 1)

    walk(usb_items)
    print()
    divider()
    print(f"  Total USB devices found : {all_count}")
    print(f"  USB 2.0 devices         : {usb2_count}")
    print(f"  Non-USB-2 devices       : {all_count - usb2_count}")


# ── USB-2 hub port availability via ioreg ────────────────────────────────────

def show_usb2_hub_ports() -> None:
    """
    Query IORegistry for USB 2.0 hub port count vs. used ports.
    Gives a rough picture of how many physical USB-2 slots exist.
    """
    divider("USB 2.0 Hub Port Availability (ioreg)")

    raw = run(["ioreg", "-r", "-c", "IOUSBHub", "-l", "-d", "2"])
    lines = raw.splitlines()

    hub_name    = None
    port_count  = None
    child_count = 0
    hubs_found  = 0

    def flush_hub() -> None:
        nonlocal hub_name, port_count, child_count, hubs_found
        if hub_name is None:
            return
        hubs_found += 1
        used = child_count
        total = port_count if port_count is not None else "?"
        free  = (total - used) if isinstance(total, int) else "?"
        status = "available" if (isinstance(free, int) and free > 0) else "full / unknown"
        print(f"\n  Hub  : {hub_name}")
        print(f"  Ports: {total} total  |  ~{used} used  |  ~{free} free  [{status}]")
        hub_name   = None
        port_count = None
        child_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("+-o ") and "IOUSBHub" in stripped:
            flush_hub()
            hub_name    = stripped.split("+-o ", 1)[-1].split(" <")[0]
            child_count = 0
            port_count  = None
        elif '"PortCount"' in stripped or '"kUSBHubPortCount"' in stripped:
            try:
                port_count = int(stripped.split("=")[-1].strip())
            except ValueError:
                pass
        elif "+-o Port" in stripped:
            child_count += 1

    flush_hub()

    if hubs_found == 0:
        print("  No USB hubs detected via ioreg (normal on Apple Silicon Macs).")
        print("  Use the device tree above to see what is plugged in.")


# ── /dev scan for USB serial nodes (bonus) ───────────────────────────────────

def show_usb_serial_nodes() -> None:
    """Show /dev entries that look like USB serial adapters."""
    divider("USB Serial Nodes in /dev (adapters / Arduino / etc.)")
    raw = run(["ls", "/dev/"])
    nodes = [
        f"/dev/{d}" for d in raw.splitlines()
        if "usb" in d.lower() or d.startswith("tty.") or d.startswith("cu.")
    ]
    if nodes:
        for n in sorted(nodes):
            print(f"  {n}")
    else:
        print("  None found.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        USB-2 Port & Device Checker  (macOS)         ║")
    print("╚══════════════════════════════════════════════════════╝")

    show_usb_device_tree()
    show_usb2_hub_ports()
    show_usb_serial_nodes()

    print("\nDone.\n")
