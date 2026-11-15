"""
DJI RS4 RavenEye — WiFi stream receiver.

Real protocol (from pcap analysis)
------------------------------------
The Ronin app sends 48-byte UDP keepalive packets (magic 0x30 0x80) to
192.168.2.1 on ports 9004 and 9005, from source port 10002, every 2 s.
This is NOT the DUML (0x55) protocol.

Usage
-----
# 0. Find your WiFi interface name:
networksetup -listallhardwareports | grep -A1 "Wi-Fi"

# 1. Find your local IP when connected to RavenEye WiFi:
ipconfig getifaddr en0

# 2. Connect Mac to the RavenEye_XXXXXX WiFi access point.

# 3. Diagnostic — replay keepalives and print raw RavenEye responses:
python3 dji_rs4/replayer.py --local-ip 192.168.2.187

# 4. Once the response format is confirmed, run the full stream + display:
python3 dji_rs4/main.py --stream --local-ip 192.168.2.187

# 5. Passive sniffer (sudo required) — capture all device traffic:
sudo venv_11/bin/python3 dji_rs4/main.py --sniff --iface en0
"""

import argparse
import queue
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="DJI RS4 RavenEye — WiFi H.264 stream receiver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stream", action="store_true",
        help="Connect to RavenEye, receive H.264 and display live frames.",
    )
    mode.add_argument(
        "--sniff", action="store_true",
        help=(
            "Passive UDP sniffer — prints every packet from the device. "
            "Requires sudo (raw-socket access). "
            "Run together with --stream in a second terminal for diagnosis."
        ),
    )

    p.add_argument("--host",       default="192.168.2.1", help="RavenEye IP address.")
    p.add_argument("--local-ip",   default="", dest="local_ip",
                   help=(
                       "Local IP to bind the UDP socket to (e.g. 192.168.2.187). "
                       "IMPORTANT: if a VPN is active (even 'disconnected') it may "
                       "intercept traffic to 192.168.2.x via the tunnel interface. "
                       "Setting this to your IP on the RavenEye network forces packets "
                       "through the correct WiFi interface and bypasses the VPN. "
                       "Find your IP with: ipconfig getifaddr en0"
                   ))
    p.add_argument("--iface",      default="en0",
                   help=(
                       "Network interface for --sniff mode. "
                       "MacBooks without Ethernet: en0. "
                       "Mac Mini/Studio with Ethernet: WiFi is usually en1. "
                       "Verify with: networksetup -listallhardwareports | grep -A1 Wi-Fi"
                   ))
    return p


def run_stream(args: argparse.Namespace) -> None:
    from stream.receiver import Receiver
    from display.viewer import Viewer

    video_queue: queue.Queue = queue.Queue(maxsize=256)

    receiver = Receiver(
        host=args.host,
        local_ip=args.local_ip,
        video_queue=video_queue,
    )
    viewer = Viewer(video_queue=video_queue)

    try:
        receiver.start()
        viewer.run()          # blocks on main thread (required by macOS OpenCV)
    finally:
        receiver.stop()


def run_sniff(args: argparse.Namespace) -> None:
    # Lazy import — scapy initialisation is slow and noisy; skip it in --stream mode.
    from sniffer.capture import start_sniff

    sniffer = start_sniff(iface=args.iface)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sniffer.stop()
        print("\n[Sniffer] Stopped.")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.stream:
        run_stream(args)
    else:
        run_sniff(args)


if __name__ == "__main__":
    main()
