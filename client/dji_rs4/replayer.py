"""
RavenEye packet replayer — diagnostic tool.

Sends the exact 48-byte keepalive packets observed in the captured pcap and
prints every UDP datagram that the device sends back.  Run this while
connected to the RavenEye WiFi to discover:

  1. Which port/address the device uses to stream video back.
  2. The format of the response (0x55 DUML, raw H.264 NALs, RTSP, or other).

Protocol observed in pcap
--------------------------
  - Source port  : 10002 (phone)
  - Dest ports   : 9004 and 9005 on 192.168.2.1, alternating every 2 s
  - Packet size  : 48 bytes, always
  - Magic bytes  : 0x30 0x80  (NOT DJI DUML 0x55)
  - Bytes  2- 3  : 2 random/session bytes (change each round)
  - Bytes  4- 6  : 0x00 0x00 0x00  (always zero)
  - Bytes  7- 9  : 3 random/session bytes (change each round)
  - Bytes 10-47  : fixed stream-config payload (see _FIXED_TAIL below)
  - Each packet is sent 3 times in quick succession (≈10 ms apart).

Usage
-----
  # Connect to RavenEye-XXXXXX WiFi first, then:
  python3 dji_rs4/replayer.py --local-ip 192.168.2.187

  # Specify a different host or run only N rounds:
  python3 dji_rs4/replayer.py --local-ip 192.168.2.187 --host 192.168.2.1 --count 5
"""

import argparse
import os
import socket
import threading
import time

# Fixed tail (bytes 10-47) captured from every packet in the pcap.
# Values decoded as uint16-LE: 100, 100, 1472(MTU), 20, 100, 200, 200, ...
_FIXED_TAIL: bytes = bytes.fromhex(
    "6400" "6400" "c005" "1400" "0064" "00c8" "00c8"
    "00c0" "0514" "0000" "6400" "1400" "6400" "c005"
    "1400" "0064" "0001" "0104" "0a02"
)  # 38 bytes → total packet = 10 + 38 = 48

_LOCAL_PORT   = 10002
_REMOTE_HOST  = "192.168.2.1"
_REMOTE_PORTS = (9004, 9005)
_INTERVAL     = 2.0   # seconds between rounds (observed: exactly 2.0 s)
_COPIES       = 3     # redundant copies per round (observed in pcap)
_COPY_GAP     = 0.01  # seconds between copies
_RECV_BUF     = 65535


def _build_packet() -> bytes:
    """Build a 48-byte keepalive packet with fresh random session bytes."""
    return (
        b"\x30\x80"       # magic
        + os.urandom(2)   # bytes 2-3 : session id A
        + b"\x00\x00\x00" # bytes 4-6 : always zero
        + os.urandom(3)   # bytes 7-9 : session id B
        + _FIXED_TAIL     # bytes 10-47
    )


def _hex_dump(data: bytes, indent: str = "  ") -> None:
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        h = " ".join(f"{b:02x}" for b in chunk)
        a = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"{indent}[{i:4d}] {h:<48}  {a}")


def _recv_loop(sock: socket.socket) -> None:
    """Print all incoming UDP datagrams; runs in a daemon thread."""
    while True:
        try:
            data, addr = sock.recvfrom(_RECV_BUF)
        except socket.timeout:
            continue
        except OSError:
            break

        ts = time.strftime("%H:%M:%S")
        first = data[0] if data else 0
        magic = "DUML(0x55)" if first == 0x55 else f"0x{first:02x}"
        print(
            f"\n[{ts}] ← {addr[0]}:{addr[1]}  len={len(data)}  "
            f"magic={magic}",
            flush=True,
        )
        _hex_dump(data[:64])  # show first 64 bytes; enough to identify format
        if len(data) > 64:
            print(f"  ... ({len(data) - 64} more bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay RavenEye keepalive packets and print responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--local-ip", default="", metavar="IP",
        help="Local IP on the RavenEye WiFi (e.g. 192.168.2.187). "
             "Leave empty to let the OS choose (may route via VPN!).",
    )
    ap.add_argument("--host", default=_REMOTE_HOST, help="RavenEye IP address")
    ap.add_argument(
        "--count", type=int, default=0, metavar="N",
        help="Stop after N rounds (0 = run forever)",
    )
    ap.add_argument(
        "--interval", type=float, default=_INTERVAL, metavar="SEC",
        help="Seconds between rounds",
    )
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1)  # short timeout so _recv_loop stays responsive
    try:
        sock.bind((args.local_ip, _LOCAL_PORT))
    except OSError as exc:
        print(f"[ERROR] Cannot bind {args.local_ip or '*'}:{_LOCAL_PORT}: {exc}")
        return

    t = threading.Thread(target=_recv_loop, args=(sock,), daemon=True, name="recv")
    t.start()

    bound = f"{args.local_ip or '*'}:{_LOCAL_PORT}"
    print(f"[Replayer] Bound to {bound}")
    print(f"[Replayer] Sending to {args.host}:9004/9005 every {args.interval}s "
          f"({_COPIES} copies each)")
    print(f"[Replayer] Listening for responses on {bound}")
    print(f"[Replayer] Press Ctrl+C to stop.\n")

    port_idx = 0
    round_n  = 0
    try:
        while args.count == 0 or round_n < args.count:
            port = _REMOTE_PORTS[port_idx % 2]
            pkt  = _build_packet()
            ts   = time.strftime("%H:%M:%S")
            print(f"[{ts}] → {args.host}:{port}  {pkt[:10].hex()}...", flush=True)
            for _ in range(_COPIES):
                try:
                    sock.sendto(pkt, (args.host, port))
                except OSError as exc:
                    print(f"[WARN] send error: {exc}")
                time.sleep(_COPY_GAP)

            port_idx += 1
            round_n  += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[Replayer] Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
