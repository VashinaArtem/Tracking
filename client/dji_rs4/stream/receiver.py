"""
UDP receiver for the DJI RavenEye stream.

Real protocol (discovered from pcap analysis)
----------------------------------------------
The DJI Ronin app does NOT use DUML (0x55) to talk to RavenEye.
It uses a different 48-byte keepalive/announce protocol:

  Offset   Size  Value         Meaning
  ------   ----  -----         -------
     0      1    0x30          magic byte 1
     1      1    0x80          magic byte 2
   2-3      2    random        session ID part A (changes each round)
   4-6      3    0x00 00 00    always zero
   7-9      3    random        session ID part B (changes each round)
  10-47    38    fixed payload stream config (MTU=1472, fps=100, etc.)

The packet is sent 3 times per round, every 2 seconds, alternating
between ports 9004 and 9005 on 192.168.2.1.
Source port is always 10002.

The device's response format is still unknown (the captured pcap only
contains phone→device traffic).  This receiver enqueues all raw UDP
datagrams it receives so the caller can inspect them.

Lifecycle
---------
1. Instantiate Receiver(host, local_ip, video_queue).
2. Call start() — spawns the daemon threads and begins sending keepalives.
3. Raw UDP payloads from the device are put into *video_queue*.
4. Call stop() for a clean shutdown.

NOTE on local_ip
----------------
Always pass --local-ip (e.g. "192.168.2.187") so the socket is bound to
the RavenEye WiFi interface, bypassing any active VPN tunnel that would
otherwise intercept traffic destined for 192.168.2.x.
"""

import os
import queue
import socket
import threading
import time

# Fixed stream-config tail (bytes 10-47).  Values as uint16-LE:
#   100, 100, 1472(MTU), 20, 100, 200, 200, 1472, 20, 100, 20, 100, 1472, 20, 100
#   trailing: 01 01 04 0a 02  (likely codec/version flags)
_FIXED_TAIL: bytes = bytes.fromhex(
    "6400" "6400" "c005" "1400" "0064" "00c8" "00c8"
    "00c0" "0514" "0000" "6400" "1400" "6400" "c005"
    "1400" "0064" "0001" "0104" "0a02"
)

_LOCAL_PORT   = 10002
_REMOTE_PORTS = (9004, 9005)
_COPIES       = 3     # each keepalive is sent 3 times (pcap-observed)
_COPY_GAP     = 0.01  # seconds between copies
_KEEPALIVE_S  = 2.0   # interval between rounds (pcap-observed: exactly 2 s)
_RECV_BUFSIZE = 65535
_SOCK_TIMEOUT = 5.0


def _build_keepalive() -> bytes:
    """Build a fresh 48-byte keepalive packet with random session bytes."""
    return (
        b"\x30\x80"
        + os.urandom(2)    # bytes 2-3
        + b"\x00\x00\x00"  # bytes 4-6
        + os.urandom(3)    # bytes 7-9
        + _FIXED_TAIL
    )


class Receiver:
    """
    Sends RavenEye keepalive packets and collects raw UDP responses.

    Parameters
    ----------
    host        : IP address of the RavenEye (default "192.168.2.1").
    local_ip    : Local IP on the RavenEye WiFi interface (e.g. "192.168.2.187").
                  **Always set this** to avoid VPN routing issues on macOS.
    video_queue : bounded ``queue.Queue`` for incoming UDP payloads.
    """

    def __init__(
        self,
        host: str = "192.168.2.1",
        local_ip: str = "",
        video_queue: queue.Queue = None,
    ) -> None:
        self.host       = host
        self.local_ip   = local_ip
        self.queue      = video_queue if video_queue is not None else queue.Queue(maxsize=256)

        self._running   = False
        self._send_thr  = None
        self._recv_thr  = None
        self._sock      = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open socket, spawn recv + send threads."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(_SOCK_TIMEOUT)
        try:
            self._sock.bind((self.local_ip, _LOCAL_PORT))
        except OSError as exc:
            raise OSError(
                f"Cannot bind {self.local_ip or '*'}:{_LOCAL_PORT} "
                f"(port already in use?): {exc}"
            ) from exc

        self._running = True

        self._recv_thr = threading.Thread(
            target=self._recv_loop, daemon=True, name="raveneye-recv"
        )
        self._send_thr = threading.Thread(
            target=self._send_loop, daemon=True, name="raveneye-send"
        )
        # Start recv first so no incoming packet is missed.
        self._recv_thr.start()
        time.sleep(0.05)
        self._send_thr.start()

    def stop(self) -> None:
        """Signal threads to stop and close the socket."""
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        for thr in (self._recv_thr, self._send_thr):
            if thr is not None:
                thr.join(timeout=3.0)
        self._recv_thr = self._send_thr = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, data: bytes) -> None:
        """Put *data* into the bounded queue; drop oldest on overflow."""
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass

    def _send_loop(self) -> None:
        """Send keepalive packets every 2 s, alternating between 9004 and 9005."""
        port_idx = 0
        no_send_warned = False
        while self._running:
            port = _REMOTE_PORTS[port_idx % 2]
            pkt  = _build_keepalive()
            ts   = time.strftime("%H:%M:%S")
            print(f"[RavenEye] → {self.host}:{port}  {pkt[:10].hex()}...", flush=True)
            for _ in range(_COPIES):
                try:
                    self._sock.sendto(pkt, (self.host, port))
                    no_send_warned = False
                except OSError as exc:
                    if not no_send_warned:
                        print(f"[RavenEye] send error: {exc}")
                        no_send_warned = True
                time.sleep(_COPY_GAP)
            port_idx += 1
            time.sleep(_KEEPALIVE_S)

    def _recv_loop(self) -> None:
        """Receive all UDP datagrams and put them into the video queue."""
        no_data_warned = False
        while self._running:
            try:
                data, addr = self._sock.recvfrom(_RECV_BUFSIZE)
            except socket.timeout:
                if not no_data_warned:
                    print(
                        f"[RavenEye] No response from {self.host} — "
                        "is the device on and WiFi connected?"
                    )
                    no_data_warned = True
                continue
            except OSError:
                break  # socket closed by stop()

            no_data_warned = False
            if len(data) == 0:
                continue

            first_byte = data[0]
            # Log first few bytes to help identify the response protocol.
            print(
                f"[RavenEye] ← {addr[0]}:{addr[1]}  "
                f"len={len(data)}  magic=0x{first_byte:02x}",
                flush=True,
            )
            self._enqueue(data)
