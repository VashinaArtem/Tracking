"""
Passive UDP sniffer for diagnosing the RavenEye stream.

PURPOSE
-------
When --stream mode does not produce video (e.g. the video-start DUML command
bytes differ for this firmware version), run this sniffer in a second terminal
**while --stream is running in the first**.  It captures every UDP packet that
the device sends TO the PC and prints the source/destination ports and payload
size.  This reveals the actual stream port and approximate packet cadence so
you can adjust _CMD_START_VIDEO in receiver.py.

REQUIREMENTS
------------
* Must be run with sudo (Scapy needs raw-socket / BPF access on macOS):
      sudo venv/bin/python main.py --sniff --iface en0

* The WiFi interface name (en0 / en1 / …) must match the interface your Mac
  uses for WiFi.  Verify with:
      networksetup -listallhardwareports | grep -A1 "Wi-Fi"

NOTE
----
This does NOT capture phone↔RavenEye traffic.  Standard promiscuous mode on
a WiFi AP network only sees packets addressed to the PC's own MAC, plus
broadcast/multicast.  Capturing phone traffic would require monitor mode.
"""

from scapy.all import AsyncSniffer, IP, UDP  # type: ignore


def _on_packet(pkt) -> None:
    if IP not in pkt or UDP not in pkt:
        return
    ip   = pkt[IP]
    udp  = pkt[UDP]
    size = len(bytes(udp.payload))
    print(
        f"[Sniffer] {ip.src}:{udp.sport} → {ip.dst}:{udp.dport}  "
        f"payload={size} B"
    )


def start_sniff(iface: str = "en0") -> AsyncSniffer:
    """
    Start an asynchronous sniffer on *iface*.

    Returns the running :class:`AsyncSniffer` object.  Call ``.stop()`` on
    it to finish sniffing.
    """
    sniffer = AsyncSniffer(
        iface=iface,
        filter="src host 192.168.2.1 and udp",
        prn=_on_packet,
        store=False,
    )
    sniffer.start()
    print(
        f"[Sniffer] Listening on {iface} for UDP from 192.168.2.1 — "
        "press Ctrl+C to stop."
    )
    return sniffer
