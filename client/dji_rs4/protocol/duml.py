"""
DJI DUML (DJI Universal Message Layer) packet builder and parser.

Packet layout
-------------
[0x55] [LEN_L] [LEN_H+VER] [CRC8] [SRC] [DST] [SEQ_L] [SEQ_H]
[CMD_TYPE] [CMDSET] [CMDID] [...PAYLOAD...] [CRC16_L] [CRC16_H]

  LEN_L / LEN_H+VER : bits 0-9 = total packet length,
                       bits 10-15 = version (always 0x01).
  CRC8               : crc8(bytes[0:3])  - poly 0xD5, init 0x77.
  SRC / DST          : device-type IDs.
                       PC/mobile client = 0x2a, RavenEye/gimbal = 0x28.
  CMD_TYPE           : bit 7 = is_ack, bit 6 = need_ack,
                       bits 0-2 = encryption (0 = none).
                       NO separate ENC byte.
  CRC16              : crc16(bytes[0:len-2]) - poly 0x8408, init 0x3692.

Minimum packet size: 13 bytes (11 header + 0 payload + 2 CRC16).
"""

from collections import namedtuple
from .crc import crc8, crc16

# Sentinel for a parse failure.
DumlPacket = namedtuple("DumlPacket", ["cmdset", "cmdid", "payload", "is_ack"])

_MAGIC = 0x55
_VERSION = 0x01
_MIN_LEN = 13  # header (11) + CRC16 (2)

# Common device-type IDs.
SRC_PC = 0x2A
DST_DEVICE = 0x28

# CMD_TYPE convenience values.
CMD_REQUEST = 0x40   # need_ack=1, is_ack=0, enc=0
CMD_PUSH    = 0x00   # need_ack=0, is_ack=0, enc=0


def build(
    src: int,
    dst: int,
    seq: int,
    cmd_type: int,
    cmdset: int,
    cmdid: int,
    payload: bytes = b"",
) -> bytes:
    """
    Assemble a DUML packet and return it as *bytes*.

    Parameters
    ----------
    src, dst   : device-type IDs (e.g. SRC_PC, DST_DEVICE).
    seq        : 16-bit sequence number (wraps at 0xFFFF).
    cmd_type   : CMD_REQUEST / CMD_PUSH, or any byte value.
    cmdset     : command-set byte.
    cmdid      : command-ID byte.
    payload    : optional payload bytes (default empty).
    """
    total_len = _MIN_LEN + len(payload)

    len_l   = total_len & 0xFF
    len_h_v = ((_VERSION & 0x3F) << 2) | ((total_len >> 8) & 0x03)

    header3 = bytes([_MAGIC, len_l, len_h_v])
    crc_head = crc8(header3)

    seq_l = seq & 0xFF
    seq_h = (seq >> 8) & 0xFF

    body = (
        header3
        + bytes([crc_head, src & 0xFF, dst & 0xFF, seq_l, seq_h,
                 cmd_type & 0xFF, cmdset & 0xFF, cmdid & 0xFF])
        + payload
    )

    crc_tail = crc16(body)
    return body + bytes([crc_tail & 0xFF, (crc_tail >> 8) & 0xFF])


def parse(data: bytes):
    """
    Parse a raw DUML UDP payload.

    Returns a :class:`DumlPacket` on success, or ``None`` if the data is
    too short, has a wrong magic byte, or fails a CRC check.
    """
    if len(data) < _MIN_LEN:
        return None

    if data[0] != _MAGIC:
        return None

    total_len = data[1] | ((data[2] & 0x03) << 8)

    if total_len < _MIN_LEN or total_len > len(data):
        return None

    # Verify header CRC-8 (over bytes 0, 1, 2).
    if crc8(data[:3]) != data[3]:
        return None

    # Verify full-packet CRC-16 (over bytes 0 .. total_len-3).
    expected_crc = data[total_len - 2] | (data[total_len - 1] << 8)
    if crc16(data[:total_len - 2]) != expected_crc:
        return None

    cmd_type = data[8]
    is_ack   = bool(cmd_type & 0x80)
    cmdset   = data[9]
    cmdid    = data[10]
    payload  = bytes(data[11 : total_len - 2])

    return DumlPacket(cmdset=cmdset, cmdid=cmdid, payload=payload, is_ack=is_ack)
