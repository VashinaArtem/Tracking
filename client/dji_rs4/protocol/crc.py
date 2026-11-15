"""
DJI DUML CRC helpers.

CRC-8  : poly 0xD5, init 0x77, computed over header bytes 0-2.
CRC-16 : logical poly 0x1021, init 0x3692, but implemented as LSB-first
         (reflected) using effective table poly 0x8408 and the formula
             crc = (crc >> 8) ^ TABLE[(crc ^ byte) & 0xFF]
         Computed over the entire packet except the final 2 bytes.
"""

# ---------------------------------------------------------------------------
# CRC-8  (poly 0xD5, MSB-first, non-reflected)
# ---------------------------------------------------------------------------

def _build_crc8_table() -> list:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ 0xD5
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC8_TABLE = _build_crc8_table()


def crc8(data: bytes) -> int:
    """Compute DJI CRC-8 with init=0x77 over *data*."""
    crc = 0x77
    for b in data:
        crc = _CRC8_TABLE[(crc ^ b) & 0xFF]
    return crc


# ---------------------------------------------------------------------------
# CRC-16  (poly 0x8408 reflected, init 0x3692)
# ---------------------------------------------------------------------------

def _build_crc16_table() -> list:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
        table.append(crc)
    return table


_CRC16_TABLE = _build_crc16_table()


def crc16(data: bytes) -> int:
    """Compute DJI CRC-16 with init=0x3692 over *data*."""
    crc = 0x3692
    for b in data:
        crc = (crc >> 8) ^ _CRC16_TABLE[(crc ^ b) & 0xFF]
    return crc
