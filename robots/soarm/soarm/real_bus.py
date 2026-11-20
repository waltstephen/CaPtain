"""Feetech STS3215 bus of the real SO-101 (IDs 1-6, 1 Mbps): sync read of positions, sync write of goals."""
from __future__ import annotations
import os

import glob
import sys

try:
    import scservo_sdk as scs
except ImportError:   # the env server runs in venv-srv; the SDK (pure Python + pyserial) lives in venv-arm
    sys.path += glob.glob(os.path.expandvars("$SOARM_ROOT/venv-arm/lib/python3*/site-packages"))
    import scservo_sdk as scs

A_TORQUE, A_ACC, A_GOAL, A_SPEED, A_TLIM = 40, 41, 42, 46, 48
A_POS, A_LOAD, A_TEMP, A_STATUS = 56, 60, 63, 65


class Bus:
    def __init__(self, port: str, ids, baud: int = 1_000_000) -> None:
        self.ids = [int(i) for i in ids]
        self.port = scs.PortHandler(port)
        self.pk = scs.PacketHandler(0)
        if not self.port.openPort() or not self.port.setBaudRate(baud):
            raise IOError(f"cannot open {port}")
        self._read = scs.GroupSyncRead(self.port, self.pk, A_POS, 2)
        for i in self.ids:
            self._read.addParam(i)
        self._write = scs.GroupSyncWrite(self.port, self.pk, A_GOAL, 2)

    def _retry(self, fn, *args):
        for _ in range(3):
            out = fn(self.port, *args)
            if out[-2 if len(out) == 3 else 0] == scs.COMM_SUCCESS:
                return out
        raise IOError(f"servo bus: {fn.__name__}{args}: {self.pk.getTxRxResult(out[-2 if len(out) == 3 else 0])}")

    def r1(self, i, a):
        return self._retry(self.pk.read1ByteTxRx, i, a)[0]

    def r2(self, i, a):
        return self._retry(self.pk.read2ByteTxRx, i, a)[0]

    def w1(self, i, a, v):
        self._retry(self.pk.write1ByteTxRx, i, a, int(v))

    def w2(self, i, a, v):
        self._retry(self.pk.write2ByteTxRx, i, a, int(v))

    def positions(self) -> dict[int, int]:
        for _ in range(3):
            if self._read.txRxPacket() == scs.COMM_SUCCESS and all(self._read.isAvailable(i, A_POS, 2) for i in self.ids):
                return {i: int(self._read.getData(i, A_POS, 2)) for i in self.ids}
        return {i: int(self.r2(i, A_POS)) for i in self.ids}   # sync read failed three times: one by one

    def goals(self, goals: dict[int, int]) -> None:
        self._write.clearParam()
        for i, v in goals.items():
            v = max(0, min(4095, int(v)))
            self._write.addParam(int(i), [v & 0xFF, (v >> 8) & 0xFF])
        self._write.txPacket()

    def close(self) -> None:
        self.port.closePort()
