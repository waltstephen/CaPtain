"""Message codec between the harness (py3.11, numpy 2) and the SimplerEnv server (py3.10, numpy 1).

numpy pickles are not portable across numpy majors, so arrays travel as (dtype, shape, bytes)
tuples inside plain-python pickles. Used by both sides; imports nothing from captain_harness.
"""
from __future__ import annotations

import pickle
import struct

import numpy as np

_ND = "__nd__"


def encode(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (_ND, array.dtype.str, array.shape, array.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(encode(v) for v in value)
    return value


def decode(value):
    if isinstance(value, tuple) and len(value) == 4 and value[0] == _ND:
        return np.frombuffer(value[3], dtype=np.dtype(value[1])).reshape(value[2]).copy()
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(decode(v) for v in value)
    return value


def send(stream, message) -> None:
    payload = pickle.dumps(encode(message), protocol=4)
    stream.write(struct.pack("<Q", len(payload)) + payload)
    stream.flush()


def recv(stream):
    header = stream.read(8)
    if len(header) < 8:
        raise EOFError("simplerenv server closed the pipe")
    (size,) = struct.unpack("<Q", header)
    payload = stream.read(size)
    if len(payload) < size:
        raise EOFError("truncated message")
    return decode(pickle.loads(payload))
