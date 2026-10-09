"""Stable independent experiment streams; no mutable global generators."""
import hashlib


def stream_seed(seed, name, index=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}:{index}'.encode()).digest()[:8], 'big') % (2**63)
