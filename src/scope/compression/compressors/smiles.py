# -*- coding: utf-8 -*-
"""
    ScOPE
    Compression Functions - Improved
    Jesus Alan Heernandez Galvan
"""
import smilez

from .base_ import BaseCompressor


class SmilezCompressor(BaseCompressor):
    """Smilez compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 1):
        super().__init__(
            compressor_name="smilez",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return smilez.compress(sequence)
