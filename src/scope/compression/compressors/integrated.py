# -*- coding: utf-8 -*-
"""
    ScOPE
    Compression Functions
    Jesus Alan Heernandez Galvan
"""
import gzip
import bz2
import zlib
import smaz
import smilez

from .base_ import BaseCompressor


class SmazCompressor(BaseCompressor):
    def __init__(self, compression_level: int = 1):
        super().__init__(
            compressor_name="smaz",
            compression_level=compression_level,
        )
    
    def compress(self, sequence: bytes) -> bytes:
        sequence_: str = sequence.decode("utf-8")
        return smaz.compress(sequence_)

class Bz2Compressor(BaseCompressor):
    """BZ2 compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="bz2",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using BZ2 algorithm, removing header for size optimization."""
        # return bz2.compress(sequence, compresslevel=self._compression_level)[15:]
        return bz2.compress(sequence, compresslevel=self._compression_level)
        
        
class ZlibCompressor(BaseCompressor):
    """ZLIB compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using ZLIB algorithm with raw deflate (no headers)."""
        # return zlib.compress(sequence, level=self._compression_level, wbits=-15)
        return zlib.compress(sequence, level=self._compression_level)


class GZipCompressor(BaseCompressor):
    """ZLIB compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using ZLIB algorithm with raw deflate (no headers)."""
        return gzip.compress(sequence, compresslevel=self._compression_level)
        # return zlib.compress(sequence, level=self._compression_level)


class SmilezCompressor(BaseCompressor):
    """Smilez compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 1):
        super().__init__(
            compressor_name="smilez",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return smilez.compress(sequence)
