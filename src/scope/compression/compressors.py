# -*- coding: utf-8 -*-

import bz2
import lz4
import gzip
import lzma
import zlib
import brotli
from zstandard import ZstdCompressor

from enum import Enum
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional


class CompressorType(Enum):
    BROTLI = "brotli"
    BZ2 = "bz2"
    GZIP = "gzip"
    LZ4 = "lz4"
    # LZMA = "lzma"
    ZLIB = "zlib"
    ZSTD = "zstd"
    

class _BaseCompressor(ABC):

    def __init__(self, compressor_name: str, min_size_threshold: Optional[int] = None, compression_level: int = 9):
        """Initializes the BaseCompressor with the specified compression module.

        Args:
            compressor_name (str): The name of the compression method being used (e.g., 'gzip' or 'bz2').
            min_size_threshold (int, optional): Minimum size for effective compression. Defaults to 50.
            compression_level (int, optional): The level of compression to apply (1-9). Defaults to 9.
        """
        
        if compression_level < 1 or compression_level > 9:
            raise ValueError("Compression level must be betwe`en 1 and 9.")
        
        self._min_size_threshold = min_size_threshold if min_size_threshold else 0
        
        self._compressor_name: str = compressor_name
        self._compression_level: int = compression_level
    
    def _should_pad_sequence(self, sequence: bytes) -> bool:
        return len(sequence) < self._min_size_threshold

    @abstractmethod
    def compress(self, sequence: bytes) -> bytes:
        """
        Compresses the input sequence using the specified compression method.

        Args:
            sequence (Union[str, bytes]): The input data to compress.

        Returns:
            bytes: The compressed data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __repr__(self) -> str:
        return f'(Compressor: {self._compressor_name}, Compression Level: {self._compression_level})'

    
    def __call__(self, sequence: Union[str, bytes]) -> Tuple[Union[str, bytes], bytes, int]:
        """Compresses the input sequence and returns the original, compressed data, and the size of the compressed data.

        Args:
            sequence (Union[str, bytes]): The input data to compress.

        Raises:
            TypeError: If the input sequence is not of type 'str' or 'bytes'.

        Returns:
            Tuple[Union[str, bytes], bytes, int]: A tuple containing the original sequence, the compressed data, and the size of the compressed data.
        """
        if not isinstance(sequence, (bytes, str)):
            raise TypeError("Input sequence must be of type 'str' or 'bytes'.")
        
        if isinstance(sequence, str):
            sequence_encoded = sequence.encode('utf-8')
        
        if isinstance(sequence, bytes):
            sequence_encoded = sequence
        
        original_length = len(sequence_encoded)
        target_length = self._min_size_threshold
        
        if self._should_pad_sequence(sequence_encoded):
            if original_length == 0:
                raise ValueError("Sequence must have at least 1 item")
            
            repetitions = (target_length // original_length) + 1
            sequence_to_compress = sequence_encoded * repetitions
                    
        else:
            sequence_to_compress = sequence_encoded
        
        sequence_compressed = self.compress(sequence_to_compress)

        return sequence, sequence_compressed, len(sequence_compressed)


class Brotli(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="brotli",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        return brotli.compress(sequence, quality=self._compression_level)


class Bz2(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="bz2",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        return bz2.compress(sequence, compresslevel=self._compression_level)


class Gzip(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="gzip",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        return gzip.compress(sequence, compresslevel=self._compression_level, mtime=0)


class Lz4(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="lz4",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        return lz4.frame.compress(
            sequence,
            compression_level=self._compression_level,
        )


class Lzma(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="lzma",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        return lzma.compress(
            sequence,
            # format=lzma.FORMAT_RAW,
            preset=self._compression_level
            # filters=[
            #     {
            #         "id": lzma.FILTER_LZMA2,
            #         "preset": self._compression_level
            #     }
            # ]
        )
        
        
class Zlib(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        return zlib.compress(sequence, level=self._compression_level, wbits=-15)
    
    
class ZStandard(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="zstandard",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )
    
    def compress(self, sequence: bytes) -> bytes:
        compressor = ZstdCompressor(level=self._compression_level,
                                    write_content_size=False,
                                    write_checksum=False,
                                    write_dict_id=False
            )
        return compressor.compress(sequence)


COMPRESSOR_STRATEGIES = {
    CompressorType.BROTLI: Brotli,
    CompressorType.BZ2: Bz2,
    CompressorType.GZIP: Gzip,
    CompressorType.LZ4: Lz4,
    # CompressorType.LZMA: Lzma,
    CompressorType.ZLIB: Zlib,
    CompressorType.ZSTD: ZStandard
}


def get_compressor(
    name: Union[str, CompressorType],
    compression_level: int = 9,
    min_size_threshold: Optional[int] = 0
) -> _BaseCompressor:
    if isinstance(name, str):
        try:
            compressor_enum = CompressorType(name.lower())
        except ValueError:
            allowed = sorted(c.value for c in CompressorType)
            raise ValueError(
                f"'{name}' is not a valid compressor name. "
                f"Expected one of: {', '.join(allowed)}"
            )
    elif isinstance(name, CompressorType):
        compressor_enum = name
    else:
        raise TypeError("Expected 'name' to be str or CompressorType.")
    
    compressor_class = COMPRESSOR_STRATEGIES[compressor_enum]
    return compressor_class(
        compression_level=compression_level,
        min_size_threshold=min_size_threshold
    )
    

if __name__ == '__main__':
    compressor = get_compressor("gzip", compression_level=6, min_size_threshold=100)
    original, compressed, compressed_size = compressor("some string to compress")
    print(compressor)