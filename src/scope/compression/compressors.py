# -*- coding: utf-8 -*-
"""
    ScOPE
    Compression Functions
    Jesus Alan Heernandez Galvan
"""

import bz2
import zlib
from enum import Enum
from zstandard import ZstdCompressor
from abc import ABC, abstractmethod
from typing import Union, Optional
from collections import Counter


class CompressorType(Enum):
    """Enumeration of supported compression algorithms."""
    BZ2 = "bz2"
    ZLIB = "zlib"
    ZSTD = "zstd"
    RLE = "rle"
    HUFFMAN = "huffman"
    LZ77 = "lz77"
    

class BaseCompressor(ABC):
    """Abstract base class for all compression algorithms."""

    def __init__(self, compressor_name: str, compression_level: int = 9, 
                 min_size_threshold: Optional[int] = None, 
                 padding_method: Optional[str] = None):
        """Initialize the base compressor with specified parameters.

        Args:
            compressor_name: Name of the compression method (e.g., 'gzip', 'bz2')
            compression_level: Compression level (1-9, where 9 is maximum)
            min_size_threshold: Minimum size for effective compression
            padding_method: Method for padding small sequences ('zeros' or 'repeat')
        
        Raises:
            ValueError: If compression level is not between 1-9 or invalid padding method
        """
        if not 1 <= compression_level <= 9:
            raise ValueError("Compression level must be between 1 and 9")
        
        if min_size_threshold and min_size_threshold > 0:
            if padding_method and padding_method not in {"zeros", "repeat"}:
                raise ValueError("padding_method must be 'zeros' or 'repeat'")
            padding_method = padding_method or 'zeros'

        self._min_size_threshold = min_size_threshold or 0
        self._padding_method: str = padding_method
        self._compressor_name: str = compressor_name
        self._compression_level: int = compression_level
    
    def _should_pad_sequence(self, sequence: bytes) -> bool:
        """Check if sequence needs padding based on minimum size threshold."""
        return len(sequence) < self._min_size_threshold

    @abstractmethod
    def compress(self, sequence: bytes) -> bytes:
        """Abstract method for compressing input sequence.

        Args:
            sequence: Input data to compress

        Returns:
            Compressed data as bytes
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def __repr__(self) -> str:
        """Return string representation of the compressor."""
        return f'Compressor(name={self._compressor_name}, level={self._compression_level})'

    def __call__(self, sequence: Union[str, bytes]) -> bytes:
        """Compress input sequence and return compressed data.

        Args:
            sequence: Input data to compress (str or bytes)

        Returns:
            Compressed data as bytes
            
        Raises:
            ValueError: If sequence is empty
            TypeError: If input type is invalid
        """
        if len(sequence) == 0:
            raise ValueError(
                f"Empty sequence provided to {self._compressor_name} compressor. "
                "Compression requires non-empty input data."
            )
                
        if not isinstance(sequence, (bytes, str)):
            raise TypeError("Input sequence must be of type 'str' or 'bytes'")
        
        # Convert string to bytes if necessary
        sequence_encoded = sequence.encode('utf-8') if isinstance(sequence, str) else sequence
        
        # Apply padding if needed
        sequence_to_compress = self._apply_padding(sequence_encoded)
        
        return self.compress(sequence_to_compress)

    def _apply_padding(self, sequence: bytes) -> bytes:
        """Apply padding to sequence if it's below minimum threshold."""
        if not self._should_pad_sequence(sequence):
            return sequence
        
        original_length = len(sequence)
        if original_length == 0:
            raise ValueError("Sequence must have at least 1 byte")
        
        padding_needed = self._min_size_threshold - original_length
        
        if self._padding_method == "zeros":
            return sequence + (b'\x00' * padding_needed)
        elif self._padding_method == "repeat":
            full_repeats = padding_needed // original_length
            remainder = padding_needed % original_length
            return sequence + (sequence * full_repeats) + sequence[:remainder]
        else:
            return sequence


class Bz2Compressor(BaseCompressor):
    """BZ2 compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = None):
        super().__init__(
            compressor_name="bz2",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using BZ2 algorithm, removing header for size optimization."""
        return bz2.compress(sequence, compresslevel=self._compression_level)[15:]
        
        
class ZlibCompressor(BaseCompressor):
    """ZLIB compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = None):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using ZLIB algorithm with raw deflate (no headers)."""
        return zlib.compress(sequence, level=self._compression_level, wbits=-15)
    
    
class ZStandardCompressor(BaseCompressor):
    """Zstandard compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = None):
        super().__init__(
            compressor_name="zstandard",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )
    
    def compress(self, sequence: bytes) -> bytes:
        """Compress using Zstandard algorithm with minimal overhead."""
        compressor = ZstdCompressor(
            level=self._compression_level, 
            write_content_size=False, 
            write_checksum=False, 
            write_dict_id=False
        )
        return compressor.compress(sequence)


class RLECompressor(BaseCompressor):
    """Run Length Encoding - basic compression for repetitive data."""
    
    def __init__(self, compression_level: int = 1, min_size_threshold: Optional[int] = None):
        super().__init__(
            compressor_name="rle",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using Run Length Encoding algorithm."""
        if not sequence:
            return b""
        
        compressed = []
        current_byte = sequence[0]
        count = 1
        
        for byte in sequence[1:]:
            if byte == current_byte and count < 255:  # Max count of 255
                count += 1
            else:
                # Format: count (1 byte) + value (1 byte)
                compressed.extend([count, current_byte])
                current_byte = byte
                count = 1
        
        compressed.extend([count, current_byte])
        return bytes(compressed)


class HuffmanCompressor(BaseCompressor):
    """Simplified Huffman Coding - frequency-based compression."""
    
    def __init__(self, compression_level: int = 1, min_size_threshold: Optional[int] = None):
        super().__init__(
            compressor_name="huffman",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using simplified Huffman coding algorithm."""
        if not sequence:
            return b""
        
        # Count byte frequencies
        freq = Counter(sequence)
        
        # If too few unique symbols, compression is ineffective
        if len(freq) <= 2:
            return sequence
        
        # Sort by frequency (most frequent gets shorter codes)
        sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Assign simple codes based on frequency
        codes = self._generate_codes(sorted_bytes)
        
        # Compress sequence
        bit_string = ''.join(codes[byte] for byte in sequence)
        
        return self._bits_to_bytes(bit_string)
    
    def _generate_codes(self, sorted_bytes):
        """Generate variable-length codes for bytes based on frequency."""
        codes = {}
        code_length = 1
        code_value = 0
        
        for byte, _ in sorted_bytes:
            # Shorter codes for more frequent bytes
            if code_value >= (1 << code_length):
                code_length += 1
                code_value = 0
            
            codes[byte] = format(code_value, f'0{code_length}b')
            code_value += 1
        
        return codes
    
    def _bits_to_bytes(self, bit_string: str) -> bytes:
        """Convert bit string to bytes with padding information."""
        padding = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * padding
        
        compressed = bytearray([padding])  # First byte stores padding info
        
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte_bits = bit_string[i:i+8]
                compressed.append(int(byte_bits, 2))
        
        return bytes(compressed)


class LZ77Compressor(BaseCompressor):
    """LZ77 algorithm with configurable window size."""
    
    def __init__(self, compression_level: int = 1, min_size_threshold: Optional[int] = None):
        super().__init__(
            compressor_name="lz77",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )
        # Window size scales with compression level
        self.window_size = 10 + (compression_level * 2)  # 12-28
        self.lookahead_size = 5 + compression_level       # 6-14

    def compress(self, sequence: bytes) -> bytes:
        """Compress using LZ77 sliding window algorithm."""
        if not sequence:
            return b""
        
        compressed = bytearray()
        i = 0
        
        while i < len(sequence):
            # Search window (backward)
            start = max(0, i - self.window_size)
            search_window = sequence[start:i]
            
            # Lookahead buffer (forward)
            lookahead_end = min(len(sequence), i + self.lookahead_size)
            lookahead = sequence[i:lookahead_end]
            
            # Find longest match
            best_match = self._find_longest_match(search_window, lookahead)
            
            if best_match[1] > 0:
                # Match token: flag(1) + offset(1) + length(1)
                compressed.extend([1, best_match[0], best_match[1]])
                i += best_match[1]
            else:
                # Literal token: flag(0) + byte
                compressed.extend([0, sequence[i]])
                i += 1
        
        return bytes(compressed)
    
    def _find_longest_match(self, search_window: bytes, lookahead: bytes) -> tuple:
        """Find the longest match in the search window."""
        best_match = (0, 0)  # (offset, length)
        
        for j in range(len(search_window)):
            match_len = 0
            while (match_len < len(lookahead) and 
                   j + match_len < len(search_window) and
                   search_window[j + match_len] == lookahead[match_len]):
                match_len += 1
            
            if match_len > best_match[1] and match_len >= 3:  # Minimum 3 bytes
                best_match = (len(search_window) - j, match_len)
        
        return best_match


# Strategy pattern mapping
COMPRESSOR_STRATEGIES = {
    CompressorType.BZ2: Bz2Compressor,
    CompressorType.ZLIB: ZlibCompressor,
    CompressorType.ZSTD: ZStandardCompressor,
    CompressorType.RLE: RLECompressor,
    CompressorType.HUFFMAN: HuffmanCompressor,
    CompressorType.LZ77: LZ77Compressor
}


def get_compressor(
    name: Union[str, CompressorType],
    compression_level: int = 9,
    min_size_threshold: Optional[int] = None
) -> BaseCompressor:
    """Factory function to get a specific compressor instance.
    
    Args:
        name: Compressor name ('rle', 'huffman', 'lz77', 'zlib', 'bz2', 'zstd')
        compression_level: Compression level (1-9)
        min_size_threshold: Minimum size for padding
    
    Returns:
        Compressor instance
        
    Raises:
        ValueError: If compressor name is invalid
        TypeError: If name type is invalid
    """
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
        raise TypeError("Expected 'name' to be str or CompressorType")
    
    compressor_class = COMPRESSOR_STRATEGIES[compressor_enum]
    return compressor_class(
        compression_level=compression_level,
        min_size_threshold=min_size_threshold
    )


def compute_compression(
    sequence: str, 
    compressor: str, 
    compression_level: int, 
    min_size_threshold: Optional[int] = None
) -> bytes:
    """Compute compression for a given sequence using specified compressor.
    
    Args:
        sequence: Input string to compress
        compressor: Name of the compressor to use
        compression_level: Compression level (1-9)
        min_size_threshold: Minimum size threshold for padding
        
    Returns:
        Compressed data as bytes
    """
    compressor_instance = get_compressor(
        name=compressor.lower(),
        compression_level=compression_level,
        min_size_threshold=min_size_threshold
    )
    
    return compressor_instance(sequence)