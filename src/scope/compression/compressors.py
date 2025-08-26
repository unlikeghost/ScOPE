# -*- coding: utf-8 -*-

import bz2
import zlib
from zstandard import ZstdCompressor
from enum import Enum
from abc import ABC, abstractmethod
from typing import Union, Optional
from collections import Counter


class CompressorType(Enum):
    BZ2 = "bz2"
    ZLIB = "zlib"
    ZSTD = "zstd"
    RLE = "rle"
    HUFFMAN = "huffman"
    LZ77 = "lz77"
    

class _BaseCompressor(ABC):

    def __init__(self, compressor_name: str, compression_level: int = 9, min_size_threshold: Optional[int] = None, padding_method: Optional[str] = None):
        """Initializes the BaseCompressor with the specified compression module.

        Args:
            compressor_name (str): The name of the compression method being used (e.g., 'gzip' or 'bz2').
            min_size_threshold (int, optional): Minimum size for effective compression. Defaults to 50.
            compression_level (int, optional): The level of compression to apply (1-9). Defaults to 9.
        """
        
        if compression_level < 1 or compression_level > 9:
            raise ValueError("Compression level must be between 1 and 9.")
        
        if min_size_threshold and min_size_threshold > 0:
            if padding_method and padding_method not in ["zeros", "repeat"]:
                raise ValueError("padding_method must be 'zeros' or 'repeat'")
            padding_method = padding_method or 'zeros'

        self._min_size_threshold = min_size_threshold if min_size_threshold else 0
        self._padding_method: str = padding_method
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

    
    def __call__(self, sequence: Union[str, bytes]) -> bytes:
        """Compresses the input sequence and returns the compressed data.

        Args:
            sequence (Union[str, bytes]): The input data to compress.

        Raises:
            TypeError: If the input sequence is not of type 'str' or 'bytes'.

        Returns:
            bytes: The compressed data.
        """
        if len(sequence) == 0:
            raise ValueError(
                f"Empty sequence provided to {self._compressor_name} compressor. "
                f"Compression requires non-empty input data. "
                f"Check your data preprocessing pipeline for sources of empty strings."
            )
                
        if not isinstance(sequence, (bytes, str)):
            raise TypeError("Input sequence must be of type 'str' or 'bytes'.")
        
        if isinstance(sequence, str):
            sequence_encoded = sequence.encode('utf-8')
        else:
            sequence_encoded = sequence
        
        original_length = len(sequence_encoded)
        target_length = self._min_size_threshold
        
        if self._should_pad_sequence(sequence_encoded):
            if original_length == 0:
                raise ValueError("Sequence must have at least 1 item")
            
            padding_needed = target_length - original_length
                        
            if self._padding_method == "zeros":
                sequence_to_compress = sequence_encoded + (b'\x00' * padding_needed)
            
            elif self._padding_method == "repeat":
                full_repeats = padding_needed // original_length
                remainder = padding_needed % original_length
                sequence_to_compress = sequence_encoded + (sequence_encoded * full_repeats) + sequence_encoded[:remainder]
                
        else:
            sequence_to_compress = sequence_encoded
        
        return self.compress(sequence_to_compress)


class Bz2(_BaseCompressor):
    def __init__(self, compression_level: int = 9, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="bz2",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        return bz2.compress(sequence, compresslevel=self._compression_level)[15:]
        
        
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
        compressor = ZstdCompressor(level=self._compression_level, write_content_size=False, write_checksum=False, write_dict_id=False)
        return compressor.compress(sequence)


class RLE(_BaseCompressor):
    """Run Length Encoding - compresión muy baja"""
    
    def __init__(self, compression_level: int = 1, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="rle",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        if not sequence:
            return b""
        
        compressed = []
        current_byte = sequence[0]
        count = 1
        
        for byte in sequence[1:]:
            if byte == current_byte and count < 255:  # Límite de 255
                count += 1
            else:
                # Formato: count (1 byte) + valor (1 byte)
                compressed.extend([count, current_byte])
                current_byte = byte
                count = 1
        
        compressed.extend([count, current_byte])
        return bytes(compressed)


class Huffman(_BaseCompressor):
    """Huffman Coding simplificado - compresión baja a moderada"""
    
    def __init__(self, compression_level: int = 1, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="huffman",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )

    def compress(self, sequence: bytes) -> bytes:
        if not sequence:
            return b""
        
        # Para simplificar, usar una compresión básica que simule Huffman
        # pero sin la complejidad del árbol
        freq = Counter(sequence)
        
        # Si hay pocos símbolos únicos, no comprime bien
        if len(freq) <= 2:
            return sequence
        
        # Ordenar por frecuencia (más frecuente = código más corto)
        sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Asignar códigos simples basados en frecuencia
        codes = {}
        code_length = 1
        code_value = 0
        
        for byte, _ in sorted_bytes:
            # Códigos más cortos para bytes más frecuentes
            if code_value >= (1 << code_length):
                code_length += 1
                code_value = 0
            
            codes[byte] = format(code_value, f'0{code_length}b')
            code_value += 1
        
        # Comprimir
        bit_string = ''.join(codes[byte] for byte in sequence)
        
        # Padding y conversión a bytes
        padding = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * padding
        
        compressed = bytearray([padding])  # Primer byte = padding
        
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte_bits = bit_string[i:i+8]
                compressed.append(int(byte_bits, 2))
        
        return bytes(compressed)


class LZ77(_BaseCompressor):
    """LZ77 básico con ventana pequeña - compresión moderada"""
    
    def __init__(self, compression_level: int = 1, min_size_threshold: Optional[int] = 0):
        super().__init__(
            compressor_name="lz77",
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )
        # Ventana más pequeña = menos compresión
        self.window_size = 10 + (compression_level * 2)  # 12-28
        self.lookahead_size = 5 + compression_level       # 6-14

    def compress(self, sequence: bytes) -> bytes:
        if not sequence:
            return b""
        
        compressed = bytearray()
        i = 0
        
        while i < len(sequence):
            # Ventana de búsqueda
            start = max(0, i - self.window_size)
            search_window = sequence[start:i]
            
            # Buffer de lookahead
            lookahead_end = min(len(sequence), i + self.lookahead_size)
            lookahead = sequence[i:lookahead_end]
            
            # Buscar coincidencia más larga
            best_match = (0, 0)  # (offset, length)
            
            for j in range(len(search_window)):
                match_len = 0
                while (match_len < len(lookahead) and 
                       j + match_len < len(search_window) and
                       search_window[j + match_len] == lookahead[match_len]):
                    match_len += 1
                
                if match_len > best_match[1] and match_len >= 3:  # Mínimo 3 bytes
                    best_match = (len(search_window) - j, match_len)
            
            if best_match[1] > 0:
                # Token de coincidencia: flag(1) + offset(1) + length(1)
                compressed.extend([1, best_match[0], best_match[1]])
                i += best_match[1]
            else:
                # Literal: flag(0) + byte
                compressed.extend([0, sequence[i]])
                i += 1
        
        return bytes(compressed)


COMPRESSOR_STRATEGIES = {
    CompressorType.BZ2: Bz2,
    CompressorType.ZLIB: Zlib,
    CompressorType.ZSTD: ZStandard,
    CompressorType.RLE: RLE,
    CompressorType.HUFFMAN: Huffman,
    CompressorType.LZ77: LZ77
}


def get_compressor(
    name: Union[str, CompressorType],
    compression_level: int = 9,
    min_size_threshold: Optional[int] = 0
) -> _BaseCompressor:
    """
    Obtiene un compresor específico.
    
    Args:
        name: nombre del compresor ('rle', 'huffman', 'lz77', 'zlib', 'bz2', 'zstd')
        compression_level: nivel de compresión (1-9)
        min_size_threshold: tamaño mínimo para padding
    
    Returns:
        Instancia del compresor
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
        raise TypeError("Expected 'name' to be str or CompressorType.")
    
    compressor_class = COMPRESSOR_STRATEGIES[compressor_enum]
    return compressor_class(
        compression_level=compression_level,
        min_size_threshold=min_size_threshold
    )


def compute_compression(sequence: str, compressor: str, compression_level: int, min_size_threshold: int) -> bytes:
    compressor = compressor.lower()
    compressor_function = get_compressor(
        name=compressor,
        compression_level=compression_level,
        min_size_threshold=min_size_threshold
    )
    data = sequence.encode('utf-8')
    
    compression = compressor_function.compress(data)
    
    return compression