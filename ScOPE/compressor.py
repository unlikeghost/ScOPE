# encoding: utf-8
"""
    Created on Fri Nov 24 2023
    Author: Jesus Alan Hernandez Galvan
    Email: alanhernandezgalvan@icloud.com
"""
import numpy as np
from typing import Union
from importlib import import_module


class Compressor:
    
    def __init__(self, compressor:str) -> None:
        """Compressor class constructor

        Args:
            compressor (str): Compressor name

        Raises:
            ValueError: If compressor is not one of the following: {'bz2', 'gzip', 'zlib'}
        """
        
        __supported_compressors__ = {'bz2', 'gzip', 'zlib'}
        
        self.__supported_types__ = {'str', 'array'}
        
        if compressor not in __supported_compressors__:
            print(f'Error: {compressor} is not a supported compressor')
            print(f'Please choose one of the following: {__supported_compressors__}')
            raise ValueError
        
        self.compressor = import_module(compressor)
        self.compressor_name = compressor

    def __repr__(self) -> str:
        return f'Compressor({self.compressor_name}))'
    
    def __str__(self) -> str:
        return f'Compressor({self.compressor_name}))'
    
    def __compress_text__(self, sequence:str) -> float:
        return float(len(self.compressor.compress(sequence.encode('utf-8'), compresslevel=6)))
    
    def __compress_array__(self, sequence:np.ndarray) -> float:
        return float(len(self.compressor.compress(sequence.tobytes())))
        
    def __call__(self, sequence:Union[str, np.ndarray]) -> float:
        """Compress a sequence

        Args:
            sequence (Any[str, np.ndarray]): Compressible sequence, either a string or a numpy array

        Returns:
            float: Compressed size
        """
        return self.compress(sequence)
    
    def compress(self, sequence:Union[str, np.ndarray]) -> float:
        """Compress a sequence

        Args:
            sequence (Union[str, np.ndarray]): Compressible sequence, either a string or a numpy array

        Raises:
            ValueError: If sequence is not one of the following: {str, np.ndarray}

        Returns:
            float: Compressed size
        """
        if isinstance(sequence, str):
            return self.__compress_text__(sequence)
        elif type(sequence) in [np.ndarray, list, tuple]:
            sequence:np.ndarray = np.array(sequence)
            return self.__compress_array__(sequence)
        else:
            print(f'Error: {type(sequence)} is not a supported type')
            print(f'Please choose one of the following: {self.__supported_types__}')
            raise ValueError
    
if __name__ == '__main__':
    compressor = Compressor('gzip')
    print(compressor)
    print(compressor(sequence='Hola'))
    print(compressor(sequence=[1, 2, 3, 4, 5]))
    print(compressor(sequence=np.array([1, 2, 3, 4, 5])))
   