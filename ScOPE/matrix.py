import numpy as np
from typing import List, Union, Callable

class Matrix:
    def __init__(self, compressor:Callable, distance:Callable, typ:str) -> None:
        
        __supported_types__ = {'text', 'array', 'text_as_array'}
        if typ not in __supported_types__:
            print(f'Error: {typ} is not a supported type')
            print(f'Please choose one of the following: {__supported_types__}')
            raise ValueError
        
        self.compressor = compressor
        self.distance = distance
        self.typ = typ
    
    def __repr__(self) -> str:
        return f'Matrix({self.compressor}, {self.distance})'
    
    def __str__(self) -> str:
        return f'Matrix({self.compressor}, {self.distance})'
    
    def __calc_distance_text__(self, samples:np.ndarray) -> np.ndarray:
        distances:np.ndarray = np.zeros(shape=(len(samples), len(samples)),
                                        dtype=np.float32)
        
        for index_i in range(len(samples)):
            x1:str = samples[index_i]
            x1_compressed_len:float = self.compressor(sequence=x1)
            
            for index_j in range(index_i, len(samples)):
                x2:str = samples[index_j]
                x2_compressed_len:float = self.compressor(sequence=x2)
                
                x1x2:str = ' '.join([x1, x2])
                x1x2_compressed_len:float = self.compressor(sequence=x1x2)
                                
                distance = self.distance(x1= x1_compressed_len,
                                         x2= x2_compressed_len,
                                         x1x2= x1x2_compressed_len)
                
                distances[index_i, index_j] = distance
                distances[index_j, index_i] = distance
    
        return distances
    
    def __calc_distance_array__(self, samples:np.ndarray) -> np.ndarray:
        distances:np.ndarray = np.zeros(shape=(len(samples), len(samples)),
                                        dtype=np.float32)
        
        for index_i in range(len(samples)):
            x1:np.ndarray = samples[index_i]
            x1_compressed_len:float = self.compressor(sequence=x1)
            
            for index_j in range(index_i, len(samples)):
                x2:np.ndarray = samples[index_j]
                x2_compressed_len:float = self.compressor(sequence=x2)         
                x1x2:np.ndarray = np.array([x1, x2])
                x1x2_compressed_len:float = self.compressor(sequence=x1x2)
                                
                distance = self.distance(x1= x1_compressed_len,
                                         x2= x2_compressed_len,
                                         x1x2= x1x2_compressed_len)
                
                distances[index_i, index_j] = distance
                distances[index_j, index_i] = distance
    
        return distances
    
    def __calc_distance_textasarray__(self, samples:np.ndarray) -> np.ndarray:
        distances:np.ndarray = np.zeros(shape=(len(samples), len(samples)),
                                        dtype=np.float32)
        
        for index_i in range(len(samples)):
            x1:np.array = np.array([samples[index_i]])
            x1_compressed_len:float = self.compressor(sequence=x1)
            
            for index_j in range(index_i, len(samples)):
                x2:np.array = np.array([samples[index_j]])
                x2_compressed_len:float = self.compressor(sequence=x2)
                x1x2:np.ndarray = np.append(x1, x2)
                x1x2_compressed_len:float = self.compressor(sequence=x1x2)
                
                distance = self.distance(x1= x1_compressed_len,
                                         x2= x2_compressed_len,
                                         x1x2= x1x2_compressed_len)
                
                distances[index_i, index_j] = distance
                distances[index_j, index_i] = distance
                
        return distances
    
    def get_matrix(self, sample, classes):
        
        classes:dict = {index:values for index, values in enumerate(classes)}
        
        matrix:np.ndarray = np.zeros(shape=(len(classes),
                                            len(classes[0])+1, len(classes[0])+1),
                                     dtype=np.float32)
        
        for class_ in classes:
            if self.typ == 'text':
                samples:np.ndarray = np.append(classes[class_], sample)
                matrix[class_, :, :] = self.__calc_distance_text__(samples)
            elif self.typ == 'array':
                sample:np.ndarray = np.expand_dims(sample, axis=0) if len(sample.shape) == 1 else sample
                samples:np.ndarray = np.concatenate([classes[class_], sample])
                matrix[class_, :, :] = self.__calc_distance_array__(samples)
            else:
                samples:np.ndarray = np.append(classes[class_], sample)
                matrix[class_, :, :] = self.__calc_distance_textasarray__(samples)
        
        return matrix
        
if __name__ == '__main__':
    from compressor import Compressor
    from distance import Distance
    
    compressor = Compressor('gzip')
    distance = Distance('ncd')
    
    matrix = Matrix(compressor, distance, typ='text')
    class0 = ['Hola', 'Adios', 'Buenos dias']
    class1 = ['Hello', 'Goodbye', 'Good morning']
    sample = 'Hello'
    print(matrix.get_matrix(sample, [class0, class1]))
    print()
    
    matrix = Matrix(compressor, distance, typ='array')
    class0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    class1 = np.array([[10.0, 11.0, 12.0], [13, 14, 15], [16, 17, 18]])
    sample = np.array([10.1, 11.1, 12.1])
    print(matrix.get_matrix(sample, [class0, class1]))
    print()
    
    matrix = Matrix(compressor, distance, typ='text_as_array')
    class0 = np.array(['Hola', 'Adios', 'Buenos dias'])
    class1 = np.array(['Hello', 'Goodbye', 'Good morning'])
    sample = np.array(['Helloo'])
    print(matrix.get_matrix(sample, [class0, class1]))
    print()