import numpy as np
from typing import List, Union, Callable


class Matrix:
    def __init__(self, compressor: callable, distance: callable, sep: str = ' ', append_type: str = 'array'):

        __supported_types__: tuple = ('text',
                                      'text_as_array')

        if append_type not in __supported_types__:
            print(f'Error: {append_type} is not a supported type')
            print(f'Please choose one of the following: {__supported_types__}')
            raise ValueError

        self.distance = distance
        self.compressor = compressor
        self.sep = sep
        self.append_type = append_type

    def __repr__(self) -> str:
        return f'Matrix({self.compressor}, {self.distance})'

    def __str__(self) -> str:
        return f'Matrix({self.compressor}, {self.distance})'

    def join(self, x1, x2):
        if self.append_type == 'text':
            return f'{self.sep}'.join([x1, x2])
        elif self.append_type == 'text_as_array':
            return np.append(x1, x2)

    def __calc_matrix__(self, samples: np.ndarray) -> np.ndarray:

        distances: np.ndarray = np.zeros(shape=(len(samples), len(samples)),
                                         dtype=np.float32)
        for index_i in range(len(samples)):
            current_x1: np.ndarray = samples[index_i]
            x1_compressed_len: float = self.compressor(sequence=current_x1)
            for index_j in range(index_i, len(samples)):
                current_x2: np.ndarray = samples[index_j]
                x2_compressed_len: float = self.compressor(sequence=current_x2)
                x1x2: str = self.join(current_x1, current_x2)
                x1x2_compressed_len: float = self.compressor(sequence=x1x2)
                distance = self.distance(x1=x1_compressed_len,
                                         x2=x2_compressed_len,
                                         x1x2=x1x2_compressed_len)
                distances[index_i, index_j] = distance
                distances[index_j, index_i] = distance

        return distances

    def get_matrix(self, sample, classes):
        classes: dict = {index: values for index, values in enumerate(classes)}
        matrix: np.ndarray = np.zeros(shape=(len(classes),
                                             len(classes[0])+1,
                                             len(classes[0])+1),
                                      dtype=np.float32)
        for class_ in classes:
            samples = np.append(classes[class_], sample)
            matrix[class_, :, :] = self.__calc_matrix__(samples)
        return matrix


if __name__ == '__main__':
    from compressor import Compressor
    from distance import Distance

    compressor = Compressor('gzip')
    distance = Distance('ncd')

    matrix = Matrix(compressor, distance, append_type='text')
    class0 = ['Hola', 'Adios', 'Buenos dias']
    class1 = ['Hello', 'Goodbye', 'Good morning']
    sample = 'Hello'
    print(matrix.get_matrix(sample, [class0, class1]))

    matrix = Matrix(compressor, distance, append_type='text_as_array')
    class0 = np.array(['Hola', 'Adios', 'Buenos dias'])
    class1 = np.array(['Hello', 'Goodbye', 'Good morning'])
    sample = 'Helloo'
    print(matrix.get_matrix(sample, [class0, class1]))