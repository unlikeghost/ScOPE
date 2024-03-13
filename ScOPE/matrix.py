import numpy as np
from ScOPE.distance import Distance
from ScOPE.compressor import Compressor


class Matrix:
    def __init__(self, compressor_name: str, distance_name: str, sep: str = ' ', append_type: str = 'text') -> None:
        """
        @param compressor_name: name of the compressor to use
        @param distance_name: name of the distance to use
        @param sep: separator between query text and sample if using append_type = text
        @param append_type: type of append text (text or text_as_array)
        """
        __supported_types__: tuple = ('text',
                                      'text_as_array')

        if append_type not in __supported_types__:
            print(f'Error: {append_type} is not a supported type')
            print(f'Please choose one of the following: {__supported_types__}')
            raise ValueError

        self.distance = Distance(distance_name)
        self.compressor = Compressor(compressor_name)
        self.sep = sep
        self.append_type = append_type

    def __repr__(self) -> str:
        return f'Matrix({self.compressor}, {self.distance})'

    def __str__(self) -> str:
        return f'Matrix({self.compressor}, {self.distance})'

    def join(self, x1, x2) -> str | np.ndarray:
        """
        Way to join two samples
        @param x1: sample 1
        @param x2: sample 2
        @return: Str or np.ndarray, depends on append_type
        """
        if self.append_type == 'text':
            return f'{self.sep}'.join([x1, x2])
        elif self.append_type == 'text_as_array':
            return np.append(x1, x2)

    def __calc_matrix__(self, samples: np.ndarray) -> np.ndarray:
        """
        Calculate the matrix distances
        @param samples: np.ndarray of shape (n_samples + query, n_samples + query)
        @return: np.ndarray of shape (n_samples + query, n_samples + query)
        """
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

    def get_matrix(self, sample: str | list | np.ndarray, classes: list | np.ndarray) -> np.ndarray:
        """
        @param sample: sample to query
        @param classes: samples of known classes
        @return: np.ndarray with shape (len(classes), len(samples) + query, len(samples) + query)
        """
        classes: dict = {index: values for index, values in enumerate(classes)}
        matrix: np.ndarray = np.zeros(shape=(len(classes),
                                             len(classes[0])+1,
                                             len(classes[0])+1),
                                      dtype=np.float32)
        for class_ in classes:
            samples = np.append(classes[class_], sample)
            matrix[class_, :, :] = self.__calc_matrix__(samples)
        return matrix


class MatrixEnsamble:
    def __init__(self, compressor_names: str | list, distance_names: str | list, **kwargs):
        if compressor_names == '__all__':
            compressor_names = ['bz2', 'gzip']
        if distance_names == '__all__':
            distance_names = ['cdm', 'clm', 'ncd']

        self.matrices = [
            Matrix(compressor_name=current_compressor, distance_name=current_distance, **kwargs)
            for current_compressor in compressor_names
            for current_distance in distance_names
        ]

    def get_matrix(self, sample: str | list | np.ndarray, classes: list | np.ndarray) -> np.ndarray:
        """
       @param sample: sample to query
        @param classes: samples of known classes
        @return: Matrix with shape (#_class, len(compressor_names) * len(distance_names), len(samples) + query)
        """
        data: list = []
        for matrix in self.matrices:
            data.append(matrix.get_matrix(sample, classes))
        data: np.ndarray = np.array(data)
        return data.reshape(
            data.shape[1],
            data.shape[0],
            data.shape[2],
            data.shape[3]
        )


if __name__ == '__main__':

    matrix = Matrix(compressor_name='gzip',
                    distance_name='ncd',
                    append_type='text')
    class0 = ['Hola', 'Adios', 'Buenos dias']
    class1 = ['Hello', 'Goodbye', 'Good morning']
    sample = 'Hello'
    print(matrix.get_matrix(sample, [class0, class1]))

    matrix = Matrix(compressor_name='gzip',
                    distance_name='ncd',
                    append_type='text_as_array')
    class0 = np.array(['Hola', 'Adios', 'Buenos dias'])
    class1 = np.array(['Hello', 'Goodbye', 'Good morning'])
    sample = 'Helloo'
    print(matrix.get_matrix(sample, [class0, class1]))

    matrixEnsamble = MatrixEnsamble(compressor_names='__all__', distance_names='__all__',
                                    append_type='text')
    class0 = np.array(['Hola', 'Adios', 'Buenos dias'])
    class1 = np.array(['Hello', 'Goodbye', 'Good morning'])
    sample = 'Helloo'
    print(matrixEnsamble.get_matrix(sample, [class0, class1]).shape)
