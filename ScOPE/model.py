import numpy as np
from typing import Union

from ScOPE import Distance
from ScOPE.matrix import Matrix
from ScOPE.compressor import Compressor
from ScOPE.matching import MatchingMethods


class ScOPEModel:
    default_matrix_args: dict = {
        'compressor_name': 'bz2',
        'distance_name': 'ncd',
        'append_type': 'text',
        'sep': '\t'
    }

    allowed_methods: set = {
        'matching',
        'jaccard',
        'dice',
        'overlap'
    }

    def __init__(self, method: str, **matrix_kwargs):
        if method not in self.allowed_methods:
            raise ValueError(f'Method "{method}" is not allowed')
        matrix_kwargs: dict = self.default_matrix_args | matrix_kwargs

        self.matrix = Matrix(**matrix_kwargs)
        self.method = method
        self.compressor = Compressor(matrix_kwargs['compressor_name'])
        self.distance = Distance(matrix_kwargs['distance_name'])

    def __best_sigma__(self, query, *samples) -> float:

        def current_sigma(sample: str) -> float:
            def join(x1, x2) -> Union[str, np.ndarray]:
                if self.default_matrix_args['append_type'] == 'text':
                    return f'{self.default_matrix_args["sep"]}'.join([x1, x2])
                elif self.default_matrix_args['append_type'] == 'text_as_array':
                    return np.append(x1, x2)

            x1_compressed_len: float = self.compressor(sequence=sample)
            x1x1: str = join(sample, sample)
            x1x1_compressed_len: float = self.compressor(sequence=x1x1)
            current_sigma = self.distance(x1=x1_compressed_len,
                                          x2=x1_compressed_len,
                                          x1x2=x1x1_compressed_len)
            return current_sigma

        sigmas: list = [current_sigma(query)]
        for cls in samples:
            sigmas.extend(list(map(lambda sample: current_sigma(sample), cls)))

        return np.array(sigmas).mean()

    def predict(self, query: Union[str, list, np.ndarray], *classes: Union[list, np.ndarray],
                softmax: bool = False) -> np.ndarray:

        sigma = self.__best_sigma__(query, *classes,)

        compression_matrix: np.ndarray = self.matrix.get_matrix(query, classes) # noqa
        gauss_matrix: np.ndarray = np.exp(-0.5 * np.square((compression_matrix / sigma)))
        classifications: np.ndarray = np.zeros(shape=(len(gauss_matrix)),
                                               dtype=np.float32)
        for index, x in enumerate(gauss_matrix):
            query: np.ndarray = x[-1]
            samples: np.ndarray = x[:-1]  # noqa

            for index_s, sample in enumerate(samples):
                mm = MatchingMethods(query, sample, self.method) # noqa
                classifications[index] -= mm()

        if softmax is True:
            return np.exp(classifications)/sum(np.exp(classifications))
        return classifications


if __name__ == '__main__':
    model = ScOPEModel(method='dice',
                       compressor_name='bz2',
                       distance_name='ncd',
                       append_type='text',
                       )

    class0 = ['Hola', 'aloooo', 'ola']
    class1 = ['Hello', 'Hellow', 'Heloooo']
    sample = 'Helow'

    pred = model.predict(sample, class0, class1, softmax=True)
    print(pred)
