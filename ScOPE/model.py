import numpy as np

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
        self.compressor = Compressor(matrix_kwargs['compressor_name'])
        self.distance = Distance(matrix_kwargs['distance_name'])
        self.__predict__ = np.vectorize(lambda query, sample: -MatchingMethods(query, sample, method)(),
                                        signature='(n), (m)->()')

    def __best_sigma__(self, query, *samples) -> float:

        def current_sigma(sequence: str) -> float:
            def join(x1, x2) -> str | np.ndarray:
                if self.default_matrix_args['append_type'] == 'text':
                    return f'{self.default_matrix_args["sep"]}'.join([x1, x2])
                elif self.default_matrix_args['append_type'] == 'text_as_array':
                    return np.append(x1, x2)

            x1_compressed_len: float = self.compressor(sequence=sequence)
            x1x1: str = join(sequence, sequence)
            x1x1_compressed_len: float = self.compressor(sequence=x1x1)
            this_sigma = self.distance(x1=x1_compressed_len,
                                       x2=x1_compressed_len,
                                       x1x2=x1x1_compressed_len)
            return this_sigma

        get_sigma = np.vectorize(current_sigma)
        sigmas: list = [current_sigma(query)]

        sigmas.extend(get_sigma(samples).flatten())
        return np.array(sigmas).mean()

    def predict(self, query: str | list | np.ndarray, *classes: list | np.ndarray,
                softmax: bool = False) -> np.ndarray:

        sigma = self.__best_sigma__(query, *classes,)

        compression_matrix: np.ndarray = self.matrix.get_matrix(query, classes) # noqa
        gauss_matrix: np.ndarray = np.exp(-0.5 * np.square((compression_matrix / sigma)))
        classifications: np.ndarray = np.zeros(shape=(len(gauss_matrix)),
                                               dtype=np.float32)

        for index, x in enumerate(gauss_matrix):
            query: np.ndarray = x[-1]
            samples: np.ndarray = x[:-1]  # noqa

            classifications[index] = self.__predict__(query, samples).sum()

        if softmax is True:
            return np.exp(classifications)/sum(np.exp(classifications))
        return classifications


if __name__ == '__main__':
    model = ScOPEModel(method='dice',
                       compressor_name='bz2',
                       distance_name='ncd',
                       append_type='text',
                       )

    class0: list = ['Hola', 'aloooo', 'ola', 'holi']
    class1: list = ['Hello', 'Hellow', 'Heloooo', 'hellow']
    class3: list = ['adio', 'adiso', 'bye', 'bie']
    test: str = 'Helow'

    pred = model.predict(test, class0, class1, class3, softmax=True)
    print(pred)
