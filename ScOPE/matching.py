import numpy as np


class MatchingMethods:
    def __init__(self, x1: np.ndarray, x2: np.ndarray, method: str = '') -> None:
        self.x1 = x1
        self.x2 = x2
        self.method = method

    def matching(self) -> np.ndarray:
        return np.sum(np.minimum(self.x1, self.x2), dtype=np.float32)

    def jaccard(self) -> np.ndarray:
        _matching = self.matching()
        return 1.0 - (_matching / np.sum(np.maximum(self.x1, self.x2)))

    def dice(self) -> np.ndarray:
        _matching = 2 * self.matching()
        return 1.0 - (_matching / (np.sum(self.x1) + np.sum(self.x2)))

    def overlap(self) -> np.ndarray:
        _matching = self.matching()
        return 1.0 - (_matching / np.sum(self.x1) + np.sum(self.x2))

    def __call__(self, *args, **kwargs):
        allowed_methods: dict = {
            'matching': self.matching(),
            'jaccard': self.jaccard(),
            'dice': self.dice(),
            'overlap': self.overlap()
        }
        if self.method != '':
            return allowed_methods[self.method]


if __name__ == '__main__':
    x1: np.ndarray = np.array([1, 2, 3, 4, 5])
    x2: np.ndarray = np.array([1])

    mm_str = MatchingMethods(x1, x2, 'dice')
    print(mm_str())

    mm_func = MatchingMethods(x1, x2)
    print(mm_func.matching(), mm_func.jaccard(),
          mm_func.dice(), mm_func.overlap())