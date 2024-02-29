import numpy as np


def generate_samples(x: np.ndarray, y: np.ndarray, samples: int = 5) -> tuple:
    classes: np.ndarray = np.unique(y)
    for index in range(len(x)):
        test_y: np.ndarray = y[index]
        test_x = x[index]
        kw_samples: list = []
        for classIndex in range(len(classes)):
            mask: np.ndarray = np.where(y == classes[classIndex])[0]
            while True:
                random_index: np.ndarray = np.random.choice(mask, size=samples, replace=True)
                if index not in random_index:
                    break
            kw_samples.append(x[random_index])
        yield test_x, kw_samples, test_y


def gauss(x: np.ndarray, sigma: float = 0.12) -> np.ndarray:
    return np.exp(-0.5 * np.square((x/sigma)))
