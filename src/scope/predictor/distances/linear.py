import numpy as np


def squared_euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    u = np.asarray(x1, dtype=float)
    v = np.asarray(x2, dtype=float)
    
    u_sq = np.sum(u * u)
    v_sq = np.sum(v * v) 
    uv = np.sum(u * v)
    
    dist = u_sq + v_sq - 2 * uv
    return float(dist)

def euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    x1, x2 = np.asarray(x1), np.asarray(x2)
    return float(
        np.linalg.norm(x1 - x2)
    )

def manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
    x1, x2 = np.asarray(x1), np.asarray(x2)
    return float(np.sum(np.abs(x1 - x2)))

def chebyshev(x1: np.ndarray, x2: np.ndarray) -> float:
    x1, x2 = np.asarray(x1), np.asarray(x2)
    return float(np.max(np.abs(x1 - x2)))