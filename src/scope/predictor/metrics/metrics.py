import ot
import numpy as np
from typing import Optional


def squared_euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    u = np.asarray(x1, dtype=float)
    v = np.asarray(x2, dtype=float)
    
    u_sq = np.sum(u * u)
    v_sq = np.sum(v * v) 
    uv = np.sum(u * v)
    
    dist = u_sq + v_sq - 2 * uv
    return float(dist)
    
def matching(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute matching (intersection) between two arrays"""
    return np.sum(
            np.minimum(x1, x2), 
            dtype=np.float32
        ).item()

def jaccard(x1: np.ndarray, x2: np.ndarray) -> float:
    """Jaccard distance: 1 - (intersection / union)"""
    intersection = matching(x1, x2)
    union = np.sum(np.maximum(x1, x2), dtype=np.float32)
    
    if union < 1e-12:
        return 0.0
    
    return 1.0 - (intersection / union)

def dice(x1: np.ndarray, x2: np.ndarray) -> float:
    """Dice distance: 1 - (2 * intersection / (sum1 + sum2))"""
    intersection = matching(x1, x2)  # Corregido: era self._matching
    sum1 = np.sum(x1, dtype=np.float32)
    sum2 = np.sum(x2, dtype=np.float32)
    denominator = sum1 + sum2
    
    if denominator < 1e-12:
        return 0.0
    
    return 1.0 - (2 * intersection / denominator)

def overlap(x1: np.ndarray, x2: np.ndarray) -> float:
    """Overlap distance: 1 - (intersection / min(sum1, sum2))"""
    intersection = matching(x1, x2)  # Corregido: era self._matching
    sum1 = np.sum(x1, dtype=np.float32)
    sum2 = np.sum(x2, dtype=np.float32)
    denominator = min(sum1, sum2)
    
    if denominator < 1e-12:
        return 0.0
    
    return 1.0 - (intersection / denominator)

matchin_cost_matrix: dict = {
    'jaccard': lambda x1, x2: jaccard(x1, x2),
    'dice': lambda x1, x2: dice(x1, x2),
    'overlap': lambda x1, x2: overlap(x1, x2),
}

def wasserstein(x1: np.ndarray, x2: np.ndarray, cost_matrix: Optional[str] = None) -> float:
    
    cluster_weights = np.ones(x2.shape[0]) / x2.shape[0]            
    sample_weights = np.ones(x1.shape[0]) / x1.shape[0]
    
    if cost_matrix:
        cost_matrix_func = matchin_cost_matrix[cost_matrix]
        cost_matrix_values = np.array([
            [
                cost_matrix_func(
                    x1=sample.reshape(1, -1), 
                    x2=kw_sample.reshape(1, -1)
                ) for sample in x1]
            for kw_sample in x2
            ]
        )
    else:
        cost_matrix_values = ot.dist(x2, x1, metric='euclidean')
    
    return ot.emd2(cluster_weights, sample_weights, cost_matrix_values)
