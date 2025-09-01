import numpy as np
from scope.predictor.base import _BasePredictor

from .metrics import squared_euclidean, wasserstein


class ScOPEPredictor(_BasePredictor):

    def __init__(self,
                 evaluation_metric: str = 'squared_euclidean',
                 **kwargs) -> None:
        
        super().__init__(
            **kwargs
        )
        
        
        self.supported_metrics = {
            "squared_euclidean": lambda x1, x2: squared_euclidean(x1, x2),
            "wasserstein": lambda x1, x2: wasserstein(x1, x2),
            "wasserstein-jaccard": lambda x1, x2: wasserstein(x1, x2, 'jaccard'),
            "wasserstein-dice": lambda x1, x2: wasserstein(x1, x2, 'dice'),
            "wasserstein-overlap": lambda x1, x2: wasserstein(x1, x2, 'overlap'),
        }
        
        if evaluation_metric not in self.supported_metrics:
                raise ValueError(f"Unsupported distance metric: {evaluation_metric}")
        
        self._using_wasserstein: bool = True if evaluation_metric.find('wasserstein') != -1 else False
        self._metric = self.supported_metrics[evaluation_metric]
    
    
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        
        
        if self._using_wasserstein:
            if self.aggregation_method is not None:
                batch_size, n_samples, compressors, metrics = current_cluster.shape
                current_cluster = current_cluster.reshape(batch_size, n_samples, compressors * metrics).squeeze(0)
                current_sample = current_sample.reshape(batch_size, n_samples, compressors * metrics).squeeze(0)
            
            score = self._metric(
                x1=current_sample,
                x2=current_cluster
            )
            
            return score
        
        if self.aggregation_method is not None:
            score = self._metric(
                x1=current_sample,
                x2=current_cluster
            )
        
        else:
            scores = np.array(
                [
                    self._metric(x1=current_sample, x2=sample) 
                    for sample in current_cluster]
            )
            score = np.sum(scores)
        
        if hasattr(score, 'item'):
            score = score.item()
            
        else:
            score = float(score)
        
        return score