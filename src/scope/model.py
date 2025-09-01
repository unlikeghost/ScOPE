import json
import numpy as np
from copy import deepcopy
from typing import Union, List, Optional, Dict, Any

from .compression import CompressionMatrix
from .predictor import _BasePredictor, ScOPEPredictor


class ScOPE:
    
    def __init__(self,
                 evaluation_metric: str = 'squared_euclidean',
                 aggregation_method: Optional[str] = None,
                 compressor_names: Union[str, List[str]] = 'gzip',
                 compression_metric_names: Union[str, List[str]] = 'ncd',
                 compression_level: int = 9,
                 join_string: str = '',
                 get_sigma: bool = True,
                 n_jobs: int = 5,
                 ):
        
        self.predictor: _BasePredictor = ScOPEPredictor(
            aggregation_method=aggregation_method,
            evaluation_metric=evaluation_metric
        )
        
        self.compression_matrix: CompressionMatrix = CompressionMatrix(
            compressor_names=compressor_names,
            compression_metric_names=compression_metric_names,
            compression_level=compression_level,
            join_string=join_string,
            get_sigma=get_sigma,
            n_jobs=n_jobs
        )
        
        self._evaluation_metric = evaluation_metric
        self._aggregation_method = aggregation_method

        self._compressor_names = compressor_names
        self._compression_metric_names = compression_metric_names
        self._compression_level = compression_level
        self._join_string = join_string
        self._get_sigma = get_sigma
    
    def _coerce_kwargs_to_serializable(self, kwargs: dict) -> dict:
        try:
            json.dumps(kwargs)
            return deepcopy(kwargs)
        except Exception:
            return {k: str(v) for k, v in kwargs.items()}
        
    def to_dict(self) -> dict:
        
        params = {
            'compressor_names': self._compressor_names,
            'compression_metric_names': self._compression_metric_names,
            'compression_level': self._compression_level,
            'join_string': self._join_string,
            'get_sigma': self._get_sigma,
            'evaluation_metric': self._evaluation_metric,
            'aggregation_method': self._aggregation_method,
        }
        
        return params
    
    def __pre_forward__(self,
                        samples: Union[List[str], str],
                        kw_samples: Union[Dict[Union[int, str], str],
                                   List[Dict[Union[int, str], str]]]) ->  Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        
        list_of_data = self.compression_matrix(
            samples=samples,
            kw_samples=kw_samples
        )
        
        if list_of_data is None or not list_of_data:
            return []
        
        if list_of_data and isinstance(list_of_data, dict):
            list_of_data = [list_of_data]
        
        return list_of_data

    def forward(self, list_of_data: Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
                
        if len(list_of_data) == 0:
            return []
        
        predictions = self.predictor(list_of_data)
        
        return predictions
    
    def __call__(self, samples: Union[List[str], str],
                 kw_samples: Union[Dict[Union[int, str], str],
                                   List[Dict[Union[int, str], str]]]) ->  Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        
        list_of_data = self.__pre_forward__(
            samples=samples,
            kw_samples=kw_samples
        )
        
        return self.forward(list_of_data)