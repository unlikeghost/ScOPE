from typing import Type, Dict, Any, Optional, List

from .pd import ScOPEPD
from .ot import ScOPEOT
from .base import _BasePredictor

_EPSILON = 1e-12


class PredictorRegistry:
    
    _predictors: Dict[str, Type[_BasePredictor]] = {}
    _defaults: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[_BasePredictor],defaults: Dict[str, Any] = None):
        cls._predictors[name] = model_class
        cls._defaults[name] = defaults or {}
    
    
    @classmethod
    def create(cls,
               name: str,
               epsilon: float = _EPSILON,
               aggregation_method: Optional[str] = None,
               **kwargs) -> _BasePredictor:
        
        if name not in cls._predictors:
            available = list(cls._predictors.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        config = cls._defaults[name].copy()
        
        config['aggregation_method'] = aggregation_method
        config['epsilon'] = epsilon
        
        config.update(kwargs)
        return cls._predictors[name](**config)
    
    @classmethod
    def available(cls) -> List[str]:
        return list(cls._predictors.keys())
    

PredictorRegistry.register(
    name="ot",
    model_class=ScOPEOT,
    defaults={
        "matching_metric": None,
        "epsilon": _EPSILON
    }
)

PredictorRegistry.register(
    name="pd",
    model_class=ScOPEPD,
    defaults={
        "distance_metric": "euclidean",
        "epsilon": _EPSILON
    }
)