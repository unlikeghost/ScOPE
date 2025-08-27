from typing import List, Tuple
from dataclasses import dataclass, field
from itertools import chain, combinations

from scope.compression.dissimilarity import COMPRESSION_METRICS
from scope.compression.compressors import CompressorType



def all_subsets(elements: List[str]) -> List[List[str]]:
    """Genera todas las combinaciones posibles de tamaño 1 a N."""
    return list(
        map(list, chain.from_iterable(combinations(elements, r) for r in range(1, len(elements) + 1)))
    )


@dataclass
class ParameterSpace:
    compressor_names_options: List[List[str]] = field(
        default_factory=lambda: all_subsets([c.value for c in CompressorType])
    )

    compression_metric_names_options: List[List[str]] = field(
        default_factory=lambda: all_subsets([metric for metric in COMPRESSION_METRICS])
    )
    
    concat_value_options: List[str] = field(
        default_factory= lambda: [' ', '\n']
        # default_factory= lambda: ["|||SEP_SAFE_DELIM_SEP|||"]
    )
    
    model_types_options: List[str] = field(
        default_factory=lambda: ["ot", "pd"]
    )
    
    aggregation_method_options: List[str] = field(
        default_factory= lambda: ['mean', 'median', 'sum', 'average', 'gmean', '']
    )
    
    # Enteros
    compression_levels_range: Tuple[int] = field(
        default_factory=lambda: (1, 9)
    )

    # Booleanos
    get_sigma_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )

    # Específicos por tipo de modelo
    matching_metrics: List[str] = field(
        default_factory=lambda: ["jaccard", "dice", "overlap", '']
    )
    
    distance_metrics: List[str] = field(
        default_factory=lambda: ["manhattan", "euclidean", "squared_euclidean", "chebyshev"]
    )
