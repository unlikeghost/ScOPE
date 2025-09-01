from typing import List
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
        default_factory= lambda: [' ', '||SEP_SAFE_DELIM_SEP|||']
    )
    
    aggregation_method_options: List[str] = field(
        default_factory= lambda: ['mean', 'median', 'sum', 'gmean', '']
    )

    get_sigma_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )

    evaluation_metrics: List[str] = field(
        default_factory=lambda: ["squared_euclidean", "wasserstein", "wasserstein-jaccard", "wasserstein-dice", "wasserstein-overlap"]
    )
    
