# -*- coding: utf-8 -*-
"""
    ScOPE
    Compression Matrix
    Jesus Alan Heernandez Galvan
"""
import warnings
import numpy as np
from copy import deepcopy
from itertools import product
from functools import lru_cache
from scipy.stats.mstats import hmean
from typing import Union, List, Dict, Tuple

from .compressors import compute_compression, get_compressor
from .dissimilarity import compute_compression_metric


class CompressionMatrix:
    epsilon: float = 1e-4
        
    compressors = {
        'bz2': 0,
        'zlib': 1,
        'zstd': 2,
        'rle': 3,
        'huffman': 4,
        'lz77': 5,
        'gzip': 6
    }

    compression_metrics = {
        'ncd': 0,
        'cdm': 1,
        'cd': 2,
        'ucd': 3,
        'ncc': 4,
        'nccd': 5
    }

    def __init__(self,
                 compressor_names: Union[str, List[str]] = 'gzip',
                 compression_metric_names: Union[str, List[str]] = 'ncd',
                 compression_level: int = 9,
                 join_string: str = "|||ScOPE_SEPARATOR_BOUNDARY|||",
                 get_sigma: bool = True,
                 ):
                     
        if isinstance(compressor_names, str):
            compressor_names = [compressor_names]

        if isinstance(compression_metric_names, str):
            compression_metric_names = [compression_metric_names]

        invalid_compressors = [c for c in compressor_names if c not in self.compressors]
        if invalid_compressors:
            raise ValueError(
                f"Invalid compressor(s): {', '.join(invalid_compressors)}. "
                f"Valid options are: {', '.join(self.compressors.keys())}"
            )

        invalid_metrics = [m for m in compression_metric_names if m not in self.compression_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid compression metric(s): {', '.join(invalid_metrics)}. "
                f"Valid options are: {', '.join(self.compression_metrics)}"
            )

        if not isinstance(join_string, str):
            raise ValueError(
                f"Invalid join string: {join_string}. join_string must be a string object."
            )

        self.compression_level = compression_level
        self.join_string = join_string
        self.compressor_names = set(compressor_names)
        self.compression_metric_names = set(compression_metric_names)
        self.get_sigma = get_sigma

    @lru_cache(maxsize=None)
    def __sequence_concatenation__(self, x1: Union[str, bytes], x2: Union[str, bytes]) -> bytes:
        
        x1 = x1 if isinstance(x1, bytes) else x1.encode('utf-8')
        
        x2 = x2 if isinstance(x2, bytes) else x2.encode('utf-8')
        
        return x1 + self.join_string.encode('utf-8') + x2
    
    @lru_cache(maxsize=None)
    def __apply_threshold_padding__(self, sequence: str, compressor: str) -> bytes:
        """Aplica padding al string si está por debajo del threshold del compresor"""
        compressor_instance = get_compressor(compressor, self.compression_level)
    
        sequence_bytes = sequence.encode('utf-8')
        padded_sequence_bytes = compressor_instance._apply_padding(sequence_bytes)
                
        return padded_sequence_bytes
    
    @lru_cache(maxsize=None)
    def __get_compression_size__(self, sequence: Union[str, bytes], compressor: str) -> int:
        """Comprime un string y retorna solo el tamaño"""
        if len(sequence) == 0:
            raise ValueError(f"WARNING: Empty sequence for compression with {compressor}")
                            
        compressed_sequence = compute_compression(
            sequence=sequence,
            compressor=compressor,
            compression_level=self.compression_level,
        )
        
        c_sequence = len(compressed_sequence)
                
        return min(len(sequence), c_sequence)
    
    @lru_cache(maxsize=None)
    def __compute_dissimilarity_metric__(self, x1: str, x2: str, compressor: str, metric: str) -> float:
        
        x1_ = self.__apply_threshold_padding__(sequence=x1, compressor=compressor)
        x2_ = self.__apply_threshold_padding__(sequence=x2, compressor=compressor)
        x1x2 = self.__sequence_concatenation__(x1=x1_, x2=x2_)
        x2x1 = self.__sequence_concatenation__(x1=x2_, x2=x1_)
        
        c_x1 = self.__get_compression_size__(x1_, compressor)
        c_x2 = self.__get_compression_size__(x2_, compressor)
        c_x1x2 = self.__get_compression_size__(x1x2, compressor)
        c_x2x1 = self.__get_compression_size__(x2x1, compressor)


        _score = compute_compression_metric(
            c_x1=c_x1,
            c_x2=c_x2,
            c_x1x2=c_x1x2,
            c_x2x1=c_x2x1,
            metric=metric
        )
        
        # assert _score >= 0, f"Expected disimilarity score >= 0, but got {_score}, x1:{x1}, x2:{x2}"

        return max(self.epsilon, _score)
                
    def compute_sigma(self, samples: List[str]) -> float:

        def _compute(x1: str) -> List[float]:
            sigmas: list = []
            for compressor in self.compressor_names:

                for metric in self.compression_metric_names:
                    _score = self.__compute_dissimilarity_metric__(
                        x1=x1,
                        x2=x1,
                        compressor=compressor,
                        metric=metric
                    )
                    
                    sigmas.append(max(_score, self.epsilon))

            return sigmas
        sigmas = np.array(
            list(map(_compute, samples))
        ).flatten()
        
        sigma = hmean(sigmas)
        
        return sigma.item()

    def compute_ovo(self, x1: str, x2: str) -> np.ndarray:

        matrix_values = np.full(
            shape=(
                len(self.compressors),
                len(self.compression_metrics),
            ),
            fill_value=np.nan
        )
        
        for compressor in self.compressor_names:
            compressor_index = self.compressors[compressor]
            
            for metric in self.compression_metric_names:
                metric_index = self.compression_metrics[metric]
                
                _score = self.__compute_dissimilarity_metric__(
                    x1=x1,
                    x2=x2,
                    compressor=compressor,
                    metric=metric
                )

                matrix_values[compressor_index, metric_index] = _score
            
        nan_mask = ~np.isnan(matrix_values)

        result = matrix_values[nan_mask].reshape(
            len(self.compressor_names),
            len(self.compression_metric_names)
        )

        return result

    def compute_ova(self, sample: str, cluster: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:

        sigma = float('inf')

        cluster = deepcopy(cluster)

        combinations = list(product(cluster, repeat=2))

        cluster_v_cluster = combinations[:-len(cluster)]

        sample_matrix = np.zeros(
            shape=(
                1,
                len(cluster),
                len(self.compressor_names),
                len(self.compression_metric_names),
            )
        )

        cluster_matrix = np.zeros(
            shape=(
                len(cluster) - 1,
                len(cluster),
                len(self.compressor_names),
                len(self.compression_metric_names),
            )
        )

        for index, kw_cluster_sample in enumerate(cluster):
            sample_matrix[0, index, :, :,] = self.compute_ovo(
                x1=sample,
                x2=kw_cluster_sample
            )

        for index, (x1, x2) in enumerate(cluster_v_cluster):
            kw_sample_index = index // (len(cluster))
            kw_sample_vs_index = index % len(cluster)
            cluster_matrix[kw_sample_index, kw_sample_vs_index, :, :] = self.compute_ovo(
                x1=x1,
                x2=x2
            )

        if self.get_sigma:
            sigma = self.compute_sigma(
                samples=cluster
            )

        return sample_matrix, cluster_matrix, sigma

    def get_one_compression_matrix(self, sample: str, kw_samples: Dict[Union[int, str], List[str]]) ->  Dict[str, np.ndarray]:

        if not isinstance(kw_samples, dict):
            raise ValueError(
                "kw_samples must be a dictionary."
            )

        str_keys = [k for k in kw_samples.keys() if isinstance(k, str)]

        if str_keys:
            warnings.warn(
                "String-type keys were detected in kw_samples. "
                "This may affect evaluation metrics. "
                "It is recommended to encode the keys as integers.",
                UserWarning
            )

        seq_lengths = [len(sample)]
        for v in kw_samples.values():
            if isinstance(v, str):
                seq_lengths.append(len(v))
            elif isinstance(v, list):
                seq_lengths.extend(len(s) for s in v if isinstance(s, str))

        results = {}

        for cluster_key, cluster_values in kw_samples.items():
            sample_matrix, cluster_matrix, sigma = self.compute_ova(
                sample=sample,
                cluster=cluster_values
            )

            results[f"ScOPE_Cluster_{cluster_key}"] = cluster_matrix
            results[f"ScOPE_Sample_{cluster_key}"] = sample_matrix

            if self.get_sigma:
                results[f'ScOPE_Sigma_{cluster_key}'] = sigma

        return results

    def get_multiple_compression_matrix(self, samples: List[str], kw_samples: List[Dict[Union[int, str], List[str]]]) ->  List[Dict[str, np.ndarray]]:

        results = []

        if len(samples) != len(kw_samples):
            raise ValueError(
                f"'samples' and 'kw_samples' must have the same length "
                f"(got {len(samples)} and {len(kw_samples)})."
            )

        for index, sample in enumerate(samples):
            compression_matrix = self.get_one_compression_matrix(
                sample=sample,
                kw_samples=kw_samples[index]
            )

            results.append(compression_matrix)

        return results

    def __call__(self,
                 samples: Union[List[str], str],
                 kw_samples: Union[Dict[Union[int, str], List[str]],
                                   List[Dict[Union[int, str], List[str]]]]) ->  Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(kw_samples, dict):
            kw_samples = [kw_samples]

        if len(samples) != len(kw_samples):
            raise ValueError(
                f"'samples' and 'kw_samples' must have the same length "
                f"(got {len(samples)} and {len(kw_samples)})."
            )

        if len(samples) == 1:
            return self.get_one_compression_matrix(
                sample=samples[0],
                kw_samples=kw_samples[0]
            )

        return self.get_multiple_compression_matrix(
                samples=samples,
                kw_samples=kw_samples
            )
