# -*- coding: utf-8 -*-
"""
    ScOPE
    Compression Matrix
    Jesus Alan Heernandez Galvan
"""
import warnings
import numpy as np
from itertools import product
from scipy.stats.mstats import hmean
from typing import Union, List, Dict, Tuple

from .compressors import compute_compression
from .dissimilarity import compute_compression_metric


class CompressionMatrix:
    epsilon: float = 1e-4
        
    compressors = {
        'bz2': 0,
        'zlib': 1,
        'rle': 2,
        'lz77': 3,
        'gzip': 4,
        'smilez': 5,
        'smaz': 6
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
        
        self._total_compressors = len(self.compressors)
        self._total_metrics = len(self.compression_metrics)
        
        self._n_compressors = len(self.compressor_names)
        self._n_metrics = len(self.compression_metric_names)

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

        return c_sequence
    
    def __compute_dissimilarity_metric__(self, c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: float, metric: str) -> float:

        _score = compute_compression_metric(
            c_x1=c_x1,
            c_x2=c_x2,
            c_x1x2=c_x1x2,
            c_x2x1=c_x2x1,
            metric=metric
        )
        
        if _score <= 0:
            warnings.warn(
                f"Expected disimilarity score <= 0, but got {_score}"
                f"with metric {metric}",
                category=UserWarning
            )

        return _score
                
    def compute_sigma(self, samples: List[str]) -> float:

        def _compute(x1: str) -> List[float]:
            sigmas: list = []
            x1x2 = self.join_string.join([x1, x1])
            
            for compressor in self.compressor_names:
                c_x1 = self.__get_compression_size__(x1, compressor)
                c_x1x2 = self.__get_compression_size__(x1x2, compressor)
                
                for metric in self.compression_metric_names:
                    _score = self.__compute_dissimilarity_metric__(
                        c_x1=c_x1,
                        c_x2=c_x1,
                        c_x1x2=c_x1x2,
                        c_x2x1=c_x1x2,
                        metric=metric
                    )
                    
                    sigmas.append(max(_score, self.epsilon))

            return sigmas
        sigmas = np.array([_compute(sample) for sample in samples]).flatten()
        
        sigma = hmean(sigmas)
        
        return sigma.item()

    def compute_ovo(self, x1: str, x2: str) -> np.ndarray:

        matrix_values = np.full(
            shape=(
                self._total_compressors,
                self._total_metrics,
            ),
            fill_value=np.nan
        )
        
        x1x2 = self.join_string.join([x1, x2])
        x2x1 = self.join_string.join([x2, x1])
        
        sequences = [x1, x2, x1x2, x2x1]
        
        for compressor in self.compressor_names:
            compressor_index = self.compressors[compressor]
            compressed_sizes = [
                len(compute_compression(seq, compressor, self.compression_level))
                for seq in sequences
            ]
            c_x1, c_x2, c_x1x2, c_x2x1 = compressed_sizes
            
            for metric in self.compression_metric_names:
                metric_index = self.compression_metrics[metric]
                
                _score = self.__compute_dissimilarity_metric__(
                    c_x1=c_x1,
                    c_x2=c_x2,
                    c_x1x2=c_x1x2,
                    c_x2x1=c_x2x1,
                    metric=metric
                )
                
                matrix_values[compressor_index, metric_index] = _score
            
        nan_mask = ~np.isnan(matrix_values)

        result = matrix_values[nan_mask].reshape(
            self._n_compressors,
            self._n_metrics
        )

        return result

    def compute_ova(self, sample: str, cluster: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:

        sigma = float('inf')

        combinations = list(product(cluster, repeat=2))

        cluster_v_cluster = combinations[:-len(cluster)]

        sample_matrix = np.zeros(
            shape=(
                1,
                len(cluster),
                self._n_compressors,
                self._n_metrics,
            )
        )

        cluster_matrix = np.zeros(
            shape=(
                len(cluster) - 1,
                len(cluster),
                self._n_compressors,
                self._n_metrics,
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

        if len(samples) != len(kw_samples):
            raise ValueError(
                f"'samples' and 'kw_samples' must have the same length "
                f"(got {len(samples)} and {len(kw_samples)})."
            )
            
        return [
            self.get_one_compression_matrix(sample, kw_sample)
            for sample, kw_sample in zip(samples, kw_samples)
        ]

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
