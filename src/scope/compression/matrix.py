# -*- coding: utf-8 -*-

import json
import warnings
import numpy as np
from copy import deepcopy
from itertools import product
from scipy.stats.mstats import hmean
from typing import Union, List, Dict, Optional, Tuple

from .compressors import compute_compression
from .metrics import compute_compression_metric


class CompressionMatrix:

    compressors = {'bz2': 0, 'zlib': 1, 'zstd': 2, 'rle': 3, 'huffman': 4, 'lz77': 5}

    compression_metrics = {'ncd': 0, 'cdm': 1, 'nrc': 2}

    def __init__(self,
                 compressors_names: Union[str, List[str]] = 'gzip',
                 compression_metric_names: Union[str, List[str]] = 'ncd',
                 compression_level: int = 9,
                 min_size_threshold: int = 0,
                 join_string: str = '',
                 get_sigma: bool = True,
                 qval: Optional[int] = -1
                 ):

        if qval == -1:
            qval = None

        if isinstance(compressors_names, str):
            compressors_names = [compressors_names]

        if isinstance(compression_metric_names, str):
            compression_metric_names = [compression_metric_names]

        invalid_compressors = [c for c in compressors_names if c not in self.compressors]
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

        if qval is not None:
            if not isinstance(qval, int) or qval < 0:
                raise ValueError(
                    f"Invalid qval: {qval}. qval must be a non-negative integer or -1"
                )

        if not isinstance(join_string, str):
            raise ValueError(
                f"Invalid join string: {join_string}. join_string must be a string object."
            )

        self.qval = qval
        self.compression_level = compression_level
        self.min_size_threshold = min_size_threshold
        self.join_string = join_string
        self.compressors_names = set(compressors_names)
        self.compression_metric_names = set(compression_metric_names)
        self.get_sigma = get_sigma


    @staticmethod
    def __find_ngrams__(input_sequence, n: int) -> list:
        """Extract n-grams from a sequence."""
        return list(
            zip(
                *[input_sequence[i:] for i in range(n)]
            )
        )
    
    def __get_compression_size__(self, sequence: str, compressor: str) -> int:
            if len(sequence) == 0:
                print(f"WARNING: Empty sequence for compression with {compressor}")
                return 0
            
            compressed_sequence = compute_compression(
                sequence=sequence,
                compressor=compressor,
                compression_level=self.compression_level,
                min_size_threshold=self.min_size_threshold
            )
            
            size = len(compressed_sequence)
            if size == 0:
                raise ValueError(
                    f"Compression resulted in zero-size output for compressor '{compressor}'. "
                    f"Original sequence length: {len(sequence)}. "
                    f"This may indicate an issue with the compressor configuration, "
                    f"min_size_threshold ({self.min_size_threshold}), or input data. "
                    f"Consider using a different compressor or adjusting parameters."
                )
                
            return size
            
    def __string_concatenation__(self, x1: str, x2: str, compressor: str) -> str:
        """Find concatenation order with best compression ratio for two sequences."""

        # sequences = [x1, x2]
        # best_size = float('inf')
        # best_concatenation = ""

        # for permutation in permutations(sequences):
        #     concatenated = self.join_string.join(permutation)
        #     size = self.__get_compression_size__(concatenated, compressor)

        #     if size < best_size:
        #         best_size = size
        #         best_concatenation = concatenated

        # return best_concatenation

        return self.join_string.join([x1, x2])

    def __multiset_to_string__(self, multiset: Union[List, set]) -> str:

        unique_items = set(multiset)

        sorted_items = sorted(unique_items, key=str)

        return self.join_string.join(json.dumps(item) for item in sorted_items)

    def __get_sequences__(self, sequence: str) -> List[str]:

        processed = []

        if isinstance(sequence, str):
            if self.qval == 0:
                # Split by words/spaces
                processed.append(sequence.split())
            elif self.qval == 1:
                # Character-level
                processed.append(list(sequence))
            else:
                processed.append(self.__find_ngrams__(sequence, self.qval))

        return processed[0]

    def compute_sigma(self, samples: List[str]) -> float:

        def _compute(x1: str) -> List[float]:
            sigmas: list = []
            for compressor in self.compressors_names:
                x1x2: str = self.__string_concatenation__(x1=x1, x2=x1, compressor=compressor)
                c_x1 = self.__get_compression_size__(x1, compressor)
                c_x2 = self.__get_compression_size__(x1, compressor)
                c_x1x2 = self.__get_compression_size__(x1x2, compressor)

                for metric in self.compression_metric_names:
                    values = compute_compression_metric(
                        c_x1=c_x1,
                        c_x2=c_x2,
                        c_x1x2=c_x1x2,
                        metric=metric
                    )
                    sigmas.append(values)

            return sigmas
        sigmas = np.array(
            list(map(_compute, samples))
        ).flatten()

        sigma = hmean(sigmas)
        
        return sigma

    def compute_ovo(self, x1: str, x2: str) -> np.ndarray:

        matrix_values = np.full(
            shape=(
                len(self.compressors),
                len(self.compression_metrics),
            ),
            fill_value=np.nan
        )

        if self.qval is not None:
            processed_query = self.__get_sequences__(x1)
            processed_class = self.__get_sequences__(x2)

            x1 = self.__multiset_to_string__(processed_query)
            x2 = self.__multiset_to_string__(processed_class)

        for compressor in self.compressors_names:
            compressor_index = self.compressors[compressor]

            x1x2: str = self.__string_concatenation__(x1=x1, x2=x2, compressor=compressor)

            c_x1 = self.__get_compression_size__(x1, compressor)
            c_x2 = self.__get_compression_size__(x2, compressor)
            c_x1x2 = self.__get_compression_size__(x1x2, compressor)

            for metric in self.compression_metric_names:
                metric_index = self.compression_metrics[metric]

                matrix_values[compressor_index, metric_index] = compute_compression_metric(
                    c_x1=c_x1,
                    c_x2=c_x2,
                    c_x1x2=c_x1x2,
                    metric=metric
                )

        nan_mask = ~np.isnan(matrix_values)

        result = matrix_values[nan_mask].reshape(
            len(self.compressors_names),
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
                len(self.compressors_names),
                len(self.compression_metric_names),
            )
        )

        cluster_matrix = np.zeros(
            shape=(
                len(cluster) - 1,
                len(cluster),
                len(self.compressors_names),
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

    def get_one_compression_matrix(self, sample: str, kw_samples: Dict[Union[int, str], str]) ->  Dict[str, np.ndarray]:

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

        min_len = min(seq_lengths)

        if self.qval and self.qval > min_len:
            warnings.warn(
                f"qval ({self.qval}) is greater than the smallest sequence length ({min_len}). "
                f"Reducing qval to {min_len}.",
                UserWarning
            )
            self.qval = min_len

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

    def get_multiple_compression_matrix(self, samples: List[str], kw_samples: List[Dict[Union[int, str], str]]) ->  List[Dict[str, np.ndarray]]:

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
                 kw_samples: Union[Dict[Union[int, str], str],
                                   List[Dict[Union[int, str], str]]]) ->  Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:

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
