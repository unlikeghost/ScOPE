import numpy as np
from scipy.stats import gmean
from collections import Counter
from typing import Dict, Union, Any, List, Optional
from abc import abstractmethod, ABC


class _BasePredictor(ABC):
    start_key_value_cluster: str = 'ScOPE_Cluster_'
    start_key_value_sample: str = 'ScOPE_Sample_'
    start_key_sigma: str = 'ScOPE_Sigma'
    
    valid_methods = {
        "mean": lambda data: np.mean(data, axis=0),
        "median": lambda data: np.median(data, axis=0),
        "sum": lambda data: np.sum(data, axis=0),
        "gmean": lambda data: gmean(data)
    }
    
    def __init__(self,
                 epsilon: float = 1e-12,
                 aggregation_method: Optional[str] = None
                 ):
        
        self.epsilon = epsilon
        self.aggregation_method = aggregation_method
        
        if aggregation_method is not None and aggregation_method not in self.valid_methods:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}. "
                           f"Valid options: {self.valid_methods} or None for no aggregation")
    
    @staticmethod
    def __compute_gaussian_function__(x, sigma):
        return np.exp(-0.5 * np.square((x / sigma)))
    
    def __compute_aggregated_prototype__(self, data: np.ndarray) -> np.ndarray:
        """Compute prototype using specified aggregation method"""
        prototype = self.valid_methods[self.aggregation_method](
            data
        )
        prototype = np.expand_dims(prototype, axis=0)

        return prototype

    def __compute_probas__(self, dists: np.ndarray) -> np.ndarray:
        dists = np.array(dists, dtype=float)
        scores = 1.0 / (dists + self.epsilon) ** 2
        probas = scores / np.sum(scores)
        return probas
    
    @abstractmethod
    def __forward__(self, current_cluster: np.ndarray, current_sample: np.ndarray) -> float:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def forward(self, list_of_data: List[Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
        
        n_compressors = 1
        n_metrics = 1
        
        if not isinstance(list_of_data, list):
            raise ValueError("Input should be a list of dictionaries containing data matrices.")
        
        if not list_of_data:
            return []
        
        output: List[Dict[str, Any]] = []
        
        for data_matrix in list_of_data:
            
            cluster_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_cluster),
                    data_matrix.keys()
                )
            )

            sample_keys: list = list(
                filter(
                    lambda x: x.startswith(self.start_key_value_sample),
                    data_matrix.keys()
                )
            )
            
            this_output: Dict[str, Any] = {
                'scores': {
                    cluster_key[len(self.start_key_value_cluster):]: [0.0]
                    for cluster_key in cluster_keys
                },
                'probas': {
                    cluster_key[len(self.start_key_value_cluster):]: [0.0]
                    for cluster_key in cluster_keys
                },
                'predicted_class': None,
            }
                        
            for cluster_key in cluster_keys:

                real_cluster_name: str = cluster_key[len(self.start_key_value_cluster):]
                current_sample_key: str = list(
                    filter(
                        lambda x: x.endswith(real_cluster_name),
                        sample_keys)
                )[0]
                
                cluster_: np.ndarray = data_matrix[cluster_key]
                sample_: np.ndarray = data_matrix[current_sample_key]
                
                _, _, n_compressors, n_metrics = cluster_.shape
                
                if data_matrix.get(f'{self.start_key_sigma}_{real_cluster_name}'):
                    sigma = data_matrix[f'{self.start_key_sigma}_{real_cluster_name}']
                    cluster_ = self.__compute_gaussian_function__(
                        cluster_,
                        sigma
                    )
                    sample_ = self.__compute_gaussian_function__(
                        sample_,
                        sigma
                    )
                
                if self.aggregation_method is not None:
                    cluster_ = self.__compute_aggregated_prototype__(cluster_)
                                
                    current_score = self.__forward__(
                        cluster_,
                        sample_
                    )
                    this_class_scores = current_score
                                        
                else:
                    this_class_scores = []
                                        
                    for index_c in range(n_compressors):
                        for index_m in range(n_metrics):
                            current_score = self.__forward__(
                                cluster_[:, :, index_c, index_m],
                                sample_[:, :, index_c, index_m]
                            )
                            this_class_scores.append(current_score)
                            
                this_output['scores'][real_cluster_name] = this_class_scores
                
            scores = this_output['scores']
            classes = list(scores.keys())

            if self.aggregation_method is not None or (n_compressors == 1 and  n_metrics == 1):
                distances_values = np.array([scores[cls] for cls in classes])

                all_probas = np.apply_along_axis(self.__compute_probas__, 0, distances_values)

            else:
                distances_values = np.array([scores[cls] for cls in classes])

                min_indices = np.argmin(distances_values, axis=0)

                winning_classes = [classes[idx] for idx in min_indices]
                
                votes = Counter(winning_classes)
                
                all_probas = np.array([votes.get(cls, 0) / len(winning_classes) for cls in classes])

            this_output['probas'] = {
                classes[i]: all_probas[i].item()
                for i in range(len(classes))
            }
            
            this_output['predicted_class'] = classes[
                np.argmax(all_probas)
            ]

            output.append(this_output)
            
        return output

    def __call__(self, list_of_data: Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:

        if not list_of_data:
            return []

        if isinstance(list_of_data, dict):
            list_of_data = [list_of_data]
            
        return self.forward(list_of_data)