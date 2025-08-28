import os
import pickle
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict

from scope.model import ScOPE
from scope.utils.report_generation import make_report
from .params import ParameterSpace


class ScOPEOptimizer(ABC):
    """Updated ScOPE optimizer for the unified ScOPE class."""

    def __init__(self,
                 parameter_space: Optional[ParameterSpace] = None, 
                 free_cpu: int = 0,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_optimization",
                 output_path: str = "./results",
                 n_trials: int = 50,
                 target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                 use_cache: bool = True
                 ):
        
        self.parameter_space = parameter_space or ParameterSpace()
        
        _os_cpu = os.cpu_count() or 0
        
        self.n_jobs = max(1, _os_cpu - free_cpu)
        self.random_seed: int = random_seed
        self.cv_folds = cv_folds
        self.study_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_name = study_name
        self.output_path = output_path
        
        # Store results
        self.study = None
        self.best_params = None
        self.best_model = None
        
        self.n_trials = n_trials
        
        if isinstance(target_metric, str):
            self.target_metric_name = target_metric
            self.target_metric_weights = None
            self.is_combined = False
        elif isinstance(target_metric, dict):
            self.target_metric_name = 'combined'
            self.target_metric_weights = target_metric
            self.is_combined = True
            
            total = sum(target_metric.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Metric weights must sum to 1.0, got {total}")
        else:
            raise ValueError("target_metric must be str or dict")
        
        self.use_cache = use_cache
        self._eval_cache = {}
        
    def _make_cache_key(self, model):
        """Create cache key for model parameters."""
        # Use the model's to_dict method if available
        model_params = model.to_dict()
        
        key_str = json.dumps(model_params, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _eval_cache_size_(self) -> float:
        """Get current cache size."""
        return len(self._eval_cache)

    def print_parameter_space(self):
        """Print detailed parameter space information."""
        print("Parameter space includes:")
        print("=" * 60)
        
        # Basic parameters
        print("BASIC PARAMETERS:")
        print(f"  • Compressor combinations ({len(self.parameter_space.compressor_names_options)})")
        print(f"  • Compression metric combinations ({len(self.parameter_space.compression_metric_names_options)})")
        print(f"  • Join string options ({len(self.parameter_space.concat_value_options)}): {[repr(s) for s in self.parameter_space.concat_value_options]}")
        print(f"  • Get sigma options: {self.parameter_space.get_sigma_options}")
        print(f"  • Model types: {self.parameter_space.model_types_options}")
        print(f"  • Aggregation methods: {self.parameter_space.aggregation_method_options}")
        
        print("\nMODEL-SPECIFIC PARAMETERS:")
        
        # ScOPE-OT parameters
        print("  ScOPE-OT:")
        print(f"    • Matching metrics: {self.parameter_space.matching_metrics}")
        
        # ScOPE-PD parameters  
        print("  ScOPE-PD:")
        print(f"    • Distance metrics: {self.parameter_space.distance_metrics}")
        
        print("=" * 60)
        
    def create_model_from_params(self, params: Dict[str, Any]) -> ScOPE:
        """Create ScOPE model based on parameters."""
        
        # Convert string lists back to actual lists if needed
        compressor_names = params['compressor_names']
        if isinstance(compressor_names, str):
            compressor_names = compressor_names.split(',')
        
        compression_metric_names = params['compression_metric_names']
        if isinstance(compression_metric_names, str):
            compression_metric_names = compression_metric_names.split(',')
        
        # Handle aggregation_method None properly
        aggregation_method = params.get('aggregation_method')
        if aggregation_method == 'None' or aggregation_method == 'null' or aggregation_method == '':
            aggregation_method = None
        
        # Base parameters for ScOPE
        base_params = {
            'model_type': params['model_type'],
            'aggregation_method': aggregation_method,
            'compressor_names': compressor_names,
            'compression_metric_names': compression_metric_names,
            'join_string': params['join_string'],
            'get_sigma': params['get_sigma'],
        }
        
        # Model-specific parameters - handle None and invalid types correctly
        model_type = params['model_type']
        
        if model_type == "ot":
            matching_metric = params.get('matching_metric')
            # ScOPEOT expects either a valid string or None (not bool or other types)
            if matching_metric is not None:
                # Convert invalid types to None
                if not isinstance(matching_metric, str) or matching_metric == 'True' or matching_metric == 'False':
                    matching_metric = None
                # Pass None or valid string to the model
                base_params['matching_metric'] = matching_metric
            else:
                # Explicitly pass None
                base_params['matching_metric'] = None
                
        elif model_type == "pd":
            distance_metric = params.get('distance_metric')
            # Only add if it's a valid string (PD doesn't accept None)
            if distance_metric is not None and isinstance(distance_metric, str):
                base_params['distance_metric'] = distance_metric

        return ScOPE(**base_params)
    
    def suggest_categorical_params(self, trial) -> Dict[str, Any]:
        """Suggest categorical parameters."""
        aggregation_method = trial.suggest_categorical(
            'aggregation_method',
            self.parameter_space.aggregation_method_options
        )
        
        return {
            'join_string': trial.suggest_categorical(
                'join_string', 
                self.parameter_space.concat_value_options
            ),
            'model_type': trial.suggest_categorical(
                'model_type',
                self.parameter_space.model_types_options
            ),
            # Convertir string vacío a None
            'aggregation_method': None if aggregation_method == '' else aggregation_method
        }
        
    def suggest_boolean_params(self, trial) -> Dict[str, Any]:
        """Suggest boolean parameters."""
        return {
            'get_sigma': trial.suggest_categorical(
                'get_sigma', 
                self.parameter_space.get_sigma_options
            )
        }
    
    def suggest_integer_params(self, trial) -> Dict[str, Any]:
        """Suggest integer parameters using ranges."""
        return {}
    
    def suggest_compressor_and_metric_params(self, trial) -> Dict[str, Any]:
        """Suggest compressor and metric combinations."""
        compressor_choices = [','.join(combo) for combo in self.parameter_space.compressor_names_options]
        metric_choices = [','.join(combo) for combo in self.parameter_space.compression_metric_names_options]
        
        compressor_string = trial.suggest_categorical('compressor_names', compressor_choices)
        metric_string = trial.suggest_categorical('compression_metric_names', metric_choices)
        
        return {
            'compressor_names': compressor_string,
            'compression_metric_names': metric_string
        }
    
    def suggest_model_specific_params(self, trial, model_type: str) -> Dict[str, Any]:
            """Suggest model-specific parameters ONLY for the selected model type."""        
            params = {}
                    
            if model_type == "ot":
                # ONLY for OT: matching metric
                matching_metric = trial.suggest_categorical(
                    'matching_metric',
                    self.parameter_space.matching_metrics
                )
                params['matching_metric'] = None if matching_metric == '' else matching_metric
                
            elif model_type == "pd":
                # ONLY for PD: distance metric
                distance_metric = trial.suggest_categorical(
                    'distance_metric',
                    self.parameter_space.distance_metrics
                )
                if distance_metric != '':
                    params['distance_metric'] = distance_metric
            
            return params
            
    def suggest_all_params(self, trial) -> Dict[str, Any]:
        """Combine all parameter suggestions."""
        params = {}
        
        # Categorical parameters
        params.update(self.suggest_categorical_params(trial))
        
        # Boolean parameters
        params.update(self.suggest_boolean_params(trial))
        
        # Integer parameters (ranges)
        params.update(self.suggest_integer_params(trial))
        
        # Compressor and metric combinations
        params.update(self.suggest_compressor_and_metric_params(trial))
        
        # Model-specific parameters
        params.update(self.suggest_model_specific_params(trial, params['model_type']))

        return params
    
    def evaluate_model(self, 
                      model: ScOPE,
                      X_samples: List[str],
                      y_true: List[str],
                      kw_samples_list: List[Dict[str, Any]]
                      ) -> Dict[str, float]:
        """Evaluate the model using cross-validation."""
        
        if self.use_cache:
            key = self._make_cache_key(model)
            if key in self._eval_cache:
                return self._eval_cache[key]
        else:
            key = None
         
        indices = np.arange(len(X_samples))
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        
        cv_scores = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1': [],
            'f2': [],
            'auc_roc': [],
            'auc_pr': [],
            'log_loss': [],
            'mcc': []
        }
        
        unique_classes = sorted(list(set(y_true)))
        if len(unique_classes) != 2:
            raise ValueError(f"Expected exactly 2 classes, but found {len(unique_classes)}: {unique_classes}")
        
        class_to_idx = {unique_classes[0]: 0, unique_classes[1]: 1}
        
        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_true)):
                X_val = [X_samples[i] for i in val_idx]
                y_val = [y_true[i] for i in val_idx]
                kw_val = [kw_samples_list[i] for i in val_idx]
                
                y_pred = []
                y_pred_proba = []
                
                for sample, kw_sample in zip(X_val, kw_val):
                    try:
                        # Process one sample at a time (like the original)
                        predictions = model(samples=sample, kw_samples=kw_sample)
                        # Extract the first prediction from the list
                        if isinstance(predictions, list) and len(predictions) > 0:
                            prediction: dict = predictions[0]
                        else:
                            raise ValueError("Expected list of predictions")
                        
                        # Extract predicted class (already computed)
                        predicted_class_str = prediction.get('predicted_class', '0')
                        try:
                            predicted_class = int(predicted_class_str)
                        except (ValueError, TypeError):
                            predicted_class = 0
                        
                        # Extract probabilities (always available)
                        probs: dict = prediction.get('probas', {})
                        
                        # Ensure probabilities are ordered consistently
                        probs = OrderedDict(sorted(probs.items()))
                        proba_values = list(probs.values())
                        
                        # Validate probabilities
                        if len(proba_values) != 2:
                            raise ValueError(f"Expected 2 class probabilities, got {len(proba_values)}")
                        
                        y_pred.append(predicted_class)
                        y_pred_proba.append(proba_values)
                        
                    except Exception as e:
                        print('Some error on prediction come up')
                        print(e)
                       
                        # Handle error by using random predictions
                        random_pred = np.random.randint(0, 2)
                        y_pred.append(random_pred)
                        
                        # Generate random probabilities
                        random_proba = np.random.dirichlet([1, 1])
                        y_pred_proba.append(random_proba.tolist())

                y_val_numeric = np.array([class_to_idx[cls] for cls in y_val])
                y_pred_numeric = np.array(y_pred)
                y_pred_proba_array = np.array(y_pred_proba)

                if len(set(y_pred_numeric)) > 1 and len(set(y_val_numeric)) > 1:
                    try:
                        report = make_report(y_val_numeric, y_pred_numeric, y_pred_proba_array)
                        
                        cv_scores['accuracy'].append(report['accuracy'])
                        cv_scores['balanced_accuracy'].append(report['balanced_accuracy'])
                        cv_scores['f1'].append(report['f1'])
                        cv_scores['f2'].append(report['f2'])
                        cv_scores['auc_roc'].append(report['auc_roc'])
                        cv_scores['auc_pr'].append(report['auc_pr'])
                        cv_scores['log_loss'].append(report['log_loss'])
                        cv_scores['mcc'].append(report['mcc'])
                        
                    except Exception as e:
                        print(f"Error en make_report: {e}")
                        cv_scores['accuracy'].append(0.0)
                        cv_scores['balanced_accuracy'].append(0.0)
                        cv_scores['f1'].append(0.0)
                        cv_scores['f2'].append(0.0)
                        cv_scores['auc_roc'].append(0.5)
                        cv_scores['auc_pr'].append(0.5)
                        cv_scores['log_loss'].append(1.0)
                        cv_scores['mcc'].append(-1.0)
                else:
                    cv_scores['accuracy'].append(0.0)
                    cv_scores['balanced_accuracy'].append(0.0)
                    cv_scores['f1'].append(0.0)
                    cv_scores['f2'].append(0.0)
                    cv_scores['auc_roc'].append(0.5)
                    cv_scores['auc_pr'].append(0.5)
                    cv_scores['log_loss'].append(1.0)
                    cv_scores['mcc'].append(-1.0)
        
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return {
                'balanced_accuracy': 0.0,
                'accuracy': 0.0,
                'f1': 0.0,
                'f2': 0.0,
                'auc_roc': 0.5,
                'auc_pr': 0.5,
                'log_loss': 1.0,
                'mcc': -1.0
            }
            
        final_scores: dict = {
            metric: np.mean(scores) for metric, scores in cv_scores.items()
        }
        
        if self.use_cache and key is not None:
            self._eval_cache[key] = final_scores
        
        return final_scores

    def calculate_objective_score(self, scores: Dict[str, float]) -> float:
        """Calculate objective score."""
        
        if self.is_combined:
            combined_score = 0.0
            for metric, weight in self.target_metric_weights.items():
                if metric in scores:
                    if metric == 'log_loss':
                        combined_score += (1 - scores[metric]) * weight
                    elif metric == 'mcc':
                        normalized_score = (scores[metric] + 1) / 2
                        combined_score += normalized_score * weight
                    else:
                        combined_score += scores[metric] * weight
                    
            return combined_score
            
        elif self.target_metric_name == 'log_loss':
            return scores['log_loss']  # Minimize
        else:
            return scores[self.target_metric_name]  # Maximize

    def get_optimization_direction(self) -> str:
        """Determine optimization direction for Optuna."""
        if self.is_combined:
            return 'maximize'  # Combined always maximizes
        elif self.target_metric_name == 'log_loss':
            return 'minimize'
        elif self.target_metric_name == 'mcc':
            return 'maximize'
        else:
            return 'maximize'

    def _create_objective_function(self,
                                  X_validation: List[str],
                                  y_validation: List[str], 
                                  kw_samples_validation: List[Dict[str, Any]]):
        """Objective function for Optuna."""
        
        def objective(trial):
            try:
                params = self.suggest_all_params(trial)
            
                model = self.create_model_from_params(params)
                
                scores = self.evaluate_model(
                    model=model,
                    X_samples=X_validation,
                    y_true=y_validation,
                    kw_samples_list=kw_samples_validation
                )
                
                return self.calculate_objective_score(scores)
                
            except Exception as e:
                print(f"Error in trial {trial.number}: {e}")
                print(f"Trial params: {params if 'params' in locals() else 'Not available'}")
                import traceback
                traceback.print_exc()
                return 0.0 if self.get_optimization_direction() == 'maximize' else 10.0

        return objective

    def validate_data(self, y_validation: List[str]):
        """Common data validation."""
        unique_classes = set(y_validation)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, but found {len(unique_classes)} classes: {unique_classes}")
        
        print(f"Data validation passed. Classes: {sorted(unique_classes)}")

    def validate_optimization_setup(self):
        """Validate that the configuration is correct before optimizing."""
        if self.parameter_space is None:
            raise ValueError("Parameter space not defined")
        
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")
        
        print("Optimization setup validation passed")

    def load_results(self, filepath: str):
        """Load previous results."""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.study = results['study']
        self.best_params = results['best_params']
        self.best_model = self.create_model_from_params(self.best_params)
        
        print(f"Results loaded from {filepath}")

    def get_trials_dataframe(self):
        """Get DataFrame with all trials."""
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        data = self.study.trials_dataframe()
        
        df_sorted = data.sort_values(by='value', ascending=False)
        
        param_cols = [col for col in df_sorted.columns if col.startswith('params_')]

        df_unique = df_sorted.drop_duplicates(subset=param_cols, keep='first')
        
        return df_unique

    def print_best_configuration(self):
        """Print best configuration."""
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
        
        print("Best configuration:")
        for param, value in self.best_params.items():
            if param == 'join_string':
                value = repr(value)
            print(f"  {param}: {value}")

    def print_target_metric_info(self):
        """Print information about the configured target metric."""
        print("Target metric configuration:")
        if self.is_combined:
            print("  Type: Combined metric")
            print("  Weights:")
            for metric, weight in self.target_metric_weights.items():
                print(f"    {metric}: {weight:.3f}")
            print(f"  Optimization direction: {self.get_optimization_direction()}")
        else:
            print("  Type: Single metric")
            print(f"  Metric: {self.target_metric_name}")
            print(f"  Optimization direction: {self.get_optimization_direction()}")

    def get_best_model(self) -> ScOPE:
        """Get the best optimized model."""
        if self.best_model is None:
            raise ValueError("No optimized model found. Run optimize() first.")
        return self.best_model

    def save_results(self, filename: Optional[str] = None):
        """Save optimization results."""
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_results_{self.study_date}.pkl"
        
        os.makedirs(self.output_path, exist_ok=True)
        filepath = os.path.join(self.output_path, filename)
        
        results = {
            'study': self.study,
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'trials_dataframe': self.study.trials_dataframe(),
            'target_metric_info': {
                'is_combined': self.is_combined,
                'target_metric_name': self.target_metric_name,
                'target_metric_weights': self.target_metric_weights
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filepath}")
    
    @abstractmethod
    def optimize(self, 
                 X_validation: List[str],
                 y_validation: List[str], 
                 kw_samples_validation: List[Dict[str, Any]]) -> Any:
        """Main optimization method - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def analyze_results(self) -> Any:
        """Results analysis - must be implemented by subclasses."""
        pass