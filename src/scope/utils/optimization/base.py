import os
import pickle
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna

from scope.model import ScOPE
from scope.utils.report_generation import make_report
from .params import ParameterSpace


def _evaluate_single_fold(fold_data: Tuple, model_params: Dict, class_to_idx: Dict, unique_classes: List) -> Dict[str, float]:
    """
    Evaluate a single fold in parallel - handles batch processing.
    This function will be executed in a separate process/thread.
    """
    fold_idx, X_val, y_val, kw_val = fold_data
    model = ScOPE(**model_params)
    fold_predictions = []
    fold_probas = []
    predictions_batch = model(samples=X_val, kw_samples=kw_val)
    try:        
        
        for i, predictions in enumerate(predictions_batch):
            try:
                if isinstance(predictions, dict):
                    prediction = predictions
                elif isinstance(predictions, list) and len(predictions) > 0:
                    prediction = predictions[0]
                else:
                    raise ValueError("Unexpected prediction format")
                
                predicted_class_str = prediction.get('predicted_class', '0')
                try:
                    predicted_class = int(predicted_class_str)
                except (ValueError, TypeError):
                    predicted_class = unique_classes[0]
                
                probs: dict = prediction.get('probas', {})
                probs = OrderedDict(sorted(probs.items()))
                proba_values = list(probs.values())
                
                if len(proba_values) != 2:
                    raise ValueError(f"Expected 2 class probabilities, got {len(proba_values)}")
                
                fold_predictions.append(predicted_class)
                fold_probas.append(proba_values)
                
            except Exception as e:
                print(f"Warning: Prediction failed for fold {fold_idx}, sample {i}: {e}")
                fold_predictions.append(unique_classes[0])
                fold_probas.append([0.5, 0.5])

        # Process results
        y_val_numeric = [class_to_idx[y] for y in y_val[:len(fold_predictions)]]
        y_pred_numeric = [class_to_idx.get(pred, 0) for pred in fold_predictions]
        y_pred_proba_array = np.array(fold_probas)
        
        fold_scores = {
            'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1': 0.0, 'f2': 0.0,
            'auc_roc': 0.5, 'auc_pr': 0.5, 'log_loss': 1.0, 'mcc': -1.0
        }
        
        if len(set(y_pred_numeric)) > 1 and len(set(y_val_numeric)) > 1:
            try:
                report = make_report(y_val_numeric, y_pred_numeric, y_pred_proba_array)
                fold_scores = {
                    'accuracy': report['accuracy'],
                    'balanced_accuracy': report['balanced_accuracy'],
                    'f1': report['f1'],
                    'f2': report['f2'],
                    'auc_roc': report['auc_roc'],
                    'auc_pr': report['auc_pr'],
                    'log_loss': report['log_loss'],
                    'mcc': report['mcc']
                }
            except Exception as e:
                print(f"Warning: Metric computation failed for fold {fold_idx}: {e}")
        
        return fold_scores
        
    except Exception as e:
        print(f"Error in fold {fold_idx}: {e}")
        return {
            'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1': 0.0, 'f2': 0.0,
            'auc_roc': 0.5, 'auc_pr': 0.5, 'log_loss': 1.0, 'mcc': -1.0
        }


class ScOPEOptimizer(ABC):
    """Updated ScOPE optimizer for the unified ScOPE class."""

    def __init__(self,
                 parameter_space: Optional[ParameterSpace] = None, 
                 n_jobs: int = 1,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_optimization",
                 output_path: str = "./results",
                 n_trials: int = 50,
                 target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                 use_cache: bool = True
                 ):
        
        self.parameter_space = parameter_space or ParameterSpace()
        
        
        self.n_jobs = n_jobs
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
        self._cache_max_size = 1000  # Límite para evitar memory leaks
        
    def _make_cache_key(self, model):
        """Create cache key for model parameters."""
        model_params = model.to_dict()
        key_str = json.dumps(model_params, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _eval_cache_size_(self) -> int:
        """Get current cache size."""
        return len(self._eval_cache)

    def print_parameter_space(self):
        """Print detailed parameter space information."""
        print("Parameter space includes:")
        print("=" * 60)
        
        print("BASIC PARAMETERS:")
        print(f"  • Compressor combinations ({len(self.parameter_space.compressor_names_options)})")
        print(f"  • Compression metric combinations ({len(self.parameter_space.compression_metric_names_options)})")
        print(f"  • Join string options ({len(self.parameter_space.concat_value_options)}): {[repr(s) for s in self.parameter_space.concat_value_options]}")
        print(f"  • Get sigma options: {self.parameter_space.get_sigma_options}")
        print(f"  • Aggregation methods: {self.parameter_space.aggregation_method_options}")
        print(f"  • Evaluation Metrics ({len(self.parameter_space.evaluation_metrics)})")
        
        print("=" * 60)
        
    def create_model_from_params(self, params: Dict[str, Any]) -> ScOPE:
        """Create ScOPE model based on parameters."""
        
        compressor_names = params['compressor_names']
        if isinstance(compressor_names, str):
            compressor_names = compressor_names.split(',')
        
        compression_metric_names = params['compression_metric_names']
        if isinstance(compression_metric_names, str):
            compression_metric_names = compression_metric_names.split(',')
        
        aggregation_method = params.get('aggregation_method')
        if aggregation_method == 'None' or aggregation_method == 'null' or aggregation_method == '':
            aggregation_method = None
        
        base_params = {
            'aggregation_method': aggregation_method,
            'evaluation_metric': params['evaluation_metric'],
            'compressor_names': compressor_names,
            'compression_metric_names': compression_metric_names,
            'join_string': params['join_string'],
            'get_sigma': params['get_sigma'],
            'n_jobs': 4
        }
        
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
            'evaluation_metric': trial.suggest_categorical(
                'evaluation_metric',
                self.parameter_space.evaluation_metrics
            ),
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
            
    def suggest_all_params(self, trial) -> Dict[str, Any]:
        """Combine all parameter suggestions."""
        params = {}
        
        params.update(self.suggest_categorical_params(trial))
        params.update(self.suggest_boolean_params(trial))
        params.update(self.suggest_integer_params(trial))
        params.update(self.suggest_compressor_and_metric_params(trial))

        return params
    
    def evaluate_model(self,
                       model: ScOPE,
                       X_samples: List[str],
                       y_true: List[str],
                       kw_samples_list: List[Dict[str, Any]]
                       ) -> Dict[str, float]:
        """Evaluate the model using cross-validation (parallel folds)."""
    
        if self.use_cache:
            key = self._make_cache_key(model)
            if key in self._eval_cache:
                return self._eval_cache[key]
        else:
            key = None
    
        indices = np.arange(len(X_samples))
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
    
        unique_classes = sorted(list(set(y_true)))
        if len(unique_classes) != 2:
            raise ValueError(f"Expected exactly 2 classes, but found {len(unique_classes)}: {unique_classes}")
    
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
        # Guardamos los parámetros del modelo para instanciar dentro de cada proceso
        model_params = model.to_dict() if hasattr(model, "to_dict") else {}
    
        # Preparar folds
        fold_data_list = []
        for fold_idx, (_, val_idx) in enumerate(skf.split(indices, y_true)):
            X_val = [X_samples[i] for i in val_idx]
            y_val = [y_true[i] for i in val_idx]
            kw_val = [kw_samples_list[i] for i in val_idx]
            fold_data_list.append((fold_idx, X_val, y_val, kw_val))
    
        # Ejecutar en paralelo
        results = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_evaluate_single_fold, fold_data, model_params, class_to_idx, unique_classes): fold_data[0]
                for fold_data in fold_data_list
            }
            for future in as_completed(futures):
                fold_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Fold {fold_idx} failed: {e}")
                    results.append({
                        'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1': 0.0, 'f2': 0.0,
                        'auc_roc': 0.5, 'auc_pr': 0.5, 'log_loss': 1.0, 'mcc': -1.0
                    })
    
        # Promediar resultados
        final_scores = {
            metric: np.mean([res[metric] for res in results])
            for metric in results[0]
        }
    
        if self.use_cache and key is not None:
            if len(self._eval_cache) > self._cache_max_size:
                keep_size = int(self._cache_max_size * 0.8)
                keys_to_keep = list(self._eval_cache.keys())[-keep_size:]
                self._eval_cache = {k: self._eval_cache[k] for k in keys_to_keep}
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
            return scores['log_loss']
        else:
            return scores[self.target_metric_name]

    def get_optimization_direction(self) -> str:
        """Determine optimization direction for Optuna."""
        if self.is_combined:
            return 'maximize'
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

    def load_results(self, study_name: Optional[str] = None):
        """Load previous results from SQLite database."""
        import optuna
        from optuna.storages import RDBStorage
        
        if study_name:
            storage = RDBStorage(f"sqlite:///{self.output_path}/optuna_{study_name}.sqlite3")
            target_study_name = study_name
        else:
            # Assume storage is already set up in subclass
            storage = getattr(self, 'storage', None)
            target_study_name = self.study_name
            
            if storage is None:
                raise ValueError("No storage configured. Cannot load results.")
        
        try:
            self.study = optuna.load_study(
                study_name=target_study_name,
                storage=storage
            )
            
            self.best_params = self.study.best_params
            self.best_model = self.create_model_from_params(self.best_params)
            
            print(f"Study loaded from SQLite: {len(self.study.trials)} trials")
            print(f"Best value: {self.study.best_value}")
            
        except Exception as e:
            raise ValueError(f"Could not load study: {e}")

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
        """Save only metadata - SQLite has the full study data."""
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_metadata_{self.study_date}.pkl"
        
        os.makedirs(self.output_path, exist_ok=True)
        filepath = os.path.join(self.output_path, filename)
        
        # Solo guardar metadatos que NO están en SQLite
        results = {
            'study_name': self.study_name,
            'study_date': self.study_date,
            'n_trials_completed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'target_metric_info': {
                'is_combined': self.is_combined,
                'target_metric_name': self.target_metric_name,
                'target_metric_weights': self.target_metric_weights
            },
            'parameter_space_config': {
                'compressor_names_options': self.parameter_space.compressor_names_options,
                'compression_metric_names_options': self.parameter_space.compression_metric_names_options,
                'model_evaluation_metrics': self.parameter_space.evaluation_metrics
            },
            'sqlite_path': f"{self.output_path}/optuna_{self.study_name}.sqlite3"
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Metadata saved to {filepath}")
        print(f"Full study data in SQLite: {results['sqlite_path']}")
    
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