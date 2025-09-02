import os
import warnings
import pandas as pd
from typing import List, Dict, Optional, Any, Union

import optuna
from optuna.storages import RDBStorage

from .base import ScOPEOptimizer
from .params import ParameterSpace

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


class ScOPEOptimizerRandom(ScOPEOptimizer):
    """Random sampler optimization for ScOPE models using Optuna's RandomSampler."""
    
    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None,
                 n_jobs: int = 1,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_random_optimization",
                 output_path: str = "./results",
                 n_trials: int = 75,
                 target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                 ):
        """Initialize the Random sampler optimizer."""
        super().__init__(
            parameter_space=parameter_space, 
            n_jobs=n_jobs,
            random_seed=random_seed,
            cv_folds=cv_folds,
            study_name=study_name,
            output_path=output_path,
            n_trials=n_trials,
            target_metric=target_metric,
        )
        
        os.makedirs(self.output_path, exist_ok=True)
                
        self.storage = RDBStorage(
            f"sqlite:///{self.output_path}/optuna_{self.study_name}.sqlite3",
            engine_kwargs={"connect_args": {"timeout": 20.0}}
        )
    
    def optimize(self,
                X_validation: List[str],
                y_validation: List[str],
                kw_samples_validation: List[Dict[str, Any]]) -> optuna.Study:
        """Run random sampler optimization"""
        
        print("=== RANDOM SAMPLER OPTIMIZATION SCOPE ===\n")
        print(f"Validation data: {len(X_validation)} samples")
        print(f"Classes: {sorted(set(y_validation))}")
        print(f"Trials: {self.n_trials}")
        print(f"CV Folds: {self.cv_folds}")
        print()
                
        self.validate_data(y_validation)
        self.validate_optimization_setup()
        
        self.print_target_metric_info()
        print()
        
        self.print_parameter_space()
        
        objective_func = self._create_objective_function(
            X_validation, y_validation, kw_samples_validation
        )
        
        direction = self.get_optimization_direction()
        
        random_sampler = optuna.samplers.RandomSampler(seed=self.random_seed)
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=random_sampler,
            study_name=f"{self.study_name}_{self.study_date}",
            storage=self.storage,
            load_if_exists=True
        )
        
        print("\nStarting RandomSampler optimization...")
        
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs
        )
        
        self.best_params = self.study.best_params
        self.best_model = self.create_model_from_params(self.best_params)
        
        print("\n=== RANDOM SAMPLER OPTIMIZATION RESULTS ===")
        print(f"Best score: {self.study.best_value:.4f}")
         
        return self.study
    
    def _analyze_random_sampler_performance(self):
        """Analyze Random Sampler's performance and exploration patterns"""
        return  # Removed verbose analysis
    
    def analyze_results(self):
        """Analyze Random Sampler optimization results"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        df_results = self.study.trials_dataframe()
        return df_results
            
    def save_analysis_report(self, filename: Optional[str] = None):
        """Save detailed Random Sampler analysis report to text file"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_random_analysis_{self.study_date}.txt"

        filepath = os.path.join(self.output_path, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCOPE RANDOM SAMPLER OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("STUDY INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Study name: {self.study_name}\n")
            f.write(f"Study date: {self.study_date}\n")
            f.write(f"Output path: {self.output_path}\n")
            f.write("Sampler: RandomSampler (uniform random exploration)\n\n")
            
            f.write("RANDOM SAMPLER CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write("Strategy: Pure random exploration with uniform distribution\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write("Parameter selection: Independent uniform random sampling\n")
            f.write("Exploitation: None (purely exploratory)\n\n")
            
            f.write("OPTIMIZATION CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            
            if self.is_combined:
                f.write("Target metric: Combined\n")
                f.write("Metric weights:\n")
                for metric, weight in self.target_metric_weights.items():
                    f.write(f"  {metric}: {weight:.3f}\n")
            else:
                f.write(f"Target metric: {self.target_metric_name}\n")
            
            f.write(f"Optimization direction: {self.get_optimization_direction()}\n")
            f.write(f"Number of trials: {self.n_trials}\n")
            f.write(f"CV folds: {self.cv_folds}\n")
            f.write(f"Best score achieved: {self.study.best_value:.6f}\n\n")
            
            f.write("RANDOM SAMPLER STRATEGY:\n")
            f.write("-" * 30 + "\n")
            f.write("Random Sampler characteristics:\n")
            f.write("• Pure exploration: No learning from previous trials\n") 
            f.write("• Uniform distribution: All parameter combinations equally likely\n")
            f.write("• Independence: Each trial completely independent\n")
            f.write("• Baseline performance: Useful for comparison with smarter algorithms\n")
            f.write("• No convergence: Performance doesn't improve over time systematically\n\n")
            
            f.write("Expected behavior:\n")
            f.write("• Consistent variance across all phases\n")
            f.write("• No systematic improvement in later trials\n")
            f.write("• Good parameter space coverage given enough trials\n")
            f.write("• Performance depends purely on parameter space quality\n\n")
            
            f.write("PARAMETER SPACE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Compressor combinations: {len(self.parameter_space.compressor_names_options)}\n")
            f.write(f"Compression metric combinations: {len(self.parameter_space.compression_metric_names_options)}\n")
            f.write(f"Join string options: {self.parameter_space.concat_value_options}\n")
            f.write(f"Get sigma options: {self.parameter_space.get_sigma_options}\n")
            f.write(f"Model evaluation metrics: {self.parameter_space.evaluation_metrics}\n")
            f.write(f"Aggregation methods: {self.parameter_space.aggregation_method_options}\n")
            
            # Add performance analysis if available
            if self.study.trials:
                completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if completed_trials:
                    values = [t.value for t in completed_trials if t.value is not None]
                    if values:
                        f.write(f"\nPERFORMANCE ANALYSIS:\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"Mean performance: {sum(values)/len(values):.6f}\n")
                        f.write(f"Standard deviation: {pd.Series(values).std():.6f}\n")
                        f.write(f"Best value: {max(values) if self.get_optimization_direction() == 'maximize' else min(values):.6f}\n")
                        f.write(f"Worst value: {min(values) if self.get_optimization_direction() == 'maximize' else max(values):.6f}\n")
                        f.write(f"Total range explored: {max(values) - min(values):.6f}\n")

        print(f"Random Sampler analysis report saved to {filepath}")
    
    def save_top_parameters(self, top_percent: float = 10.0, filename: Optional[str] = None):
        """Save top N% of parameters to JSON file"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            print("No completed trials found.")
            return
        
        # Sort trials by value (best first)
        if self.get_optimization_direction() == 'maximize':
            sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        else:
            sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        
        # Calculate how many trials to include
        n_top_trials = max(1, int(len(sorted_trials) * (top_percent / 100.0)))
        top_trials = sorted_trials[:n_top_trials]
        
        # Extract parameters and values
        top_params_data = []
        for i, trial in enumerate(top_trials):
            trial_data = {
                'rank': i + 1,
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params.copy()
            }
            top_params_data.append(trial_data)
        
        # Save to JSON file
        if filename is None:
            filename = f"{self.study_name}_top_{top_percent}percent_params_{self.study_date}.json"
        
        filepath = os.path.join(self.output_path, filename)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(top_params_data, f, indent=2, default=str)
        
        print(f"\nTop {top_percent}% parameters saved:")
        print(f"  File: {filepath}")
        print(f"  Trials included: {n_top_trials}/{len(sorted_trials)}")
        print(f"  Best value: {top_trials[0].value:.6f}")
        print(f"  Worst in top {top_percent}%: {top_trials[-1].value:.6f}")
        
        return filepath
    
    def extract_promising_parameter_space(self, top_percent: float = 20.0) -> Dict[str, Any]:
        """Extract promising parameter values from top performing trials"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            return None
        
        # Sort trials by performance
        if self.get_optimization_direction() == 'maximize':
            sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        else:
            sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        
        # Get top N% trials
        n_top_trials = max(1, int(len(sorted_trials) * (top_percent / 100.0)))
        top_trials = sorted_trials[:n_top_trials]
        
        # Extract parameter values from top trials
        promising_space = {}
        
        # Collect all parameter values from top trials
        param_values = {}
        for trial in top_trials:
            for param_name, param_value in trial.params.items():
                if param_name not in param_values:
                    param_values[param_name] = set()
                param_values[param_name].add(param_value)
        
        # Convert sets to lists for the promising space
        for param_name, values in param_values.items():
            promising_space[param_name] = list(values)
        
        print(f"\nExtracted promising parameter space from top {top_percent}% trials:")
        print(f"  Trials analyzed: {n_top_trials}/{len(completed_trials)}")
        print(f"  Performance range: {top_trials[-1].value:.4f} to {top_trials[0].value:.4f}")
        
        for param_name, values in promising_space.items():
            reduction = (1 - len(values) / len(self.parameter_space.__dict__.get(param_name + '_options', values))) * 100
            print(f"  {param_name}: {len(values)} options ({reduction:.1f}% reduction)")
        
        return promising_space
    
    def save_complete_analysis(self, top_percent: float = 10.0):
        """Save complete analysis files for Random Sampler"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        self.save_results()
        self.save_analysis_report()
        self.save_top_parameters(top_percent=top_percent)
        
        print("\nRandom Sampler analysis saved:")
        print(f"Directory: {self.output_path}/")
        print(f"  optuna_{self.study_name}.sqlite3 (FULL DATA)")
        print(f"  {self.study_name}_metadata_{self.study_date}.pkl (config only)")
        print(f"  {self.study_name}_random_analysis_{self.study_date}.txt (human readable report)")
        print(f"  {self.study_name}_top_{top_percent}percent_params_{self.study_date}.json (top parameters)")
        print(f"\nTo view results: optuna-dashboard sqlite:///{self.output_path}/optuna_{self.study_name}.sqlite3")