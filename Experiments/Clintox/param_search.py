import os
import sys
import random
import tomllib
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
import multiprocessing as mp
from collections import OrderedDict

from scope.compression.compressors import CompressorType
from scope.utils.optimization.params import ParameterSpace, all_subsets
from scope.utils import make_report, SampleGenerator, ScOPEOptimizerAuto


def setup_parameters():
    """Setup parameter space for optimization"""
    compressors = all_subsets([c.value for c in CompressorType if c != CompressorType.SMILEZ])
    dissimilarity_metrics = all_subsets([
        'ncd', 'ucd', 'cd', 'ncc'
    ])
    
    params = ParameterSpace(
        compressor_names_options=compressors,
        compression_metric_names_options=dissimilarity_metrics
    )
    
    return params

def set_multiprocessing_method():
    """Configure multiprocessing method based on OS"""
    try:
        if sys.platform == "win32":
            mp.set_start_method('spawn', force=True)
        else:
            mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

def get_safe_cpu_count():
    """Get safe number of CPUs to use"""
    cpu_count = os.cpu_count() or 1
    if sys.platform == "win32":
        return max(1, min(cpu_count - 1, 4))  # Max 4 on Windows
    else:
        return max(1, cpu_count - 1)

def load_config():
    """Load and process configuration from TOML file"""
    with open("settings.toml", "rb") as f:
        config = tomllib.load(f)
    
    # Extract config values
    TEST_SAMPLES = config["experiment"]["test_samples"]
    TRIALS = config["experiment"]["trials"] 
    CVFOLDS = config["experiment"]["cvfolds"]
    TARGET_METRIC = config["experiment"]["target_metric"]
    STUDY_NAME = f"{config["experiment"]["study_name"]}"
    RANDOM_SEED = config["experiment"]["random_seed"]
    
    SMILES_COLUMN = config["dataset"]["smiles_column"]
    LABEL_COLUMN = config["dataset"]["label_column"]
    
    RESULTS_PATH = config["paths"]["results_path"]
    ANALYSIS_RESULTS_PATH = RESULTS_PATH
    EVALUATION_RESULTS_PATH = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), "Evaluation")
    
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    return {
        'TEST_SAMPLES': TEST_SAMPLES,
        'TRIALS': TRIALS,
        'CVFOLDS': CVFOLDS,
        'TARGET_METRIC': TARGET_METRIC,
        'STUDY_NAME': STUDY_NAME,
        'RANDOM_SEED': RANDOM_SEED,
        'SMILES_COLUMN': SMILES_COLUMN,
        'LABEL_COLUMN': LABEL_COLUMN,
        'RESULTS_PATH': RESULTS_PATH,
        'ANALYSIS_RESULTS_PATH': ANALYSIS_RESULTS_PATH,
        'EVALUATION_RESULTS_PATH': EVALUATION_RESULTS_PATH
    }

def preprocess_smiles(smiles: str) -> str | None:
    """Canonicalize and clean a SMILES string. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Mantener el fragmento más grande
        # fragments = Chem.GetMolFrags(mol, asMols=True)
        # mol = max(fragments, key=lambda m: m.GetNumAtoms())

        representations = [
            Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, allHsExplicit=True),
            Chem.MolToSmiles(mol, rootedAtAtom=0),
        ]

        # Quitar duplicados y unir
        unique_reps = list(set(representations))
        combined = " |JOIN| ".join(unique_reps)

        return combined

    except Exception:
        return None

def load_and_process_datasets():
    """Load Clintox dataset and process into DataFrames"""
    # Load Clintox dataset (raw SMILES)
    tasks, datasets, _ = dc.molnet.load_clintox(
        splitter="stratified", reload=True, featurizer=dc.feat.DummyFeaturizer()
    )
    train_dataset, valid_dataset, test_dataset = datasets

    # Merge valid+test into search set
    search_dataset = dc.data.DiskDataset.merge([valid_dataset, test_dataset])

    # Convert to DataFrames with preprocessing
    def build_dataframe(dataset):
        smiles_clean = [preprocess_smiles(s) for s in dataset.X]
        mask = [s is not None for s in smiles_clean]  # drop invalids
        return pd.DataFrame({
            "smiles": np.array(smiles_clean)[mask],
            "fda_approved": dataset.y[mask, 0].astype(int),
            "ct_tox": dataset.y[mask, 1].astype(int)
        })

    df_test = build_dataframe(test_dataset)
    df_search = build_dataframe(search_dataset)

    return df_test, df_search

def main():
    set_multiprocessing_method()

    # Setup parameters and configuration
    params = setup_parameters()
    config = load_config()
    TEST_SAMPLES = config['TEST_SAMPLES']
    TRIALS = config['TRIALS']
    CVFOLDS = config['CVFOLDS']
    TARGET_METRIC = config['TARGET_METRIC']
    STUDY_NAME = config['STUDY_NAME']
    RANDOM_SEED = config['RANDOM_SEED']
    SMILES_COLUMN = config['SMILES_COLUMN']
    LABEL_COLUMN = config['LABEL_COLUMN']
    RESULTS_PATH = config['RESULTS_PATH']
    ANALYSIS_RESULTS_PATH = config['ANALYSIS_RESULTS_PATH']
    EVALUATION_RESULTS_PATH = config['EVALUATION_RESULTS_PATH']

    # Load and process datasets
    df_test, df_search = load_and_process_datasets()

    x_test, y_test = df_test[SMILES_COLUMN].values, df_test[LABEL_COLUMN].values
    x_search, y_search = df_search[SMILES_COLUMN].values, df_search[LABEL_COLUMN].values

    search_generator = SampleGenerator(
        data=x_search,
        labels=y_search,
        seed=RANDOM_SEED,
    )

    optimizer = ScOPEOptimizerAuto(
        n_jobs=get_safe_cpu_count(),
        n_trials=TRIALS,
        random_seed=RANDOM_SEED,
        target_metric=TARGET_METRIC,
        study_name=STUDY_NAME,
        output_path=ANALYSIS_RESULTS_PATH,
        cv_folds=CVFOLDS,
        parameter_space=params
    )

    search_all_x = []
    search_all_y = []
    search_all_kw = []

    for x_search_i, y_search_i, search_kw_samples_i in search_generator.generate(num_samples=TEST_SAMPLES):
        search_all_x.append(x_search_i)
        search_all_y.append(y_search_i)
        search_all_kw.append(search_kw_samples_i)

    study = optimizer.optimize(
        search_all_x,
        search_all_y,
        search_all_kw
    )

    optimizer.save_complete_analysis()

    best_model = optimizer.get_best_model()

    test_generator = SampleGenerator(
        data=x_test,
        labels=y_test,
        seed=RANDOM_SEED,
    )

    test_all_y_true = []
    test_all_y_predicted = []
    test_all_y_probas = []

    for x_test_i, y_test_i, test_kw_samples_i in test_generator.generate(num_samples=TEST_SAMPLES):

        predx = best_model(
            samples=x_test_i,
            kw_samples=test_kw_samples_i
        )[0]

        prediction: dict = predx['probas']

        sorted_dict = OrderedDict(sorted(prediction.items()))

        pred_key = max(sorted_dict, key=sorted_dict.get)

        predicted_class = int(pred_key.replace("sample_", ""))

        test_all_y_predicted.append(
            predicted_class
        )

        test_all_y_probas.append(
            list(sorted_dict.values())
        )

        test_all_y_true.append(
            y_test_i
        )

    results = make_report(
        y_true=test_all_y_true,
        y_pred=test_all_y_predicted,
        y_pred_proba=test_all_y_probas,
        save_path=EVALUATION_RESULTS_PATH
    )

    print(results)

if __name__ == "__main__":
    main()