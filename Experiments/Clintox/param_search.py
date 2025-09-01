import os
import random
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from collections import OrderedDict
import tomllib

from scope.compression.compressors import CompressorType
from scope.utils.optimization.params import ParameterSpace, all_subsets
from scope.utils import make_report, SampleGenerator, ScOPEOptimizerAuto

compressors = all_subsets([c.value for c in CompressorType if c != CompressorType.SMILEZ])
dissimilarity_metrics = all_subsets([
    'ncd', 'ucd', 'cd', 'ncc'
])

params = ParameterSpace(
    compressor_names_options=compressors,
    compression_metric_names_options=dissimilarity_metrics
)


# -------------------
# Load Config from TOML
# -------------------
with open("settings.toml", "rb") as f:
    config = tomllib.load(f)

# Extract config values
TEST_SAMPLES = config["experiment"]["test_samples"]
TRIALS = config["experiment"]["trials"] 
CVFOLDS = config["experiment"]["cvfolds"]
TARGET_METRIC = config["experiment"]["target_metric"]
USE_CACHE = config["experiment"]["use_cache"]
STUDY_NAME = f"{config["experiment"]["study_name"]}_Cache_{USE_CACHE}"
RANDOM_SEED = config["experiment"]["random_seed"]

SMILES_COLUMN = config["dataset"]["smiles_column"]
LABEL_COLUMN = config["dataset"]["label_column"]

RESULTS_PATH = config["paths"]["results_path"]
ANALYSIS_RESULTS_PATH = RESULTS_PATH
EVALUATION_RESULTS_PATH = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), "Evaluation")

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# -------------------
# SMILES preprocessing function
# -------------------
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

# -------------------
# Load Clintox dataset (raw SMILES)
# -------------------
tasks, datasets, _ = dc.molnet.load_clintox(
    splitter="stratified", reload=True, featurizer=dc.feat.DummyFeaturizer()
)
train_dataset, valid_dataset, test_dataset = datasets

# Merge valid+test into search set
search_dataset = dc.data.DiskDataset.merge([valid_dataset, test_dataset])

# -------------------
# Convert to DataFrames with preprocessing
# -------------------
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

# -------------------
# Extract arrays
# -------------------
x_test, y_test = df_test[SMILES_COLUMN].values, df_test[LABEL_COLUMN].values
x_search, y_search = df_search[SMILES_COLUMN].values, df_search[LABEL_COLUMN].values

search_generator = SampleGenerator(
    data=x_search,
    labels=y_search,
    seed=RANDOM_SEED,
)

optimizer = ScOPEOptimizerAuto(
    n_jobs=1,
    n_trials=TRIALS,
    random_seed=RANDOM_SEED,
    target_metric=TARGET_METRIC,
    study_name=STUDY_NAME,
    output_path=ANALYSIS_RESULTS_PATH,
    cv_folds=CVFOLDS,
    use_cache=USE_CACHE,
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
