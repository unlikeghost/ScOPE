import os
import random
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from collections import OrderedDict

from scope.utils import make_report, SampleGenerator, ScOPEOptimizerAuto

# -------------------
# Config
# -------------------
TEST_SAMPLES = 3
TRIALS = 100
CVFOLDS = 5
TARGET_METRIC = "log_loss"

STUDY_NAME = "Clintox"
SMILES_COLUMN = "smiles"
LABEL_COLUMN = "ct_tox"

RANDOM_SEED = 42

RESULTS_PATH = os.path.join("results")
ANALYSIS_RESULTS_PATH = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), "Optimization")
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
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),  # canónico básico
            Chem.MolToSmiles(mol, allHsExplicit=True),                    # con H explícitos
            Chem.MolToSmiles(mol, rootedAtAtom=0),                        # enraizado
        ]

        # Quitar duplicados y unir
        unique_reps = list(set(representations))
        combined = " |JOIN| ".join(unique_reps)
        
        # canonical = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)
        
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
    free_cpu=0,
    n_trials=TRIALS,
    random_seed=RANDOM_SEED,
    target_metric=TARGET_METRIC,
    study_name=f'{STUDY_NAME}_Samples_{TEST_SAMPLES}',
    output_path=ANALYSIS_RESULTS_PATH,
    cv_folds=CVFOLDS
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

optimizer.save_complete_analysis(top_n=TRIALS)

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