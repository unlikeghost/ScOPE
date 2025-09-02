from typing import List
from collections import OrderedDict
from scope.utils import make_report
from scope.utils import ScOPEOptimizerRandom, ScOPEOptimizerAuto
from scope.compression.compressors import CompressorType
from scope.utils.optimization.params import ParameterSpace, all_subsets
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def create_restricted_parameter_space(promising_space, original_params):
    """Create restricted parameter space from promising results"""
    restricted_compressors = []
    for combo_str in promising_space.get('compressor_names', []):
        combo_tuple = tuple(combo_str.split(','))
        restricted_compressors.append(combo_tuple)
    
    restricted_metrics = []
    for combo_str in promising_space.get('compression_metric_names', []):
        combo_tuple = tuple(combo_str.split(','))
        restricted_metrics.append(combo_tuple)
    
    return ParameterSpace(
        compressor_names_options=restricted_compressors or original_params.compressor_names_options,
        compression_metric_names_options=restricted_metrics or original_params.compression_metric_names_options,
        evaluation_metrics=promising_space.get('evaluation_metric', original_params.evaluation_metrics),
        concat_value_options=promising_space.get('join_string', original_params.concat_value_options),
        get_sigma_options=promising_space.get('get_sigma', original_params.get_sigma_options),
        aggregation_method_options=promising_space.get('aggregation_method', original_params.aggregation_method_options)
    )


compressors = all_subsets([c.value for c in CompressorType if c != CompressorType.SMILEZ])
dissimilarity_metrics = all_subsets([
    'ncd', 'ucd', 'cd', 'ncc'
])

params = ParameterSpace(
    compressor_names_options=compressors,
    concat_value_options=[''],
    compression_metric_names_options=dissimilarity_metrics
)

# Dataset de validación
x_validation = [
    "molecule toxic heavy metal lead", "compound dangerous poison arsenic",
    "chemical harmful mercury substance", "element toxic cadmium dangerous",
    "poison lethal cyanide compound", "toxic substance benzene harmful",
    "dangerous chemical formaldehyde", "harmful compound asbestos fiber",
    "toxic metal chromium dangerous", "poison substance strychnine lethal",
    "harmful chemical dioxin toxic", "dangerous compound pesticide toxic",
    "safe molecule water oxygen", "harmless compound sugar glucose",
    "beneficial substance vitamin C", "safe chemical sodium chloride",
    "harmless element calcium safe", "beneficial compound protein amino",
    "safe substance carbohydrate energy", "harmless chemical citric acid",
    "beneficial molecule antioxidant", "safe compound fiber cellulose",
    "harmless substance mineral zinc", "beneficial chemical enzyme natural"
]

y_validation = [0]*12 + [1]*12

kw_samples_validation = [
    {
        0: ["toxic harmful dangerous poison lethal", "mercury lead arsenic cyanide"],
        1: ["safe harmless beneficial healthy natural", "water vitamin protein calcium"]
    }
    for _ in range(24)
]

# Dataset para búsqueda de parámetros (entrenamiento)
x_train = [
    "chemical toxic pesticide harmful", "dangerous metal arsenic lead",
    "poisonous substance cyanide lethal", "harmful compound mercury cadmium",
    "toxic gas formaldehyde dangerous", "dangerous substance dioxin poison",
    "safe molecule water glucose", "beneficial compound vitamin protein",
    "harmless chemical citric acid", "safe element calcium magnesium",
    "safe substance carbohydrate fiber", "beneficial molecule antioxidant enzyme"
]

y_train = [0]*6 + [1]*6

kw_samples_train = [
    {
        0: ["toxic harmful dangerous poison lethal", "mercury lead arsenic cyanide"],
        1: ["safe harmless beneficial healthy natural", "water vitamin protein calcium"]
    }
    for _ in range(12)
]

# PHASE 1: Random exploration
print("PHASE 1: Random exploration")
random_optimizer = ScOPEOptimizerRandom(
    random_seed=42,
    n_trials=5000,
    target_metric='log_loss',
    study_name="phase1_random_search",
    n_jobs=1,
    parameter_space=params
)

random_study = random_optimizer.optimize(x_train, y_train, kw_samples_train)
promising_space = random_optimizer.extract_promising_parameter_space(top_percent=20.0)

# PHASE 2: AutoSampler with restricted space
print("\nPHASE 2: AutoSampler on promising regions")
restricted_params = create_restricted_parameter_space(promising_space, params)

auto_optimizer = ScOPEOptimizerAuto(
    random_seed=42,
    n_trials=1000,
    target_metric='log_loss',
    study_name="phase2_auto_search",
    n_jobs=1,
    parameter_space=restricted_params
)

auto_study = auto_optimizer.optimize(x_train, y_train, kw_samples_train)

# Use best model from AutoSampler
best_model = auto_optimizer.get_best_model()
auto_optimizer.save_complete_analysis()

# Validación
all_y_true = y_validation
all_y_predicted = []
all_y_probas = []

preds: List[dict] = best_model(
    samples=x_validation,
    kw_samples=kw_samples_validation
)

for predx in preds:
    prediction: dict = predx['probas']
    sorted_dict = OrderedDict(sorted(prediction.items()))
    pred_key = max(sorted_dict, key=sorted_dict.get)
    predicted_class = int(pred_key.replace("sample_", ""))
    all_y_predicted.append(predicted_class)
    all_y_probas.append(list(sorted_dict.values()))

results = make_report(
    y_true=all_y_true,
    y_pred=all_y_predicted,
    y_pred_proba=all_y_probas,
    save_path='results'
)

# Comparison
print(f"\nRandom best: {random_study.best_value:.4f}")
print(f"AutoSampler best: {auto_study.best_value:.4f}")
print(f"Improvement: {random_study.best_value - auto_study.best_value:.4f}")
print(results)