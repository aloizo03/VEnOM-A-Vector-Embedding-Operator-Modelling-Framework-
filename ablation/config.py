import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rerun_config", os.environ.get("VE_RERUN_CONFIG", os.path.join(_HERE, "..", "rerun", "config.py")))
RC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RC)

SEEDS = RC.SEEDS
LAMBDA = RC.LAMBDA
PATH_REMAP = getattr(RC, "PATH_REMAP", {})

OPERATORS = {
    "hpc":     ["linear_regression", "mlp_regression"],
    "stock":   ["linear_regression", "mlp_regression"],
    "adult":   ["mlp_regression", "svm_sgd"],
    "weather": ["mlp_regression", "svm_sgd"],
}
DATASETS = {k: dict(RC.DATASETS[k], operators=ops) for k, ops in OPERATORS.items()}

DATASETS["stock"]["lake"] = f"{RC.DATA}/data/Stock/test_"
DATASETS["weather"]["lake"] = f"{RC.DATA}/data/Weather/pre_process_DS_1000"


DIM = int(os.environ.get("VE_ABL_DIM", 100))
assert DIM in RC.NUM2VEC_CKPT, f"VE_ABL_DIM must be one of {list(RC.NUM2VEC_CKPT)}"
PAPER_CKPT = RC.NUM2VEC_CKPT[DIM]

TRAIN_DATA_CONFIG = "data/config.data.yaml"

TRAIN = dict(epochs=100, batch_size=4, micro_batch=1, lr=0.001, optimiser="Adam", val_frac=0.1)
TRAIN_SEEDS = [0]
MAX_HOURS_PER_MODEL = None

EMBED_MODES = ["paper"]

VARIANTS = {
    "ae":                dict(cls="Num2VecAE", objective="ae", group="objective,backbone"),
    "siamese":           dict(cls="Num2VecSiamese", objective="siamese", group="objective"),
    "mlp_vae":           dict(cls="MLPEncoderVAE", objective="vae", beta=1.0, group="objective,backbone"),
    "mlp_ae":            dict(cls="MLPEncoderAE", objective="ae", group="backbone"),
    **{f"beta_{b:g}": dict(cls="Num2Vec", objective="vae", beta=b, group="beta")
       for b in (0.0, 0.1, 2.0)},
}

SMOKE = None
if os.environ.get("VE_ABL_SMOKE"):
    SMOKE = dict(train_files=8, lake=12, queries=3,
                 model_info=dict(num_layers=1, d_token=16, d_out=16, head_num=2, ffn_dim=32))
    TRAIN = dict(TRAIN, epochs=1)
    SEEDS = [0, 1]
