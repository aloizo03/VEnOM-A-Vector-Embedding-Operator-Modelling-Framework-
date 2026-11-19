import os

ROOT = os.environ.get("VE_ROOT", "/various/aloizou")
REPO = f"{ROOT}/Projects/Projects_updated/Vector_Embedding"
DATA = f"{ROOT}/Results"

PATH_REMAP = {f"{ROOT}/Results/Results/": f"{DATA}/"}

QDRANT_URL = os.environ.get("VE_QDRANT_URL", "http://localhost:6333")

SEEDS = list(range(10))
LAMBDA = 0.2
LAMBDA_SWEEP = [0.05, 0.1, 0.2, 0.3]
SR_RATIOS = [0.05, 0.1, 0.2, 0.4]
SR_MODEL = "perceptron_regression"
METHODS = {
    "cosine": "cosine",
    "distance": "distance",
    "kmeans": "k-means",
    "vectordb": "vectordb",
}

NUM2VEC_CKPT = {d: f"{DATA}/NumTabData2Vec_Weights/Opt_adam_all_data_e100_v{d}/best.pt" for d in (100, 200, 300)}
GRAPH_CKPT = f"{ROOT}/Projects/Projects_updated/Wavelet-Weights/1000-DIM/graph_vec_model_1000_dim.pkl"

# Order = run order (HPC + Adult first, as requested).
DATASETS = {
    "hpc": {
        "type": "tabular", "operators": ["linear_regression", "mlp_regression"],
        "lake": f"{DATA}/data/HPC_2Krows/test", "queries": f"{DATA}/data/HPC_2Krows/test_2",
        "main_dim": 300,
        "collections": {300: "vectors_HPC_VS300_1773660566.9167697_300",
                        200: "vectors_HPC_200V_1782375429.4645462_200",
                        100: "vectors_HPC_100V_1782374287.9146304_100"},
        "embedders": {"dataset2vec": "vectors_HPC_dataset2vec_300VS_1784582818.7308002_32",
                      "tabfm": None},
        "sr": True,
    },
    "adult": {
        "type": "tabular", "operators": ["mlp_regression", "svm_sgd"],
        "lake": f"{DATA}/data/Adult/test", "queries": f"{DATA}/data/Adult/test_2",
        "main_dim": 300,
        "collections": {300: "vectors_adult_VS300_1773667107.9157124_300",
                        200: "vectors_adult_200V_1782382035.3466623_200",
                        100: "vectors_adult_100V_1782377798.1113122_100"},
        "embedders": {"dataset2vec": "vectors_adult_dataset2vec_300VS_1784583112.4585478_32",
                      "tabfm": None},
        "sr": True,
    },
    "stock": {
        "type": "tabular", "operators": ["linear_regression", "mlp_regression"],
        "lake": f"{DATA}/data/Stock/test", "queries": f"{DATA}/data/Stock/test_2",
        "main_dim": 300,
        "collections": {300: "vectors_STOCKS_VS300_1773662729.3367891_300",
                        200: "vectors_Stock_200V_1782377404.0813515_200",
                        100: "vectors_Stock_100V_1782377050.3591993_100"},
        "sr": True,
    },
    "weather": {
        "type": "tabular", "operators": ["mlp_regression", "svm_sgd"],
        "lake": f"{DATA}/data/Weather/test", "queries": f"{DATA}/data/Weather/test_2",
        "main_dim": 300,
        "collections": {300: "vectors_Weather_VS300_1773667512.4669042_300",
                        200: "vectors_weather_200V_1782382843.053541_200",
                        100: "vectors_weather_100V_1782382733.8910077_100"},
        "sr": True,
    },
    "collab": {
        "type": "graph", "operators": ["BC", "CC", "EBC"],
        "lake": f"{DATA}/Data/COLLAB/Collab_graphs/graphs_edgelist",
        "queries": f"{DATA}/Data/COLLAB/Collab_graphs/test_",
        "main_dim": 1000,
        "collections": {1000: "vectors_collab_VS1000_1773740151.6120446_1000"},
    },
    "enzymes": {
        "type": "graph", "operators": ["PR", "SR"],
        "lake": f"{DATA}/Data/ENZYMES/ENZYMES_Graphs", "queries": f"{DATA}/Data/ENZYMES/test_graphs",
        "main_dim": 1000,
        "collections": {1000: "vectors_enzyms_VS1000_1773740476.0486188_1000"},
    },
    "mnist": {
        "type": "image", "operators": ["cnn", "k_nearest_neighbors"],
        "lake": f"{DATA}/Image_Data/MNIST/mnist_train.csv",
        "queries": f"{DATA}/Image_Data/MNIST/mnist_test_venom.csv",
        "main_dim": 300,
        "collections": {300: "vectors_MNIST_VS300_1773671694.1564825_300",
                        200: "vectors_MNIST_VS200_1774619675.9581687_200",
                        100: "vectors_MNIST_VS100_1774618152.1422553_100"},
    },
    "fmnist": {
        "type": "image", "operators": ["cnn", "k_nearest_neighbors"],
        "lake": f"{DATA}/Image_Data/f-MNIST/f-mnist_train.csv",
        "queries": f"{DATA}/Image_Data/f-MNIST/f-mnist_test_venom.csv",
        "main_dim": 300,
        "collections": {300: "vectors_f-MNIST_VS300_1773735851.964988_300",
                        200: "vectors_f-MNIST_VS200_1774724593.884282_200",
                        100: "vectors_f-MNIST_VS100_1774723060.5379517_100"},
    },
}

VERIFY = [("hpc", "linear_regression"), ("adult", "mlp_regression")]

OLD_LABELS_DIR = f"{REPO}/results/full_pipeline_eval/labels"
