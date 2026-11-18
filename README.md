# Analytics Modelling over Multiple Datasets using Vector Embeddings

Dependencies:

1. `Python` version 3.9.21
2. `SQLite3` see from https://sqlite.org/download.html
3. `Qdrant` see from https://qdrant.tech/documentation/guides/installation/
4. `karateclub=1.3.3`
5. `networkx=2.6.3`
6. `numpy=1.22.0 `
7. `pandas=1.3.5`
8. `pytorch=2.4.0`
9. `flask=3.1.0`
10. `flask-wtf=1.2.2`
11. `scikit-learn=1.6.1`
12. `scipy=1.8.0`
13. `sqlite=3.45.3`
14. `statsmodels=0.13.5`
15. `tqdm=4.67.1`

For a NumTabData2Vec model first you have to denote the dataset in "data/config.data.yaml", and the model configuration on "models/model.config.yaml"
</br>
</br>
Model training:

For tabular data:

```
$ python train.py --data-config data/config.data.yaml --model-config models/model.config.yaml --epochs 100 --out-path "results/model_training" --optimiser Adam --learning-rate 0.001 --lr-scheduler --batch-size 4
```

For graph:
```
$ python train_graph2vec.py -i "/data/graphs_edgelist" --out "/results/graph_model" -e 10 -vec 256
```

</br>
When the model is trained you can use the framework:

## Framework Steps

Step 1-Dataset Vectorization:
```
$ python datasets_to_vec.py -ckpt "results/model_training/weights/best.pt" -i "/data/dataset/" -out "/results/experiments" -bz 8
```
Adding the -dt graph as an argument implementing for graphs, and -dt image implementing for images.
</br>
Adding the -s vectordb as an argument you select to store the vector collection into Vector Database


Step 2-Similarity Search:
</br>
On similarity search we give as an input the same model we did for the vectorisation and the target dataset where we want to model an operator, the $VEC_DIR where is the vector results from vectorisation and the similarity search technique.
```
$ python select_data.py -ckpt "results/model_training/weights/best.pt" -i "/data/dataset_2/dataset.csv" -out "/results/experiments" -v $VEC_DIR -s "k-means"
```
Adding the -dt graph as an argument implementing for graphs, and -dt image implementing for images.


Step 3-Operator Modelling:
Create labels:
</br>
```
$ python train_exp.py -i "/data/dataset/" -out "/results/experiments" --query "last" --operator 'linear_regression'
```

```
$ python3 operator_.py --dict-input sim_search.pickle -oi "/results/experiments/scores.csv" --out-path "/results/experiments" --operator 'linear_regression'
```

Step 4-Dataset Evolution:
</br>
```
$ python update_vectors.py ckpt "results/model_training/weights/best.pt" -i "/data/dataset/" -out "/results/experiments" -bz 8 -v collection_name -vs 300 -dt tabular -s vectordb -th 0.1
```
Adding the -dt graph as an argument implementing for graphs, and -dt image implementing for images.

### Pre-trained models for vectorisation
On https://drive.google.com/drive/folders/1F6XF3p-sVwc8uyEtlDl806sODalb3rpj?usp=sharing, we are providing four models for vectorisation and must placed on results/ directory.

### TabFM for tabular datasets

Tabular datasets can alternatively be vectorised with Google's [TabFM](https://huggingface.co/google/tabfm-1.0.0-pytorch) foundation model (`pip install 'tabfm[pytorch]'`, weights are downloaded automatically from the Hugging Face hub, so no `-ckpt` is needed). TabFM has no official embedding API, so the per-row representations feeding its in-context predictor (8 CLS tokens x 256 dims = 2048) are mean-pooled per dataset and reduced to `--vector-size` k dimensions with a deterministic random projection:
```
$ python datasets_to_vec.py --tab-model tabfm -i "/data/dataset/" -out "/results/experiments" -vs 100 -s local
$ python select_data.py --tab-model tabfm -i "/data/dataset_2/dataset.csv" -out "/results/experiments" -v $VEC_DIR -s "distance" -vs 100
```
The vectors can be stored locally (`-s local`) or in the Qdrant vector DB (`-s vectorDB`) exactly like the Num2Vec ones, and `select_data.py` must use the same `--tab-model` and `--vector-size` as the run that created the stored vectors. `--tabfm-task {classification,regression}` selects the backbone weights and `--tabfm-max-rows` caps how many rows per dataset are sampled as TabFM context (default 512).

### Dataset2Vec for tabular datasets

Tabular datasets can also be vectorised with the pretrained [Dataset2Vec](https://github.com/hadijomaa/dataset2vec) meta-feature model ("Dataset2Vec: Learning Dataset Meta-Features"). The repo is vendored under `dataset2vec/` together with its five pretrained checkpoint splits, so no `-ckpt` is needed (requires `tensorflow`). Each dataset is embedded by sampling batches of (instances x features x classes) subsets — treating the last CSV column as the target, quantile-binned when continuous — and averaging the resulting meta-features into one 32-d vector:
```
$ python datasets_to_vec.py --tab-model dataset2vec -i "/data/dataset/" -out "/results/experiments" -s local
$ python select_data.py --tab-model dataset2vec -i "/data/dataset_2/dataset.csv" -out "/results/experiments" -v $VEC_DIR -s "distance"
```
`--d2v-split {0..4}` picks the pretrained checkpoint split (default 0) and `--d2v-batches` sets how many sampled batches are averaged per dataset (default 10). Vectors are always 32-d (the checkpoint's meta-feature width), regardless of `--vector-size`, unless a smaller size is requested.

### Reproducibility

All results in the paper use random seed `0`, which seeds `random`, `numpy`, `torch` (CPU and CUDA) and the `random_state` of the surrogate models (`utils_/utils.py::set_global_seed`). Each experiment is repeated 10 times, and the reported values are the mean and standard deviation over these runs.

The seed is set once, at the start of each experiment, and the 10 runs then follow consecutively in the same process. Each run therefore draws different random numbers (dataset sampling and surrogate initialisation), and re-running the command reproduces all 10 runs:

```
$ python operator_.py -i sim_search.pickle -oi scores.csv -out results/experiments -o linear_regression -r 10 --seed 0
$ python train_op_sr.py ... -r 10 --seed 0      # random-sampling (SR) baseline
```

The configuration of every component (Qdrant, Apollo, framework parameters, NumTabData2Vec training, hardware and seed) is listed in `results/configurations.csv`.
 
### Baseline: Apollo

We use Apollo (Giannakopoulos et al.) as the baseline, run from its open-source implementation ([giagiannis/data-profiler](https://github.com/giagiannis/data-profiler), commit [`a643276`](https://github.com/giagiannis/data-profiler/commit/a64327652e682380fa05452314856382a08edc5e)) without modification and with its default configuration, except for the sampling ratio, which we set to 0.1 and 0.2 (Apollo-SR 0.1 and Apollo-SR 0.2 in the paper).
Instructions for running Apollo are provided by its authors as shell scripts in the [`_scripts/`](https://github.com/giagiannis/data-profiler/tree/master/_scripts) folder of their repository.

### Hardware, configurations and timings

Our framework runs on 4× NVIDIA A10 GPUs (24 GB each); Apollo runs on an AMD EPYC 7R32 CPU.

| File | Content |
|---|---|
| `results/configurations.csv` | Qdrant, Apollo, framework and NumTabData2Vec settings, and hardware |
| `results/hardware_timing/cost_per_operator.csv` | One-time cost (`T_once`), single-query end-to-end latency (`T_e2e`), operator cost (`T_op`), break-even number of queries (`N*`) and speedups (Appendix B, Table 12) |
| `results/nns_exact_vs_qdrant/nns_exact_vs_qdrant.csv` | Recall@k and search latency of Qdrant against exact nearest-neighbour search (Appendix E, Table 16) |

All times are in seconds, except search latency (ms per query).

### Run the UI 
</br>

The UI uses the SQLite database `server/server_utils/vector_OM.db`. Missing tables are created when the server starts; to rebuild the database from scratch, run in the sqlite3 shell:
```
.read server/server_utils/monitorDB_create_DB
```
</br>

Run the server:
```
python server/main.py
```

The UI supports the same modalities and embedding models as the command line:

| Modality | Embedding models | Operators |
|---|---|---|
| Tabular | Num2Vec (checkpoints listed in `Models_Table`), TabFM, Dataset2Vec | Linear Regression, MLP Regression, SVM (SGD), SVM, Logistic Regression, MLP |
| Graph | Graph2Vec (checkpoints listed in `Models_Table` with `data_type = 2`) | Betweenness, Edge Betweenness, Closeness and Eigenvector Centrality, PageRank, Spectral Radius |
| Image | ViT `google/vit-base-patch16-224`, projected to a size from `Vector_Dimension` | SVM (SGD), SVM, Logistic Regression, MLP, KNN |

Each set of vectors records the model that created it, and the similarity search embeds the query datasets with that same model. Relative dataset and model paths are resolved against the repository root. For graph and image operators the target labels of the query datasets are created automatically, unless a labels CSV is given.

All paths default to locations inside the repository and can be overridden with environment variables: `VENOM_DB_FILE` (default `server/server_utils/vector_OM.db`), `VENOM_LOCAL_STORE` (`server/tmp_locally_store`), `VENOM_SIM_SEARCH_STORE` (`server/sim_search_store`), `VENOM_SCORES_DIR` (`server/tmp_score_labels/`), `VENOM_OPERATORS_DIR` (`server/tmp_operators/`), `VENOM_SECRET_KEY` (Flask session key), `VENOM_GRAPH_DATA_ROOT` (`data`, used by `data/pre_process_graph_data.py`) and `VENOM_RESULTS_ROOT` (`results/`, resolved against the working directory and used by `utils_/create_operator.py` and `nns_recall_benchmark.py`).

</br>

## Citation

If you use this work in your research, please cite it as follows:

```bibtex
@inproceedings{loizou2025analytics,
  title={Analytics modelling over multiple datasets using vector embeddings},
  author={Loizou, Andreas and Tsoumakos, Dimitrios},
  booktitle={International Conference on Database and Expert Systems Applications},
  pages={237--253},
  year={2025},
  organization={Springer}
}

@inproceedings{loizou2025venom,
  title={VEnOM: A Vector Embedding Operator Modelling Framework},
  author={Loizou, Andreas and Andriotis, Nikolaos and Tsoumakos, Dimitrios},
  booktitle={Proceedings of the 19th ACM International Conference on Distributed and Event-based Systems},
  pages={245--248},
  year={2025}
}
