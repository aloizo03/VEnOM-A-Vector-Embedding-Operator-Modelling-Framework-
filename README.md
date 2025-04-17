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

For a NumTabData2Vec model first you have to denote the dataset in "data/config.data.yaml", and the model configutation on "models/model.config.yaml"
</br>
</br>
Model training:

For tabular data:

```
$ python train.py --data-config data/config.data.yaml --model-config models/model.config.yaml --epochs 100 --out-path "results/model_training" --optimiser Adam --learning-rate 0.0005 --lr-scheduler --batch-size 4
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
Adding the --graph as an argument implementing for graphs.

Step 2-Similarity Search:
</br>
On similarity search we give as an input the same model we did for the vectorisation and the target dataset where we want to model an operator, the $VEC_DIR where is the vector results from vectorisationa and the similarity search technique.
```
$ python select_data.py -ckpt "results/model_training/weights/best.pt" -i "/data/dataset_2/dataset.csv" -out "/results/experiments" -v $VEC_DIR -s "k-means"
```
Adding the --graph as an argument implementing for graphs.


Step 3-Opeartor Modelling:
Create labels:
</br>
```
$ python train_exp.py -i "/data/dataset/" -out "/results/experiments" --query "last" --operator 'linear_regression'
```

```
$ python3 operator_.py --dict-input sim_search.pickle -oi "/results/experiments/scores.csv" --out-path "/results/experiments" --operator 'linear_regression'
```

### Pre-trained models for vectorisation
On https://drive.google.com/drive/folders/1F6XF3p-sVwc8uyEtlDl806sODalb3rpj?usp=sharing, we are providing four models for vectorisation and must placed on results/ directory.

### Run the UI 
</br>

To execute the UI you must build the database server from the sqlite3 shell by typping:
```
.read server/server_utils/monitorDB_create_DB.sql
```
</br>

Run the server:
```
python server/main.py
```
