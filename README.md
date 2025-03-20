# Analytics Modelling over Multiple Datasets using Vector Embeddings

For a NumTabData2Vec model first you have to denote the dataset in "data/config.data.yaml", and the model configutation on "models/model.config.yaml"
</br>
</br>
Model training:
```
$ python train.py --data-config data/config.data.yaml --model-config models/model.config.yaml --epochs 100 --out-path "results/model_training" --optimiser Adam --learning-rate 0.0005 --lr-scheduler --batch-size 4
```
</br>
When the model is trained you can use the framework:

##Framework Steps

Step 1-Dataset Vectorization:
```
$ python datasets_to_vec.py -ckpt "results/model_training/weights/best.pt" -i "/data/dataset/" -out "/results/experiments" -bz 8
```

Step 2-Similarity Search:
</br>
On similarity search we give as an input the same model we did for the vectorisation and the target dataset where we want to model an operator, the $VEC_DIR where is the vector results from vectorisationa and the similarity search technique.
```
$ python select_data.py -ckpt "results/model_training/weights/best.pt" -i "/data/dataset_2/dataset.csv" -out "/results/experiments" -v $VEC_DIR -s "k-means"
```
Step 3-Opeartor Modelling:
Create labels:
</br>
```
$ python train_exp.py -i "/data/dataset/" -out "/results/experiments" --query "last" --operator 'linear_regression'
```

```
$ python3 operator_.py --dict-input sim_search.pickle -oi "/results/experiments/scores.csv" --out-path "/results/experiments" --operator 'linear_regression'
```

