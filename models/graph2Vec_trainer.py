from karateclub.graph_embedding import Graph2Vec
from karateclub import FeatherGraph, WaveletCharacteristic
import joblib
import os
from data.Dataset import graph_Pipeline_Dataset, load_data
import logging

logger = logging.getLogger(__name__)


class graph2Vec_Trainer:

    def __init__(self, data_config, outpath, epochs, lr, vec_dim, model_type='feather'):
        self.data_config = load_data(data_config)
        self.data_input = []

        if isinstance(self.data_config, dict):
            for dataset_name, dataset in self.data_config.items():
                self.data_input.append(dataset['path'])
        else:
            self.data_input.append(data_config)
        
        self.outpth = outpath
        self.epochs = epochs
        self.lr = lr
        self.vec_dim = vec_dim
        self.model_type = model_type.lower()

        logging.basicConfig(
            filename=os.path.join(self.outpth, 'log_file.log'),
            encoding='utf-8',
            level=logging.DEBUG
        )
        logger.setLevel(logging.INFO)

    def save_model(self, model):
        if not os.path.exists(self.outpth):
            os.makedirs(self.outpth)
        joblib.dump(model, os.path.join(self.outpth, f'graph_vec_model_{self.vec_dim}_dim.pkl'))

    def train(self):
        try:
            dataset = graph_Pipeline_Dataset(self.data_input, return_label=False)
            dataloader = dataset.get_Dataloader()
            graphs = dataloader.get_all_data()

            print(f'Start training model {self.model_type} with dim={self.vec_dim}')
            logger.info(f'Start training model {self.model_type} with dim={self.vec_dim}')

            # -----------------------------------------------------
            # Select model based on type
            # -----------------------------------------------------

            if self.model_type == 'graph2vec':
                model = Graph2Vec(
                    dimensions=self.vec_dim,
                    wl_iterations=2,
                    workers=16,
                    epochs=self.epochs,
                    learning_rate=self.lr
                )
                model.fit(graphs)

            elif self.model_type == 'feather':
                model = FeatherGraph(
                    dimensions=self.vec_dim
                )
                model.fit(graphs)

            elif self.model_type == 'waveletchar':
                model = WaveletCharacteristic()   
                model.fit(graphs)
                self.save_model(model)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.save_model(model)

        except Exception as e:
            print(f"An error occurred during training: {e}")
            logger.error(f"An error occurred during training: {e}")
            return
