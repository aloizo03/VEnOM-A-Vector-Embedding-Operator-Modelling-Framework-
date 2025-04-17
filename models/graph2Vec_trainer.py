
from karateclub.graph_embedding import Graph2Vec
import joblib
import os
from data.Dataset import graph_Pipeline_Dataset
import logging

logger = logging.getLogger(__name__)


class graph2Vec_Trainer:

    def __init__(self, data_input, outpath, epochs, lr, vec_dim):
        self.data_input = data_input
        self.outpth = outpath
        self.epochs = epochs
        self.lr = lr
        self.vec_dim = vec_dim

        logging.basicConfig(filename=os.path.join(self.outpth, 'log_file.log'), encoding='utf-8', level=logging.DEBUG)
        logger.setLevel(logging.INFO)

    def save_model(self, model):
        if not os.path.exists(self.outpth):
            os.makedirs(self.outpth)

        joblib.dump(model, os.path.join(self.outpth, f'graph2vec_model_{self.vec_dim}_dim.pkl'))

    def train(self):
        try:
            dataset = graph_Pipeline_Dataset(self.data_input, return_label=False)
            dataloader = dataset.get_Dataloader()
            graphs = dataloader.get_all_data()

            print(f'Start training graph2vec model for {self.epochs} epochs with {self.vec_dim} dimensions')
            logger.info(f'Start training graph2vec model for {self.epochs} epochs with {self.vec_dim} dimensions')
            model = Graph2Vec(dimensions=self.vec_dim, wl_iterations=2, workers=16, epochs=self.epochs, learning_rate=self.lr)
            model.fit(graphs)

            self.save_model(model)
        except Exception as e:
            print(f"An error occurred during training: {e}")
            logger.error(f"An error occurred during training: {e}")
            return


