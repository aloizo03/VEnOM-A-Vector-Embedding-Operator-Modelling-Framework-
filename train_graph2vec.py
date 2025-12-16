import argparse
import time
import numpy as np
from models.graph2Vec_trainer import graph2Vec_Trainer
from utils_.utils import check_path
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-input", type=str, help="Input datasets path or yml file")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='the output filepath to store models and result')
    parser.add_argument("-e", '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float, help="The learning rate value")
    parser.add_argument('-vec', '--vec-dim', default=128, type=int, help="The dimension of the vector")
    parser.add_argument('-a', '--algo', default='waveletchar', help="The mbedding model to get train (Avai. Feather, Graph2Vec)")

    args = parser.parse_args()

    out_path = args.out_path
    out_path = os.path.join(os.getcwd(), out_path)

    data_input = args.data_input
    data_input = os.path.join(os.getcwd(), data_input)
    epochs = args.epochs
    learning_rate = args.learning_rate
    vec_dim = args.vec_dim
    graph_algo = args.algo

    check_path(out_path)
    trainer = graph2Vec_Trainer(data_config=data_input, 
                                outpath=out_path, 
                                epochs=epochs, 
                                lr=learning_rate,
                                vec_dim=vec_dim, 
                                model_type=graph_algo)

    trainer.train()

if __name__ == '__main__':
    main()

