# https://www.google.com/search?q=pca+method&oq=PCA+Met&gs_lcrp=EgZjaHJvbWUqBwgBEAAYgAQyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDINCAMQLhivARjHARiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIHCAgQABiABDIHCAkQABiABNIBCDI4ODdqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8#vhid=l5eUzO4w9GMYCM&vssid=l
import argparse
import time
import numpy as np
from models.trainer import Trainer, Trainer_Classification_Model
from models.vectorEmbedding import VectorEmbedding_Transformer
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-config", type=str, help="Input datasets path")
    parser.add_argument("-out", "--out-path", type=str, default="results/test", help='the output filepath to store models and result')
    parser.add_argument("-e", '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float, help="The learning rate value")
    parser.add_argument('-lr2', '--learning-rate-2', default=1e-2, type=float, help="The learning rate value for the MLP model")
    parser.add_argument('-lf', '--loss-function', default='TripletMarginLoss', type=str, help="The Loss function avail. loss function: [MSE, CrossEntropy, TripletMarginLoss]")
    parser.add_argument('-m', '--miner', action="store_true", help="Use miner within the self-supervised learning")
    parser.add_argument('-bz', '--batch-size', type=int, default=64, help='total batch size')
    parser.add_argument('-mc', '--model-config', type=str, help="Path for model configuration file")
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help="Checkpoint to continue the training")
    parser.add_argument('-opt', '--optimiser', type=str, default='SGD', help="Choose optimiser for training learning")
    parser.add_argument('-ul', '--unsupervised-learning', action='store_true', help="Choose the training to be with "
                                                                                    "unsupervised learning")
    parser.add_argument('-b', '--bias', type=bool, default=False, help='Select if we will have bias')
    parser.add_argument("-le", "--label-encoder", action="store_true")
    parser.add_argument("-lrs", "--lr-scheduler",  action='store_true')

    args = parser.parse_args()

    out_path = args.out_path
    out_path = os.path.join(os.getcwd(), out_path)
    input_data_config = args.data_config
    model_config = args.model_config
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    lr_2 = args.learning_rate_2
    loss_function = args.loss_function
    optimiser = args.optimiser
    bias = args.bias
    scheduler = args.lr_scheduler
    unsupervised_learning = args.unsupervised_learning
    miner = args.miner
    # TODO: Add check for checkpoint

    trainer = Trainer(data_config=input_data_config,
                      model_config_path=model_config,
                      out_path=out_path,
                      batch_size=batch_size,
                      num_epochs=epochs,
                      lr=lr,
                      optimiser=optimiser,
                      miner=miner,
                      loss_function=loss_function,
                      bias=bias,
                      scheduler=scheduler)

    # trainer.train()
    trainer_MLP = Trainer_Classification_Model(classification=False,
                                               out_path=out_path,
                                               d_out=1,
                                               embedding_model=trainer.model,
                                               batch_size=batch_size,
                                               lr=lr_2,
                                               bias=False,
                                               num_epochs=5,
                                               optimiser='Adam',
                                               d_hidden=1024,
                                               num_of_layers=3,
                                               d_in=trainer.model_config.attention_hidden_dim)
    trainer_MLP.train(datasets=trainer.datasets, name_of_dataset='Household_power_consumption')


if __name__ == '__main__':
    main()
