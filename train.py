# https://www.google.com/search?q=pca+method&oq=PCA+Met&gs_lcrp=EgZjaHJvbWUqBwgBEAAYgAQyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDINCAMQLhivARjHARiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIHCAgQABiABDIHCAkQABiABNIBCDI4ODdqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8#vhid=l5eUzO4w9GMYCM&vssid=l
import argparse
import time
import numpy as np
from models.trainer import Trainer
from utils_.utils import check_path
import os


# todo: check outpath if exists (Done i think)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-config", type=str, help="Input datasets path")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='the output filepath to store models and result')
    parser.add_argument("-e", '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float, help="The learning rate value")

    parser.add_argument('-lf', '--loss-function', default='loss_vae', type=str,
                        help="The Loss function avail. loss function: [MSE, VAE]")
    parser.add_argument('-bz', '--batch-size', type=int, default=64, help='total batch size')
    parser.add_argument('-mc', '--model-config', type=str, help="Path for model configuration file")
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help="Checkpoint to continue the training")
    parser.add_argument('-opt', '--optimiser', type=str, default='SGD', help="Choose optimiser for training learning")

    parser.add_argument('-b', '--bias', type=bool, default=False, help='Select if we will have bias')

    parser.add_argument("-le", "--label-encoder", action="store_true")
    parser.add_argument("-lrs", "--lr-scheduler", action='store_true')

    args = parser.parse_args()

    out_path = args.out_path
    out_path = os.path.join(os.getcwd(), out_path)
    input_data_config = args.data_config
    model_config = args.model_config
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    loss_function = args.loss_function
    optimiser = args.optimiser
    bias = args.bias
    scheduler = args.lr_scheduler

    ckpt_path = args.checkpoint
    if ckpt_path is not None:
        check_path(path=ckpt_path)
        trainer = Trainer(data_config=input_data_config,
                          model_config_path=model_config,
                          out_path=out_path,
                          batch_size=batch_size,
                          num_epochs=epochs,
                          lr=lr,
                          optimiser=optimiser,
                          loss_function=loss_function,
                          bias=bias,
                          scheduler=scheduler,
                          continue_ckpt=ckpt_path)
    else:
        trainer = Trainer(data_config=input_data_config,
                          model_config_path=model_config,
                          out_path=out_path,
                          batch_size=batch_size,
                          num_epochs=epochs,
                          lr=lr,
                          optimiser=optimiser,
                          loss_function=loss_function,
                          bias=bias,
                          scheduler=scheduler)

    trainer.train(continue_train_ckpt=ckpt_path)


if __name__ == '__main__':
    main()
