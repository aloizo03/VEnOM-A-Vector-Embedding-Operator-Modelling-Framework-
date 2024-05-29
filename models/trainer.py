import matplotlib.pyplot as plt
import numpy as np
import logging
from utils_.utils import calculate_accuracy

from pytorch_metric_learning import miners
from models.Loss import Loss_Compute, RMSELoss
from models.vectorEmbedding import VectorEmbedding_Transformer
from models.utils import load_model_config, get_optimiser, get_loss_function, get_parameter_names
from models.MLmodels import MLP
from data.Dataset import load_data, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os

logger = logging.getLogger(__name__)


class Trainer_Classification_Model:
    def __init__(self,
                 embedding_model,
                 out_path,
                 num_epochs,
                 batch_size,
                 num_of_layers,
                 d_in,
                 d_out,
                 d_hidden,
                 lr,
                 optimiser='SGD',
                 classification=False,
                 bias=None,
                 model='MLP'):
        self.embedding_model = embedding_model
        self.out_path = out_path

        self.MLP_model = MLP(input_dim=d_in,
                             hidden_dim=d_hidden,
                             out_dim=d_out,
                             num_of_layers=num_of_layers,
                             bias=bias,
                             init_=True,
                             classification=classification)

        self.num_epoch = num_epochs
        self.batch_size = batch_size

        self.optimiser = get_optimiser(parameters=self.MLP_model.parameters(),
                                       optimiser_name=optimiser,
                                       lr=lr,
                                       weight_decay=0.1,
                                       momentum=0.8)

        self.loss_fun = RMSELoss()

    def test(self, dataloader):
        self.embedding_model.eval()
        self.MLP_model.eval()
        acc = 0
        loss_ = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                num_continuous_tuples, num_int_tuples, bin_tuples, cat_tuples, labels_y = batch_data
                labels_y = labels_y[0]
                labels_y = labels_y.type(torch.float64)
                labels_y = labels_y.to(self.model.device)

                embeddings_ = self.embedding_model(num_int_tuples, num_continuous_tuples, bin_tuples,
                                                   cat_tuples)

                y_predict = self.MLP_model(embeddings_)

                acc += calculate_accuracy(y_predict, labels_y)
                loss_ += self.loss_fun(y_predict, labels_y)
            acc /= (batch_idx + 1)
            loss_ /= (batch_idx + 1)
        return acc, loss_

    def train(self, datasets, name_of_dataset):
        loss_list = []
        acc_list = []
        loss_list_val = []
        acc_list_val = []
        with tqdm(total=self.num_epoch) as pbar:
            for epoch in range(self.num_epoch):
                train_databuilder, test_databuilder, val_databuilder = datasets.get_Dataset_dataloader(dataset_name=name_of_dataset)
                train_dataloader = DataLoader(train_databuilder, batch_size=self.batch_size, shuffle=False)
                self.embedding_model.train(mode=False)
                self.MLP_model.train(mode=True)

                total_loss = 0
                total_acc = 0
                for batch_idx, batch_data in enumerate(train_dataloader):
                    self.optimiser.zero_grad()
                    num_continuous_tuples, num_int_tuples, bin_tuples, cat_tuples, labels_y = batch_data
                    labels_y = labels_y
                    labels_y = labels_y.type(torch.float64)
                    labels_y = labels_y.to(self.embedding_model.device)

                    embeddings_ = self.embedding_model(num_int_tuples, num_continuous_tuples, bin_tuples,
                                                               cat_tuples)

                    y_predict = self.MLP_model(embeddings_)

                    print(y_predict.shape)
                    print(labels_y.shape)

                    acc = calculate_accuracy(y_predict, labels_y)

                    loss = self.loss_fun(y_predict, labels_y)
                    total_loss += loss.backward()
                    total_acc += acc
                    self.optimiser.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                pbar.set_description(f'Epoch :{epoch},'
                                     f' loss: {total_loss}, acc: {total_acc}')
                total_loss /= (batch_idx + 1)
                total_acc /= (batch_idx + 1)
                acc_list.append(total_acc)
                loss_list.append(total_loss)
                acc_val, loss_val = self.test(val_databuilder)
                acc_list_val.append(acc_val)
                loss_list_val.append(loss_val)


class Trainer:

    def __init__(self, data_config,
                 model_config_path,
                 out_path,
                 num_epochs,
                 batch_size,
                 lr,
                 bias,
                 optimiser,
                 loss_function,
                 miner,
                 scheduler=False):

        logging.basicConfig(filename=os.path.join(out_path, 'log_file.log'), encoding='utf-8', level=logging.DEBUG)
        logger.setLevel(logging.INFO)

        self.lr_scheduler = scheduler
        self.outpath = out_path
        model_configurations = load_model_config(config_path=model_config_path)
        self.model_config_path = model_config_path
        self.model_config = model_configurations
        self.num_of_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimiser_name = optimiser
        self.miner = miner
        self.loss_function_name = loss_function

        self.datasets = Dataset(data_conf=load_data(data_config))
        self.model = VectorEmbedding_Transformer(d_numerical_integer=self.datasets.d_num_int,
                                                 d_numerical_continues=self.datasets.d_num_con,
                                                 d_token=model_configurations.d_tokens,
                                                 d_binary=self.datasets.d_bin,
                                                 categories=None,
                                                 token_bias=bias,
                                                 use_pretrained_bert=False,
                                                 n_layers=model_configurations.num_layers,
                                                 ffn_dim=model_configurations.ffn_dim,
                                                 head_num=model_configurations.head_num,
                                                 attention_hidden_dim=model_configurations.attention_hidden_dim,
                                                 attention_dropout=model_configurations.attention_dropout,
                                                 ffn_dropout=model_configurations.ffn_dropout,
                                                 activation=model_configurations.activation,
                                                 d_out=model_configurations.attention_hidden_dim)
        logger.info(self.model)

        self.loss_function = get_loss_function(loss_function_name=self.loss_function_name)

        self.optimiser_ = get_optimiser(parameters=self.model.parameters(),
                                        optimiser_name=optimiser,
                                        weight_decay=model_configurations.weight_decay,
                                        momentum=model_configurations.momentum,
                                        lr=self.lr)

        lf = lambda x: (1 - x / (num_epochs - 1)) * (1.0 - model_configurations.lrf) + model_configurations.lrf
        self.scheduler = lr_scheduler.LambdaLR(optimizer=self.optimiser_,
                                               lr_lambda=lf) if scheduler is not None else None

        logger.info(f'Training with epochs: {num_epochs}, batch size: {batch_size}, lr: {lr}, optimiser: {optimiser},'
                    f' loss_function: {self.loss_function} ')

    def get_parameters_with_lr(self):
        model_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in model_parameters if "bias" not in name]
        params_config = [{"params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                          'lr': self.lr,
                          'weight_decay': self.model_config.weight_decay},
                         {"params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                          'lr': self.lr,
                          'weight_decay': 0}]

        return params_config

    def save_plot(self, dict_values, filepath, name_plot='plot_1.png'):
        plt.figure(figsize=(16, 8))
        plt.title(name_plot)
        for k, v in dict_values.items():
            plt.plot(range(1, len(v) + 1), v, '.-', label=k)

        plt.legend()
        plt.savefig(filepath)

    def get_model_info_dict(self):
        model_info_dict = {
            # Token config
            'd_num_con': self.datasets.d_num_con,
            'd_num_int': self.datasets.d_num_int,
            'd_num_bin': self.datasets.d_bin,
            'd_cat': self.datasets.d_cat,
            'bias': self.model.tokenizer_fe.bias,
            # Transformer Config
            'd_token': self.model_config.d_tokens,
            'num_layers': self.model_config.num_layers,
            'head_num': self.model_config.head_num,
            'att_dropout': self.model_config.attention_dropout,
            'att_hidden_dim': self.model_config.attention_hidden_dim,
            'ffn_dropout': self.model_config.ffn_dropout,
            'ffn_dim': self.model_config.ffn_dim,
            'd_out': self.model_config.attention_hidden_dim,
            'activation': self.model_config.activation
        }

        return model_info_dict

    def save_ckpt(self, epoch_num, name_file, loss_value):
        if not os.path.exists(self.outpath):
            AssertionError(f'The output file path {self.outpath} have not exist')
        full_path = os.path.join(self.outpath, 'weights')
        if not os.path.exists(full_path):
            os.mkdir(full_path)
        full_out_path = os.path.join(full_path, name_file)

        logger.info(f'Saving model at epoch {epoch_num}, in file path {full_path} with filename {name_file}')

        model_info_dict = self.get_model_info_dict()

        torch.save({'epoch': epoch_num,
                    'total_num_of_epochs': self.num_of_epochs,
                    'lr': self.lr,
                    'loss_fun': self.loss_function_name,
                    'batch_size': self.batch_size,
                    'model_config_path': self.model_config_path,
                    'model_state_dict': self.model.state_dict(),
                    'optimiser_state_dict': self.optimiser_.state_dict(),
                    'optimiser_name': self.optimiser_name,
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'loss': loss_value,
                    'model_info': model_info_dict}, full_out_path)

    def load_model_ckpt(self, model_info_dict):
        return VectorEmbedding_Transformer(d_numerical_integer=model_info_dict['d_num_int'],
                                           d_numerical_continues=model_info_dict['d_num_con'],
                                           d_token=model_info_dict['d_token'],
                                           d_binary=model_info_dict['d_num_bin'],
                                           categories=None,
                                           token_bias=model_info_dict['bias'],
                                           use_pretrained_bert=False,
                                           n_layers=model_info_dict['num_layers'],
                                           ffn_dim=model_info_dict['ffn_dim'],
                                           head_num=model_info_dict['head_num'],
                                           attention_hidden_dim=model_info_dict['att_hidden_dim'],
                                           attention_dropout=model_info_dict['att_dropout'],
                                           ffn_dropout=model_info_dict['ffn_dropout'],
                                           activation=model_info_dict['activation'],
                                           d_out=model_info_dict['d_out'])

    def load_ckpt(self, path_model):
        full_path_model = os.path.join(os.getcwd(), path_model)
        if not os.path.exists(full_path_model):
            AssertionError(f'The checkpoint {full_path_model} have not exist')

        ckpt = torch.load(full_path_model)
        self.lr = ckpt['lr']
        self.loss_function_name = ckpt['loss_fun']
        self.num_of_epochs = ckpt['total_num_of_epochs']
        self.batch_size = ckpt['batch_size']
        epoch_num_ckpt = ckpt['epoch']

        self.model = self.load_model_ckpt(model_info_dict=ckpt['model_info'])
        self.model.load_state_dict(ckpt['model_state_dict'])

        self.model_config_path = ckpt['model_config_path']
        self.model_config = load_model_config(self.model_config_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimiser_name = ckpt['optimiser_name']
        self.optimiser_ = get_optimiser(parameters=self.model.parameters(),
                                        optimiser_name=self.optimiser_name,
                                        weight_decay=self.model_config.weight_decay,
                                        momentum=self.model_config.momentum,
                                        lr=self.lr)
        self.optimiser_.load_state_dict(ckpt['optimiser_state_dict'])
        print(self.optimiser_)
        if ckpt['scheduler_state_dict'] is not None:
            lf = lambda x: (1 - x / (self.num_of_epochs - 1)) * (1.0 - self.model_config.lrf) + self.model_config.lrf
            self.scheduler = lr_scheduler.LambdaLR(optimizer=self.optimiser_,
                                                   lr_lambda=lf)
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return epoch_num_ckpt
        # self.optimiser_

    def test(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                num_continuous_tuples, num_int_tuples, bin_tuples, cat_tuples, labels_y = batch_data
                labels_y = labels_y[0]
                labels_y = labels_y.type(torch.float64)
                labels_y = labels_y.to(self.model.device)

                embedding_transform_repr_proj = self.embedding_model(num_int_tuples, num_continuous_tuples, bin_tuples,
                                                   cat_tuples)
                if self.miner:
                    hard_pairs = miner(embedding_transform_repr_proj, labels_y)
                    # Optimise MLP
                    loss_ = self.loss_function(embedding_transform_repr_proj, labels_y, hard_pairs)
                else:
                    loss_ = self.loss_function(embedding_transform_repr_proj, labels_y)
                total_loss += loss_

            total_loss /= (batch_idx + 1)
        return total_loss

    def train(self):
        # epoch_tqdm_bar = tqdm(range(self.num_of_epochs))
        if self.miner:
            miner = miners.MultiSimilarityMiner()
            logger.info('Training with miner')
        loss_values = {}
        loss_values_val = {}

        for dataset_name in self.datasets.datasets_dict.keys():
            loss_values[dataset_name] = []
            loss_values_val[dataset_name] = []
        with tqdm(total=self.num_of_epochs) as pbar:
            for epoch in range(self.num_of_epochs):
                best_loss = 100000
                epoch_loss = 0
                for dataset_idx, dataset_name in enumerate(self.datasets.datasets_dict.keys()):
                    train_databuilder, test_data_builder, val_databuilder = self.datasets.get_Dataset_dataloader(
                        dataset_name=dataset_name)
                    train_dataloader = DataLoader(train_databuilder, batch_size=self.batch_size, shuffle=False)
                    self.model.train(mode=True)

                    total_loss = 0
                    for batch_idx, batch_data in enumerate(train_dataloader):
                        self.optimiser_.zero_grad()
                        num_continuous_tuples, num_int_tuples, bin_tuples, cat_tuples, labels_y = batch_data
                        labels_y = labels_y[0]
                        labels_y = labels_y.type(torch.float64)
                        labels_y = labels_y.to(self.model.device)

                        # labels_y.requires_grad = True
                        embedding_transform_repr_proj = self.model(num_int_tuples, num_continuous_tuples, bin_tuples,
                                                                   cat_tuples)
                        if self.miner:
                            hard_pairs = miner(embedding_transform_repr_proj, labels_y)
                            # Optimise MLP
                            loss_ = self.loss_function(embedding_transform_repr_proj, labels_y, hard_pairs)
                        else:
                            loss_ = self.loss_function(embedding_transform_repr_proj, labels_y)
                        total_loss += loss_.item()
                        epoch_loss += loss_.item()
                        epoch_loss /= 2

                        loss_.backward()
                        self.optimiser_.step()

                    total_loss /= (batch_idx + 1)

                    val_dataloader = DataLoader(val_databuilder, batch_size=self.batch_size, shuffle=False)
                    val_loss = self.test(val_dataloader)

                    loss_values[dataset_name].append(total_loss)
                    acc_values[dataset_name].append(total_acc)
                    loss_values_val[dataset_name].append(val_loss)

                    pbar.set_description(f'Epoch :{epoch}, Dataset name: {dataset_name},'
                                         f' train loss: {total_loss}, validation loss: {val_loss}')
                    logger.info(str(pbar))

                if self.scheduler is not None:
                    self.scheduler.step()

                if epoch_loss < best_loss:
                    self.save_ckpt(epoch_num=epoch,
                                   loss_value=epoch_loss,
                                   name_file='best.pt')
                if epoch % 10 == 0:
                    self.save_ckpt(epoch_num=epoch,
                                   loss_value=epoch_loss,
                                   name_file=f'ckpt_epoch_{epoch}.pt')
                pbar.update(1)

            self.save_plot(loss_values, os.path.join(self.outpath, 'loss.png'), 'Loss per dataset')
            self.save_plot(acc_values, os.path.join(self.outpath, 'acc.png'), 'Accuracy per dataset')
