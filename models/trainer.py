import matplotlib.pyplot as plt
import numpy as np
import logging
from utils_.utils import calculate_accuracy, check_path
from data.data_utils import collator_fn
from pytorch_metric_learning import miners
from models.Loss import Loss_Compute, RMSELoss
from models.vectorEmbedding import Num2Vec
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

class Trainer:

    def __init__(self,
                 data_config,
                 model_config_path,
                 out_path,
                 num_epochs,
                 batch_size,
                 lr,
                 bias,
                 optimiser,
                 loss_function,
                 scheduler=False,
                 continue_ckpt=None):

        logging.basicConfig(filename=os.path.join(out_path, 'log_file.log'), encoding='utf-8', level=logging.DEBUG)
        logger.setLevel(logging.INFO)

        if continue_ckpt is not None:
            self.load_model_ckpt(ckpt_path=continue_ckpt)
            start_epoch, best_loss, loss_dict = self.load_ckpt_train_info(ckpt_path=continue_ckpt)
            logger.info(self.model)
            logger.info(
                f'Continue training at epoch {start_epoch} for total epoch {self.num_of_epochs},'
                f' batch size: {self.batch_size}, lr: {self.lr}, optimiser: {self.optimiser_name},'
                f' loss_function: {self.loss_function} ')
        else:
            self.lr_scheduler = scheduler
            self.outpath = out_path
            model_configurations = load_model_config(config_path=model_config_path)
            self.model_config_path = model_config_path
            self.model_config = model_configurations
            self.num_of_epochs = num_epochs
            self.batch_size = batch_size
            self.lr = lr
            self.optimiser_name = optimiser
            self.loss_function_name = loss_function
            self.data_config_path = data_config

            self.datasets = Dataset(data_conf=load_data(data_config),
                                    norm=True)

            self.model = Num2Vec(d_numerical=20,
                                 d_out=3,  # create a 3d representation
                                 n_layers=self.model_config.num_layers,
                                 d_token=self.model_config.d_tokens,
                                 head_num=self.model_config.head_num,
                                 attention_dropout=self.model_config.attention_dropout,
                                 ffn_dropout=self.model_config.ffn_dropout,
                                 ffn_dim=self.model_config.ffn_dim,
                                 activation=self.model_config.activation,
                                 train=True)

            self.loss_function = get_loss_function(loss_function_name=self.loss_function_name, batch_size=batch_size,
                                                   device=self.model.device, embedding_size=self.model_config.d_out)

            self.optimiser_ = get_optimiser(parameters=self.get_parameters_with_lr(),
                                            optimiser_name=optimiser,
                                            weight_decay=model_configurations.weight_decay,
                                            momentum=model_configurations.momentum,
                                            lr=self.lr)

            lf = lambda x: (1 - x / (num_epochs - 1)) * (1.0 - model_configurations.lrf) + model_configurations.lrf
            self.scheduler = lr_scheduler.LambdaLR(optimizer=self.optimiser_,
                                                   lr_lambda=lf) if self.lr_scheduler is not None else None
            logger.info(self.model)
            logger.info(
                f'Training with epochs: {num_epochs}, batch size: {batch_size}, lr: {lr}, optimiser: {optimiser},'
                f' loss_function: {self.loss_function} ')

    def get_parameters_with_lr(self):
        model_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        # TODO Check what happens when you have a decoder and encoder parameter opti
        decoder_parameters = [name for name in model_parameters if 'decoder' in name]
        encoder_parameters = [name for name in model_parameters if 'decoder' not in name]
        decay_parameters = [name for name in model_parameters if "bias" not in name]
        params_config = [{"params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                          'lr': self.lr,
                          'weight_decay': self.model_config.weight_decay},
                         {"params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                          'lr': self.lr,
                          'weight_decay': 0}]

        return params_config

    def save_plot(self, dict_values, filepath, name_plot='plot_1.png'):
        plt.set_loglevel('info')
        plt.figure(figsize=(16, 8))
        plt.title(name_plot)
        for k, v in dict_values.items():
            plt.plot(range(1, len(v) + 1), v, '.-', label=k)

        plt.legend()
        plt.savefig(filepath)

    def get_model_info_dict(self):
        model_info_dict = {
            # Token config
            'd_numerical': self.model.dim_numerical,
            'bias': False if self.model.numerical_embedding.bias is None else True,
            # Transformer Config
            'd_token': self.model_config.d_tokens,
            'num_layers': self.model_config.num_layers,
            'head_num': self.model_config.head_num,
            'att_dropout': self.model_config.attention_dropout,
            'att_hidden_dim': self.model_config.attention_hidden_dim,
            'ffn_dropout': self.model_config.ffn_dropout,
            'ffn_dim': self.model_config.ffn_dim,
            'd_out': self.model.d_out,
            'activation': self.model_config.activation
        }

        return model_info_dict

    def save_ckpt(self, epoch_num, name_file, loss_value, best_loss):
        # TODO: Change it
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
                    'scheduler': self.scheduler.state_dict() if self.lr_scheduler is not None else None,
                    'loss_fun': self.loss_function_name,
                    'batch_size': self.batch_size,
                    'model_config_path': self.model_config_path,
                    'model_state_dict': self.model.state_dict(),
                    'optimiser_state_dict': self.optimiser_.state_dict(),
                    'optimiser_name': self.optimiser_name,
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'best_loss': best_loss,
                    'loss': loss_value,
                    'model_info': model_info_dict,
                    'data_config': self.data_config_path}, full_out_path)

    def load_ckpt_train_info(self, ckpt_path):
        check_path(path=ckpt_path)

        ckpt = torch.load(ckpt_path)

        continue_epoch = ckpt['epoch']
        best_loss = ckpt['best_loss']
        loss_dict = ckpt['loss']
        return continue_epoch, best_loss, loss_dict

    def load_model_ckpt(self, ckpt_path):
        check_path(path=ckpt_path)

        ckpt = torch.load(ckpt_path)

        self.lr_scheduler = None if ckpt['scheduler'] is None else True
        # self.out_path =
        model_configurations = load_model_config(config_path=ckpt['model_config_path'])
        self.model_config_path = ckpt['model_config_path']
        self.model_config = model_configurations
        self.num_of_epochs = ckpt['total_num_of_epochs']
        self.batch_size = ckpt['batch_size']
        self.lr = ckpt['lr']
        self.optimiser_name = ckpt['optimiser_name']
        self.loss_function_name = ckpt['loss_fun']

        model_info = ckpt['model_info']
        self.model_config.d_out = model_info['d_out']
        self.model_config.d_tokens = model_info['d_token']
        self.model_config.num_layers = model_info['num_layers']
        self.model_config.head_num = model_info['head_num']
        self.model_config.attention_dropout = model_info['att_dropout']
        self.model_config.attention_hidden_dim = model_info['att_hidden_dim']
        self.model_config.ffn_dropout = model_info['ffn_dropout']
        self.model_config.ffn_dim = model_info['ffn_dim']
        self.model_config.activation = model_info['activation']

        self.datasets = Dataset(data_conf=load_data(ckpt['data_config']),
                                norm=True)

        self.model = Num2Vec(d_numerical=model_info['d_numerical'],
                             d_out=self.model_config.d_out,  # create a 3d representation
                             n_layers=self.model_config.num_layers,
                             d_token=self.model_config.d_tokens,
                             head_num=self.model_config.head_num,
                             attention_dropout=self.model_config.attention_dropout,
                             ffn_dropout=self.model_config.ffn_dropout,
                             ffn_dim=self.model_config.ffn_dim,
                             activation=self.model_config.activation,
                             train=True)
        self.model.load_state_dict(ckpt['model_state_dict'])

        self.loss_function = get_loss_function(loss_function_name=self.loss_function_name, batch_size=self.batch_size,
                                               device=self.model.device, embedding_size=self.model_config.d_out)

        self.optimiser_ = get_optimiser(parameters=self.get_parameters_with_lr(),
                                        optimiser_name=self.optimiser_name,
                                        weight_decay=self.model_config.weight_decay,
                                        momentum=self.model_config.momentum,
                                        lr=self.lr)
        self.optimiser_.load_state_dict(ckpt['optimiser_state_dict'])

    def test(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_KLD_loss = 0
        total_MSE_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(self.model.device)
                recon_batch, mu, logvar = self.model(batch_data, True)
                recon_batch = recon_batch[:, :batch_data.shape[1]]
                loss_MSE, loss_KLD = self.loss_function(recon_batch, batch_data, mu, logvar)
                loss = loss_MSE + loss_KLD

                total_loss += loss.item()
                total_KLD_loss += loss_KLD.item()
                total_MSE_loss += loss_MSE.item()
            total_loss /= (batch_idx + 1)
            total_KLD_loss /= (batch_idx + 1)
            total_MSE_loss /= (batch_idx + 1)
        return total_loss, total_MSE_loss, total_KLD_loss

    def train(self, continue_train_ckpt=None):

        start_epoch, best_loss = 0, 1e8
        loss_values = {}
        MSE_loss_values = {}
        KLD_loss_values = {}
        loss_values_val = {}
        MSE_loss_values_val = {}
        KLD_loss_values_val = {}
        for dataset_name in self.datasets.datasets_dict.keys():
            loss_values[dataset_name] = []
            MSE_loss_values[dataset_name] = []
            KLD_loss_values[dataset_name] = []
            loss_values_val[dataset_name] = []
            MSE_loss_values_val[dataset_name] = []
            KLD_loss_values_val[dataset_name] = []

        if continue_train_ckpt is not None:
            start_epoch, best_loss, loss_dict = self.load_ckpt_train_info(ckpt_path=continue_train_ckpt)
            MSE_loss_values = loss_dict['MSE_loss']
            KLD_loss_values = loss_dict['KLD_loss']
            loss_values_val = loss_dict['total_loss']
            MSE_loss_values_val = loss_dict['MSE_loss_val']
            KLD_loss_values_val = loss_dict['KLD_loss_val']

        with tqdm(total=self.num_of_epochs) as pbar:
            pbar.update(start_epoch)
            for epoch in range(start_epoch, self.num_of_epochs):
                epoch_loss = 0
                for dataset_idx, dataset_name in enumerate(self.datasets.datasets_dict.keys()):
                    train_databuilder, test_data_builder, val_databuilder = self.datasets.get_Dataset_dataloader(
                        dataset_name=dataset_name)
                    train_dataloader = DataLoader(train_databuilder, batch_size=self.batch_size, shuffle=True)
                    self.model.train(mode=True)

                    total_loss = 0
                    total_MSE_loss = 0
                    total_KLD_loss = 0
                    for batch_idx, batch_data in enumerate(train_dataloader):
                        batch_data = batch_data.to(self.model.device)
                        self.optimiser_.zero_grad()
                        recon_batch, mu, logvar = self.model(batch_data, True)
                        recon_batch = recon_batch[:, :batch_data.shape[1]]
                        loss_MSE, loss_KLD = self.loss_function(recon_batch, batch_data, mu, logvar)
                        loss = loss_MSE + loss_KLD
                        loss.backward()

                        epoch_loss += loss.item()
                        total_loss += loss.item()
                        total_KLD_loss += loss_KLD.item()
                        total_MSE_loss += loss_MSE.item()
                        self.optimiser_.step()
                        if batch_idx % 10 == 0:
                            pbar.set_description(f'Epoch :{epoch}, Dataset name: {dataset_name}, batch ID: {batch_idx}'
                                                 f' MSE loss: {loss_MSE.item()}, KLD loss: {loss_KLD.item()}')
                    total_loss /= (batch_idx + 1)
                    total_KLD_loss /= (batch_idx + 1)
                    total_MSE_loss /= (batch_idx + 1)
                    epoch_loss /= (batch_idx + 1)

                    loss_values[dataset_name].append(total_loss)
                    MSE_loss_values[dataset_name].append(total_MSE_loss)
                    KLD_loss_values[dataset_name].append(total_KLD_loss)

                    val_dataloader = DataLoader(val_databuilder, batch_size=self.batch_size, shuffle=False)
                    val_loss, val_MSE_loss, val_KLD_loss = self.test(val_dataloader)

                    loss_values_val[dataset_name].append(val_loss)
                    MSE_loss_values_val[dataset_name].append(val_MSE_loss)
                    KLD_loss_values_val[dataset_name].append(val_KLD_loss)

                    pbar.set_description(f'Epoch :{epoch}, Dataset name: {dataset_name},'
                                         f' train loss: {total_loss}, validation loss: {val_loss}')
                    logger.info(str(pbar))
                    if self.scheduler is not None:
                        self.scheduler.step()

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save_ckpt(epoch_num=epoch,
                                   loss_value={'MSE_loss': MSE_loss_values,
                                               'KLD_loss': KLD_loss_values,
                                               'total_loss': total_loss,
                                               'total_loss_val': loss_values_val,
                                               'MSE_loss_val': MSE_loss_values_val,
                                               'KLD_loss_val': KLD_loss_values_val},
                                   best_loss=best_loss,
                                   name_file='best.pt')

                if epoch % 10 == 0:
                    self.save_ckpt(epoch_num=epoch,
                                   loss_value={'MSE_loss': MSE_loss_values,
                                               'KLD_loss': KLD_loss_values,
                                               'total_loss': total_loss,
                                               'total_loss_val': loss_values_val,
                                               'MSE_loss_val': MSE_loss_values_val,
                                               'KLD_loss_val': KLD_loss_values_val},
                                   best_loss=best_loss,
                                   name_file=f'ckpt_epoch_{epoch}.pt')
                pbar.update(1)

            test_loss_out = ''
            for dataset_idx, dataset_name in enumerate(self.datasets.datasets_dict.keys()):
                train_databuilder, test_data_builder, val_databuilder = self.datasets.get_Dataset_dataloader(
                    dataset_name=dataset_name)
                test_dataloader = DataLoader(train_databuilder, batch_size=self.batch_size, shuffle=True)
                loss, MSE_loss, KLD_loss = self.test(dataloader=test_dataloader)
                test_loss_out += (f'Dataset name: {dataset_name},'
                                  f'  MSE loss: {MSE_loss},'
                                  f' KLD loss: {KLD_loss}, total loss: {loss}\n')

            logger.info('Testing Loss: \n' + test_loss_out)

            self.save_plot(loss_values, os.path.join(self.outpath, 'loss.png'), 'Loss per dataset')
            self.save_plot(MSE_loss_values, os.path.join(self.outpath, 'loss_MSE.png'), 'MSE Loss per dataset')
            self.save_plot(KLD_loss_values, os.path.join(self.outpath, 'loss_KLD.png'), 'KLD Loss per dataset')
            self.save_plot(loss_values_val, os.path.join(self.outpath, 'val_loss.png'), 'Validation per dataset')
            self.save_plot(MSE_loss_values_val, os.path.join(self.outpath, 'val_loss_MSE.png'),
                           'Validation MSE Loss per dataset')
            self.save_plot(KLD_loss_values_val, os.path.join(self.outpath, 'val_loss_KLD.png'),
                           'Validation KLD Loss per dataset')
