import matplotlib.pyplot as plt
import numpy as np
import logging
from models.torch_utils import select_device
from utils_.utils import check_path
from models.vectorEmbedding import Num2Vec
from models.utils import load_model_config, get_optimiser, get_loss_function, get_parameter_names

from data.Dataset import load_data, Dataset
from data.data_utils import collator_fn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import random

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

        if continue_ckpt is not None:
            self.load_model_ckpt(ckpt_path=continue_ckpt)
            logging.basicConfig(filename=os.path.join(self.outpath, 'log_file.log'), encoding='utf-8',
                                level=logging.DEBUG)
            logger.setLevel(logging.INFO)
            start_epoch, best_loss, loss_dict = self.load_ckpt_train_info(ckpt_path=continue_ckpt)
            logger.info(self.model)
            logger.info(
                f'Continue training at epoch {start_epoch} for total epoch {self.num_of_epochs},'
                f' batch size: {self.batch_size}, lr: {self.lr}, optimiser: {self.optimiser_name},'
                f' loss_function: {self.loss_function} ')
        else:
            logging.basicConfig(filename=os.path.join(out_path, 'log_file.log'), encoding='utf-8', level=logging.DEBUG)
            logger.setLevel(logging.INFO)

            self.bias_ = bias
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

            self.ret_all_dataset = self.model_config.ret_all_dataset

            self.datasets = Dataset(data_conf=load_data(data_config),
                                    norm=True,
                                    ret_table=self.model_config.table_input,
                                    ret_all_dataset=self.model_config.ret_all_dataset,
                                    k_dim=self.model_config.dim_k)

            if self.ret_all_dataset:
                self.model_config.dim_k = self.datasets.k_dim

            self.model = Num2Vec(d_numerical=self.model_config.input_dim,
                                 d_out=self.model_config.d_tokens,  # create a k-d representation
                                 n_layers=self.model_config.num_layers,
                                 d_token=self.model_config.d_tokens,
                                 head_num=self.model_config.head_num,
                                 attention_dropout=self.model_config.attention_dropout,
                                 ffn_dropout=self.model_config.ffn_dropout,
                                 ffn_dim=self.model_config.ffn_dim,
                                 activation=self.model_config.activation,
                                 train=True,
                                 input_table=self.model_config.table_input,
                                 ret_all_dataset=self.model_config.ret_all_dataset,
                                 d_k=self.model_config.dim_k,
                                 distribution='vmf')

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

            # self.create_decoder_per_dataset()
            logger.info(self.model)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
            self.device = select_device(batch_size=self.batch_size)

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

    def create_decoder_per_dataset(self):
        self.decoders_weights = {}

        for dataset_name, dict_values in self.datasets.datasets_dict.items():
            self.decoders_weights[dataset_name] = self.model.decoder.state_dict()

    def set_decoder_dataset(self, dataset_name):
        self.model.decoder.load_state_dict(self.decoders_weights[dataset_name])

    def save_decoder_dataset(self, dataset_name):
        self.decoders_weights[dataset_name] = self.model.decoder.state_dict()

    def save_plot(self, dict_values, filepath, name_plot='plot_1.png'):
        plt.set_loglevel('info')
        plt.figure(figsize=(16, 8))
        plt.title(name_plot)
        if type(dict_values) is dict:
            for k, v in dict_values.items():
                plt.plot(range(1, len(v) + 1), v, '.-', label=k)
        else:
            plt.plot(range(1, len(dict_values) + 1), dict_values, '.-')
        plt.legend()
        plt.savefig(filepath)

    def get_model_info_dict(self):
        model_info_dict = {
            # Token config
            'd_numerical': self.model_config.input_dim,
            'bias': False if self.bias_ is None else True,
            # Transformer Config
            'd_token': self.model_config.d_tokens,
            'num_layers': self.model_config.num_layers,
            'head_num': self.model_config.head_num,
            'att_dropout': self.model_config.attention_dropout,
            'att_hidden_dim': self.model_config.attention_hidden_dim,
            'ffn_dropout': self.model_config.ffn_dropout,
            'ffn_dim': self.model_config.ffn_dim,
            'd_out': self.model_config.d_tokens,
            'activation': self.model_config.activation,
            'ret_table': self.model_config.table_input,
            'read_all_dataset': self.model_config.ret_all_dataset,
            'dim_k': self.model_config.dim_k
        }

        return model_info_dict

    def save_ckpt(self, epoch_num, name_file, loss_value, best_loss):
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
                    'model_state_dict': self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict(),
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

        self.model_config.ret_all_dataset = model_info['read_all_dataset']
        self.model_config.d_out = model_info['d_out']
        self.model_config.d_tokens = model_info['d_token']
        self.model_config.num_layers = model_info['num_layers']
        self.model_config.head_num = model_info['head_num']
        self.model_config.attention_dropout = model_info['att_dropout']
        self.model_config.attention_hidden_dim = model_info['att_hidden_dim']
        self.model_config.ffn_dropout = model_info['ffn_dropout']
        self.model_config.ffn_dim = model_info['ffn_dim']
        self.model_config.activation = model_info['activation']
        self.model_config.dim_k = model_info['dim_k']
        self.model_config.table_input = model_info['ret_table']

        self.datasets = Dataset(data_conf=load_data(ckpt['data_config']),
                                norm=True,
                                ret_all_dataset=self.model_config.ret_all_dataset,
                                ret_table=self.model_config.table_input,
                                k_dim=self.model_config.dim_k)

        self.model = Num2Vec(d_numerical=model_info['d_numerical'],
                             d_out=self.model_config.d_out,  # create a 3d representation
                             n_layers=self.model_config.num_layers,
                             d_token=self.model_config.d_tokens,
                             head_num=self.model_config.head_num,
                             attention_dropout=self.model_config.attention_dropout,
                             ffn_dropout=self.model_config.ffn_dropout,
                             ffn_dim=self.model_config.ffn_dim,
                             activation=self.model_config.activation,
                             train=True,
                             input_table=self.model_config.table_input,
                             d_k=self.model_config.dim_k,
                             ret_all_dataset=self.ret_all_dataset)

        self.model.load_state_dict(ckpt['model_state_dict'])

        self.loss_function = get_loss_function(loss_function_name=self.loss_function_name, batch_size=self.batch_size,
                                               device=self.model.device, embedding_size=self.model_config.d_out)

        self.optimiser_ = get_optimiser(parameters=self.get_parameters_with_lr(),
                                        optimiser_name=self.optimiser_name,
                                        weight_decay=self.model_config.weight_decay,
                                        momentum=self.model_config.momentum,
                                        lr=self.lr)
        self.optimiser_.load_state_dict(ckpt['optimiser_state_dict'])

    def freeze_model_parameters(self, layer_index):
        logger.info('Fine tuning is apply')
        for i, layers_ in enumerate(self.model.Transformer_layers.transformer_layers):
            if i in layer_index:
                for param in layers_.parameters():
                    param.requires_grad = False

    def find_model_parameters_indexes(self, percentage_of_freeze_layers=0.5):
        layer_index = random.sample(range(0, len(self.model.Transformer_layers.transformer_layers)),
                                    int(len(
                                        self.model.Transformer_layers.transformer_layers) * percentage_of_freeze_layers))

        return layer_index

    def test(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_KLD_loss = 0
        total_MSE_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(self.device)
                recon_batch, mu, logvar = self.model(batch_data)
                recon_batch = recon_batch[:, :batch_data.shape[1]]
                mse_loss, kl_loss = self.loss_function(recon_batch, batch_data, mu, logvar)
                loss = mse_loss + kl_loss

                total_loss += loss.item()
                total_KLD_loss += kl_loss.item()
                total_MSE_loss += mse_loss.item()
            total_loss /= (batch_idx + 1)
            total_KLD_loss /= (batch_idx + 1)
            total_MSE_loss /= (batch_idx + 1)
        return total_loss, total_MSE_loss, total_KLD_loss

    def train(self, continue_train_ckpt=None):
        start_epoch, best_loss = 0, 1e8
        loss_values = []
        MSE_loss_values = []
        KLD_loss_values = []
        loss_values_val = []
        MSE_loss_values_val = []
        KLD_loss_values_val = []

        if continue_train_ckpt is not None:
            start_epoch, best_loss, loss_dict = self.load_ckpt_train_info(ckpt_path=continue_train_ckpt)
            MSE_loss_values = loss_dict['MSE_loss']
            KLD_loss_values = loss_dict['KLD_loss']
            loss_values_val = loss_dict['total_loss']
            MSE_loss_values_val = loss_dict['MSE_loss_val']
            KLD_loss_values_val = loss_dict['KLD_loss_val']
        train_databuilder, test_data_builder, val_databuilder = self.datasets.get_Dataset_dataloader(dataset_name=None,
                                                                                                     all_datasets=True)
        with tqdm(total=self.num_of_epochs) as pbar:
            pbar.update(start_epoch)
            for epoch in range(start_epoch, self.num_of_epochs):
                epoch_loss = 0

                train_dataloader = DataLoader(train_databuilder, batch_size=self.batch_size, shuffle=True, collate_fn=collator_fn)

                self.model.train(mode=True)

                total_loss = 0
                total_MSE_loss = 0
                total_KLD_loss = 0
                # self.set_decoder_dataset(dataset_name=dataset_name)
                for batch_idx, batch_data in enumerate(train_dataloader):
                    batch_data = batch_data.to(self.device)
                    self.optimiser_.zero_grad()
                    
                    recon_batch, mu, logvar = self.model(batch_data)
                    recon_batch = recon_batch[:, :batch_data.shape[1]]

                    mse_loss, kl_loss = self.loss_function(recon_batch, batch_data, mu, logvar)
                    loss = mse_loss + kl_loss

                    loss.backward()
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    total_KLD_loss += kl_loss.item()
                    total_MSE_loss += mse_loss.item()
                    self.optimiser_.step()
                    if batch_idx % 10 == 0:
                        pbar.set_description(f'Epoch :{epoch}, batch ID: {batch_idx} '
                                             f'Total Loss: {(mse_loss.item() + kl_loss.item())}, MSE loss: {mse_loss.item()}, KLD loss: {kl_loss.item()}')
                total_loss /= (batch_idx + 1)
                total_KLD_loss /= (batch_idx + 1)
                total_MSE_loss /= (batch_idx + 1)
                epoch_loss /= (batch_idx + 1)

                loss_values.append(total_loss)

                MSE_loss_values.append(total_MSE_loss)
                KLD_loss_values.append(total_KLD_loss)

                val_dataloader = DataLoader(val_databuilder, batch_size=self.batch_size, shuffle=False, collate_fn=collator_fn)

                val_loss, val_MSE_loss, val_KLD_loss = self.test(val_dataloader)
                loss_values_val.append(val_loss)

                MSE_loss_values_val.append(val_MSE_loss)
                KLD_loss_values_val.append(val_KLD_loss)
                # self.save_decoder_dataset(dataset_name=dataset_name)
                pbar.set_description(f'Epoch :{epoch},'
                                     f' train loss: {total_loss}, validation loss: {val_loss}')
                logger.info(str(pbar))
                if self.scheduler is not None:
                    self.scheduler.step()
                if val_loss < best_loss:
                    best_loss = val_loss
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
                    dataset_name=dataset_name, all_datasets=False)
                test_dataloader = DataLoader(test_data_builder, batch_size=self.batch_size, shuffle=True, collate_fn=collator_fn)
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
