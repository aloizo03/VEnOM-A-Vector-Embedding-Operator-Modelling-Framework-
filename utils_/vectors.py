import torch
from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset
from models.vectorEmbedding import Num2Vec
from models.torch_utils import select_device
import joblib

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import logging
import os
import time
import pickle


logger = logging.getLogger(__name__)


class Vectorise:

    def __init__(self, model_path, data_path, out_path, data_type):
        self.data_type = data_type

        if self.data_type == 1:
            ckpt = torch.load(model_path)

            model_info = ckpt['model_info']
            input_dim = model_info['d_numerical']
            d_out = model_info['d_out']
            d_tokens = model_info['d_token']
            num_layers = model_info['num_layers']
            head_num = model_info['head_num']
            attention_dropout = model_info['att_dropout']
            ffn_dropout = model_info['ffn_dropout']
            ffn_dim = model_info['ffn_dim']
            activation = model_info['activation']
            dim_k = model_info['dim_k']
            table_input = model_info['ret_table']

            self.model = Num2Vec(d_numerical=input_dim,
                                d_out=d_out,  # create a 3d representation
                                n_layers=num_layers,
                                d_token=d_tokens,
                                head_num=head_num,
                                attention_dropout=attention_dropout,
                                ffn_dropout=ffn_dropout,
                                ffn_dim=ffn_dim,
                                activation=activation,
                                train=False,
                                input_table=table_input,
                                d_k=dim_k)

            self.model.load_state_dict(ckpt['model_state_dict'])
            self.dataset = Pipeline_Dataset(data_path=data_path,
                                            norm=True,
                                            ret_table=table_input,
                                            k_dim=dim_k)
        elif self.data_type == 2:
            print(model_path)
            self.model = joblib.load(model_path)
            self.model.epochs = 1
            self.dataset = graph_Pipeline_Dataset(data_path=data_path,
                                                  return_label=False)

        self.output_path = out_path
        self.device = select_device()
        if torch.cuda.device_count() > 1 and self.data_type == 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)

        logging.basicConfig(filename=os.path.join(self.output_path, 'log_file.log'), encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)

    def convert_vectors_to_out_list(self, vectors, vectors_dataset_idx, data_builder):
        out_data = {}
        for vector, dataset_idx in zip(vectors, vectors_dataset_idx):
            dataset_name = data_builder.get_dataset_name(dataset_idx)
            out_data[dataset_name] = vector
        return out_data

    def plot_vectors(self, vectors):
        plt.set_loglevel('info')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pca = PCA(n_components=3)
        result = pca.fit_transform(vectors)

        ax.scatter(result[:, 0], result[:, 1], result[:, 2], c='blueviolet')
        ax.set(xlabel=(r' $x$'), ylabel=(r'$y$'), zlabel=(r'$z$'))
        plt.savefig(os.path.join(self.output_path, 'plot_representation.png'), dpi=100)

    def save_vectors(self, dict_, file_name=None):
        if file_name is None:
            file_name = os.path.join(self.output_path, f'vectors_{time.time()}.pickle')
        with open(file_name, 'wb') as handle:
            pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_vectors(self, batch_size, num_of_workes=0, show_progress=True, ret_vec=False):
        DataBuilder = self.dataset.get_Dataloader()

        if self.data_type == 1:
            dataloader = DataLoader(DataBuilder, batch_size=batch_size, shuffle=False, num_workers=num_of_workes)
            vectors_output = []
            vectors_dataset_idx = []
            with tqdm(total=DataBuilder.data_len) as pbar:
                total_data = 0
                with torch.no_grad():
                    for batch_idx, batch_ in enumerate(dataloader):
                        total_data += batch_size
                        features_, dataset_idx = batch_
                        features_ = features_.to(self.device)
                        _, vectors, _ = self.model(features_)
                        
                        vectors_output.append(vectors.detach().cpu().numpy())
                        vectors_dataset_idx.append(dataset_idx.numpy())

                        pbar.update(batch_size)
                        if batch_idx % 10 == 0:
                            pbar.set_description(f'Data Batch id {batch_idx}, total data vectorised {total_data}/{DataBuilder.data_len}')
                            logger.info(str(pbar))
                vectors_out_np = np.concatenate(vectors_output, axis=0)
                vectors_dataset_idx_np = np.concatenate(vectors_dataset_idx, axis=0)
        elif self.data_type ==2:
            data_graphs = DataBuilder.get_all_data()
            vectors_out_np = self.model.infer(graphs=data_graphs)
            vectors_dataset_idx_np = np.asarray(list(DataBuilder.idx_to_filename.keys()))


        out_vectors_dict = self.convert_vectors_to_out_list(vectors=vectors_out_np,
                                         vectors_dataset_idx=vectors_dataset_idx_np, 
                                         data_builder=DataBuilder)
                
        if not ret_vec:
            self.plot_vectors(vectors_out_np)
            self.save_vectors(out_vectors_dict)
        else:
            return out_vectors_dict



