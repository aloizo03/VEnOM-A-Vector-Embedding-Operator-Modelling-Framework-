import torch
from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset, Image_Pipeline_Dataset
from models.vectorEmbedding import Num2Vec
from models.torch_utils import select_device

from utils_.utils import get_file_modified_time
import joblib
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from data.data_utils import collator_fn

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

import matplotlib.pyplot as plt
import logging
import os
import time
import pickle
from server.server_utils.qdrant_controller import qdrant_controller

from transformers import pipeline

logger = logging.getLogger(__name__)

class Dataset_Evolution:

    def __init__(self, data_name, data_path, model_path, out_path, data_type, data_times_path=None, use_vectorDB=True, d_token=100):
        if use_vectorDB:
            self.save_to = 'vectorDB'
            self.use_vectorDB = True
        else:
            self.save_to = 'local'
            self.use_vectorDB = False

        self.qdrant_controller = None
        if self.use_vectorDB:
            self.qdrant_controller = qdrant_controller()

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

            self.table_input = table_input
            self.dim_k = dim_k

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
            if torch.cuda.device_count() > 1 and self.data_type == 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
            
            self.vector_dim = d_tokens
        elif self.data_type == 2:
            self.model = joblib.load(model_path)
            self.model.epochs = 1
            self.vector_dim = d_token
        elif self.data_type == 3:    
            self.model = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", pool=True, device='cuda', use_fast=True, max_new_tokens=d_token)
            
            self.vector_dim = d_token

        self.data_path = data_path
        self.dataset_latest_time = data_times_path
        self.output_path = out_path
        self.collection_name = data_name
        self.device = select_device()

        logging.basicConfig(filename=os.path.join(self.output_path, 'log_file.log'), encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)

    def get_time(self):
        datasets_dir_list = os.listdir(self.data_path)
        out_dict = {}
        for dataset_path in datasets_dir_list:
            out_dict[dataset_path] = get_file_modified_time(os.path.join(self.data_path, dataset_path))

        return out_dict
    
    def get_vectors_time(self):
        if self.use_vectorDB:
            vectors_time = self.qdrant_controller.get_vectors_time(collection_name=self.collection_name)
        else:
            vectors_time = pickle.load(self.dataset_latest_time)

        return vectors_time

    def find_updated_dataset(self, dataset_times, new_dataset_time):
        vectors_dataset_names = list(dataset_times.keys())
        datasets_to_recompute = []
        for dataset_name, dataset_time in new_dataset_time.items():
            if os.path.join(self.data_path, dataset_name) in vectors_dataset_names:
                old_time = dataset_times[os.path.join(self.data_path, dataset_name)]
                if dataset_time > old_time and np.abs(dataset_time - old_time) > 0.01:
                    
                    datasets_to_recompute.append(os.path.join(self.data_path, dataset_name))
            else:
                datasets_to_recompute.append(os.path.join(self.data_path, dataset_name))

        return datasets_to_recompute
    
    def load_dict(self, path_):
        with open(path_, 'rb') as handle:
            b = pickle.load(handle)
        return b
    
    def save_updated_vectors_locally(self, vectors_dict, times_dict):
        dataset_vectors = self.load_dict(self.collection_name)
        dataset_times = self.load_dict(self.dataset_latest_time)

        for dataset_name, vector in vectors_dict.items():
            new_time = times_dict[dataset_name]
            dataset_vectors[dataset_name] = vector
            dataset_times[dataset_name] = new_time

        with open(self.collection_name, 'wb') as handle:
                pickle.dump(dataset_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.dataset_latest_time, 'wb') as handle:
                pickle.dump(dataset_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_updated_vectors(self, vectors_out, vectors_idx, data_builder):
        vectors_dict = {}
        times_dict = {}
        for vector, dataset_idx in zip(vectors_out, vectors_idx):
            dataset_name = data_builder.get_dataset_name(dataset_idx)
            vectors_dict[dataset_name] = vector
            times_dict[dataset_name] = get_file_modified_time(dataset_name)

        if self.use_vectorDB:
            self.qdrant_controller.upsert_collection(self.collection_name, vectors_dict, times_dict)
        else:
            self.save_updated_vectors_locally()

    def update_vectors(self, batch_size, num_of_workes=0, show_progress=True, ret_vec=False):
        vectors_times_dict = self.get_vectors_time()
        datasets_updated_times_dict = self.get_time()

        datasets_to_be_updated_list = self.find_updated_dataset(dataset_times=vectors_times_dict,
                                                                new_dataset_time=datasets_updated_times_dict)


        if self.data_type == 1:
            self.dataset = Pipeline_Dataset(data_path=datasets_to_be_updated_list,
                                            norm=True,
                                            ret_table=self.table_input,
                                            k_dim=self.dim_k)
            
            DataBuilder = self.dataset.get_Dataloader()
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
                        if batch_idx % 10 == 0 and show_progress:
                            pbar.set_description(f'Data Batch id {batch_idx}, total data vectorised {total_data}/{DataBuilder.data_len}')
                            logger.info(str(pbar))
                vectors_out_np = np.concatenate(vectors_output, axis=0)
                vectors_dataset_idx_np = np.concatenate(vectors_dataset_idx, axis=0)
        elif self.data_type == 2:
            self.dataset = graph_Pipeline_Dataset(data_path=datasets_to_be_updated_list,
                                                  return_label=False)
            
            DataBuilder = self.dataset.get_Dataloader()
            data_graphs = DataBuilder.get_all_data()
            vectors_out_np = self.model.infer(graphs=data_graphs)
            vectors_dataset_idx_np = np.asarray(list(DataBuilder.idx_to_filename.keys()))

        elif self.data_type == 3:
            self.dataset = Image_Pipeline_Dataset(data_path=datasets_to_be_updated_list,
                                                  return_label=None)

            DataBuilder = self.dataset.get_Dataloader()
            imgs_filepath = DataBuilder.get_all_data()
            vectors_out_np = self.model(imgs_filepath)
            vectors_out_np = [np.asarray(embd[0], dtype=np.float64) for embd in vectors_out_np]
            vectors_out_np = np.asarray(vectors_out_np)

            projector = GaussianRandomProjection(n_components=self.vector_dim, random_state=42)
            vectors_out_np = projector.fit_transform(vectors_out_np)
            vectors_dataset_idx_np = np.asarray(list(DataBuilder.idx_to_images.keys()))
        
        self.save_updated_vectors(vectors_out_np, vectors_dataset_idx_np, data_builder=DataBuilder)