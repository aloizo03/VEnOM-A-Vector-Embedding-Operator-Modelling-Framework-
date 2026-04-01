import torch
from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset, Image_Pipeline_Dataset
from models.vectorEmbedding import Num2Vec
from models.torch_utils import select_device
import joblib
from utils_.utils import get_file_modified_time

from PIL import Image  

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd 

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

import torch
from transformers import AutoImageProcessor, AutoModel

import matplotlib.pyplot as plt
import logging
import os
import time
import pickle
from server.server_utils.qdrant_controller import qdrant_controller

from transformers import pipeline

logger = logging.getLogger(__name__)


class Vectorise:

    def __init__(self, model_path, data_path, out_path, data_type, save_to, collection_name, d_token=100):
        self.data_type = data_type

        self.save_to = save_to

        self.qdrant_controller = None
        if self.save_to.lower() == 'vectordb':
            print('Wait qdrant controller')
            self.qdrant_controller = qdrant_controller()
            print('Qdrant Controller init!!')
        if self.data_type == 1:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt = torch.load(model_path, map_location=device)

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
            
            if torch.cuda.device_count() > 1 and self.data_type == 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
            
            self.vector_dim = d_tokens
        elif self.data_type == 2:
            self.model = joblib.load(model_path)
            self.model.epochs = 1
            self.dataset = graph_Pipeline_Dataset(data_path=data_path,
                                                  return_label=False)
            
            self.vector_dim = d_token
        elif self.data_type == 3:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
            self.dataset = Image_Pipeline_Dataset(data_path=data_path,
                                                  return_label=None)
            
            self.vector_dim = d_token

        
        self.output_path = out_path
        self.collection_name = collection_name
        self.device = select_device()

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

    def save_datetime_datasets(self, data_builder, dataset_idxs):
        out_dict = {}
        for dataset_idx in dataset_idxs:
            dataset_name = data_builder.get_dataset_name(dataset_idx)
            out_dict[dataset_name] = get_file_modified_time(dataset_name)

        return out_dict

    def compute_data_sketches(self, data_builder):
        sketches_dict = {}
        
        if self.data_type in [1, 2] and hasattr(data_builder, 'idx_to_filename'):
            mapping = data_builder.idx_to_filename
        elif self.data_type == 3 and hasattr(data_builder, 'idx_to_images'):
            mapping = data_builder.idx_to_images
        else:
            return sketches_dict


        if self.data_type == 2:
            all_graphs = data_builder.get_all_data()

        for idx, filepath in mapping.items():
            try:
                if self.data_type == 1:  
                    df = pd.read_csv(filepath)
                    num_data = df.select_dtypes(include=[np.number])
                    # Sketch: Mean, Std, Min, Max of columns + Row count
                    col_stats = num_data.agg(['mean', 'std', 'min', 'max']).fillna(0).values.flatten()
                    sketch = np.concatenate([[len(df)], col_stats])
                    
                elif self.data_type == 2:  # Graph
                    graph = all_graphs[idx]
                    # Sketch: Node count and Edge count
                    sketch = np.array([graph.number_of_nodes(), graph.number_of_edges()])
                    
                elif self.data_type == 3:  # Image
                    # Sketch: 8x8 Grayscale perceptual hash equivalent
                    img = Image.open(filepath).convert("L").resize((8, 8))
                    sketch = np.array(img).flatten() / 255.0
                
                sketches_dict[filepath] = sketch
                
            except Exception as e:
                print(f"Sketch generation failed for {filepath}: {e}")
                sketches_dict[filepath] = None 

        return sketches_dict

    def save_vectors(self, dict_, time_dict, skecthes_dict, file_name=None):
        if self.save_to.lower() == 'local':
            if file_name is None:
                file_name = os.path.join(self.output_path, f'vectors_{self.collection_name}_{time.time()}_{self.vector_dim}.pickle')
                file_name_time_dict = os.path.join(self.output_path, f'vectors_time_{self.collection_name}_{time.time()}_{self.vector_dim}.pickle')
                file_name_time_dict = os.path.join(self.output_path, f'skecthes_{self.collection_name}_{time.time()}_{self.vector_dim}.pickle')
                
            with open(file_name, 'wb') as handle:
                pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(file_name_time_dict, 'wb') as handle:
                pickle.dump(time_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'Save the vectors localy at : {file_name}')
        elif self.save_to.lower() == 'vectordb':
            if self.data_type == 1:
                if torch.cuda.device_count() > 1:
                    size_ = self.model.module.d_out
                else:
                    size_ = self.model.d_out
            elif self.data_type == 2:
                size_ = self.vector_dim 
            elif self.data_type == 3:
                size_ = self.vector_dim 

            collection_name = f'vectors_{self.collection_name}_{time.time()}_{size_}'
            self.qdrant_controller.create_collection(name=collection_name, 
                                                     size=size_)
            self.qdrant_controller.insert_vectors_in_batches(collection_name=collection_name, 
                                                             data_vectors=dict_,
                                                             times_dict=time_dict,
                                                             skecthes_dict=skecthes_dict)
            
        else:
            AssertionError('Wring input for the vector save we only support: local, and vectordb')

    def compute_vectors(self, batch_size, num_of_workes=0, show_progress=True, ret_vec=False):
        #TODO: Check Images in Local storage, as well as with Vector DB
        print('Start Data Loading')
        DataBuilder = self.dataset.get_Dataloader()
        print('Data Loaded')

        skecthes_dict = self.compute_data_sketches(data_builder=DataBuilder)

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
                        features_ = features_.type(torch.float64)
                        _, vectors, _ = self.model(features_)
                        
                        vectors_output.append(vectors.detach().cpu().numpy())
                        vectors_dataset_idx.append(dataset_idx.numpy())

                        pbar.update(batch_size)
                        if batch_idx % 10 == 0 and show_progress:
                            pbar.set_description(f'Data Batch id {batch_idx}, total data vectorised {total_data}/{DataBuilder.data_len}')
                            logger.info(str(pbar))
                vectors_out_np = np.concatenate(vectors_output, axis=0)
                vectors_dataset_idx_np = np.concatenate(vectors_dataset_idx, axis=0)
        elif self.data_type ==2:
            data_graphs = DataBuilder.get_all_data()
            vectors_out_np = self.model.infer(graphs=data_graphs)
            vectors_dataset_idx_np = np.asarray(list(DataBuilder.idx_to_filename.keys()))
            self.vector_dim = vectors_out_np[0].shape[0]

        elif self.data_type == 3:
            imgs_data = DataBuilder.get_all_data(read_dataset=False)

            imgs_data = list(imgs_data)

            all_vectors = []

            self.model.eval()

            with torch.no_grad():
                for i in tqdm(range(0, len(imgs_data), batch_size), desc="Extracting image vectors"):
                    batch_paths = imgs_data[i : i + batch_size]
                    
                    batch_imgs = []
                    for path in batch_paths:
                        try:
                            img = Image.open(path).convert("RGB")
                            batch_imgs.append(img)
                        except Exception as e:
                            print(f"Error loading image {path}: {e}")

                    inputs = self.processor(images=batch_imgs, return_tensors="np")
                    pixel_array_list = inputs["pixel_values"].tolist()
                    pixel_values = torch.tensor(pixel_array_list, dtype=torch.float32).to(self.device)
                    
                    outputs = self.model(pixel_values=pixel_values)
                    
                    # Extract vectors
                    batch_vectors = outputs.pooler_output.cpu().numpy()
                    all_vectors.append(batch_vectors)

            # 6. Stack them back together
            vectors_out_np = np.vstack(all_vectors)
            vectors_out_np = vectors_out_np.astype(np.float64)

            # Your existing projection
            projector = GaussianRandomProjection(n_components=self.vector_dim, random_state=42)
            vectors_out_np = projector.fit_transform(vectors_out_np)

            vectors_dataset_idx_np = np.asarray(list(DataBuilder.idx_to_images.keys()))
        
        out_vectors_dict = self.convert_vectors_to_out_list(vectors=vectors_out_np,
                                         vectors_dataset_idx=vectors_dataset_idx_np, 
                                         data_builder=DataBuilder)
        
        out_modified_time_dict = self.save_datetime_datasets(data_builder=DataBuilder,
                                                             dataset_idxs=vectors_dataset_idx_np)
                
        if not ret_vec:
            self.plot_vectors(vectors_out_np)
            self.save_vectors(out_vectors_dict,
                              out_modified_time_dict,
                              skecthes_dict)
            
        else:
            return out_vectors_dict



