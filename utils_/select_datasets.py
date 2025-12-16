# python select_data.py -v "vectors_f-MNIST_VDBs_100_1762274534.608018_100" -i "/opt/dlami/nvme/Results/Image_Data/f-MNIST/f-mnist_test_venom.csv" -out "/home/ubuntu/Projects/Vector Embedding/results/f-mnist/100V-VDB/sim_search" -s "vectorDB" -r 0.2 -dt "image" --vector-db -vs 100 

import torch
from models.vectorEmbedding import Num2Vec
from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset, Image_Pipeline_Dataset
import logging
import time
import pickle
import joblib

import numpy as np
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import matplotlib.colors as colors


from utils_.utils import calculate_cosine_similarity
import os
from server.server_utils.qdrant_controller import qdrant_controller

from transformers import pipeline


logger = logging.getLogger(__name__)


class Data_selection:

    def __init__(self, data_path, out_path, vectors_path, model_path, data_type, use_vector_DB, d_token=100):
        self.data_type = data_type
        self.use_vec_db = use_vector_DB

        self.qdrant_controller = None
        if self.use_vec_db:
            self.qdrant_controller = qdrant_controller()
        if model_path is not None:
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
                self.d_tokens = dim_k
            elif self.data_type == 2:
                self.model = joblib.load(model_path)

                self.dataset = graph_Pipeline_Dataset(data_path=data_path,
                                         return_label=False)
                self.d_tokens = d_token
        else:
            if self.data_type == 3:
                self.model = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", pool=True, device='cuda', use_fast=True, max_new_tokens=d_token)
                self.dataset = Image_Pipeline_Dataset(data_path=data_path, return_label=None, create_op=False)
                self.d_tokens = d_token
            else:
                self.model = None
                self.dataset = None
                AssertionError('A model ckpt need to be provided')
        
        self.output_path = out_path
        if self.use_vec_db:
            self.qdrant_collection_name = vectors_path
            self.vectors = self.qdrant_controller.get_vectors(collection_name=vectors_path)
        else:
            if isinstance(vectors_path, str):
                self.vectors = self.load_dict(vectors_path)
            elif isinstance(vectors_path, dict):
                self.vectors = vectors_path
    
        logging.basicConfig(filename=os.path.join(self.output_path, 'log_file.log'),
                            encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)

    def compute_KMeans_cluster(self, vect_dict, vectors, min_s, max_s=0.5):
        silioute_score = []
        cluster_list = []
        labels_list = []
        num_of_clusters = np.arange(start=2, stop=10, step=1, dtype=np.int_)
        datasets_vectors = list(vect_dict.values())
        for n_clusters in num_of_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(datasets_vectors)
            
            labels_list.append(cluster_labels)

            silhouette_avg = silhouette_score(datasets_vectors, cluster_labels)
            logger.info(f"For n_clusters = {n_clusters} The average silhouette_score is : {silhouette_avg} ")
            silioute_score.append(silhouette_avg)
            cluster_list.append(clusterer)

        highest_score_idx = np.argmax(silhouette_avg)
        best_cluster = cluster_list[highest_score_idx]
        best_labels = labels_list[highest_score_idx]
        centers = cluster_list[highest_score_idx].cluster_centers_

        out_dict = {}
        for dataset_name, vector in vectors.items():
            label = best_cluster.predict([vector])[0]
            selected_data_idx = [i for i, label_ in enumerate(best_labels) if label_ == label]
            center = centers[label]
            

            if len(selected_data_idx) < np.ceil(min_s*len(datasets_vectors)):
                print(f"Cluster has too few points ({len(selected_data_idx)}). Adding more...")

                other_indices = np.where(best_labels != label)[0]

                other_data = [datasets_vectors[int(i)] for i in other_indices]

                # Distance of other points to the target cluster center
                distances = np.linalg.norm(other_data - center, axis=1)
                
                # Sort and get the index of the closest points
                sorted_indices = np.argsort(distances)

                needed = np.ceil(min_s*len(datasets_vectors)) - len(selected_data_idx)

                selected_indices = [sorted_indices[i] for i in range(int(needed))]
                selected_data_idx = selected_data_idx + selected_indices
                
                # Remove duplicates (if their any chance)
                selected_data_idx = list(set(selected_data_idx))
                print(selected_indices)
                print(len(selected_indices))

            elif len(selected_data_idx) > np.ceil(max_s*len(datasets_vectors)):
                print(f"Cluster has too many points ({np.ceil(max_s*len(datasets_vectors))}). Removing...")

                cluster_data = [datasets_vectors[i] for i in selected_data_idx]
                
                distances = np.linalg.norm(cluster_data - center, axis=1)
                sorted_indices = np.argsort(distances)[::-1]
                to_remove = len(selected_data_idx) - np.ceil(max_s*len(datasets_vectors))

                remove_idx = [sorted_indices[i] for i in range(int(to_remove))]
                removed_indices = [selected_data_idx[i] for i in remove_idx]
                selected_data_idx = [i for i in selected_data_idx if i not in removed_indices]

                
            selected_datanames = [dataset_name for i, dataset_name in enumerate(vect_dict) if i in selected_data_idx]
            sel_vectors = [vec for filename_, vec in self.vectors.items() if filename_ in selected_datanames]
    
            out_dict[dataset_name] = [selected_datanames, sel_vectors]

        out_dict = {'selected_dataset': out_dict,
                    'Pred_Dataset': vectors}
        return out_dict

    def compute_similarity(self, vect_dict, vectors):
        out_dict = {}
        for dataset_name, vector_1 in vectors.items():
            tmp_dict ={}
            for dataset_name_2, vector_2 in vect_dict.items():
                sim_ = calculate_cosine_similarity(a=vector_1,
                                                   b=vector_2)
                tmp_dict[dataset_name_2] = sim_

            out_dict[dataset_name] = tmp_dict

        return out_dict

    def compute_distance(self, vect_dict, vectors):
        # Euclidean distance
        out_dict = {}
        for dataset_name, vector_1 in vectors.items():
            tmp_dict ={}
            for dataset_name_2, vector_2 in vect_dict.items():
                dist_ = np.linalg.norm(vector_1 - vector_2)
                tmp_dict[dataset_name_2] = dist_
            out_dict[dataset_name] = tmp_dict
        return out_dict
    
    def select_random(self, vect_dict, vectors, ratio=0.3):
        out_dict = {}
        for dataset_name, vector_1 in vectors.items():
            filenames_list = list(vect_dict.keys())
            selected_data = np.random.choice(filenames_list, np.int_(np.floor(len(filenames_list) * ratio)), replace=False)
            out_dict[dataset_name] = selected_data.tolist()

        return out_dict

    def select_most_relevants_data(self, dict_, vectors, ratio=0.3, compare='desc'):
        ret_dict = {}
        for dataset_name, data_dict in dict_.items():

            if compare.lower() == 'desc':
                data_dict_sorted = {k: v for k, v in sorted(data_dict.items(), key=lambda x: x[1], reverse=False)}
            else:
                data_dict_sorted = {k: v for k, v in sorted(data_dict.items(), key=lambda x: x[1], reverse=True)}

            datasets_filenames = list(data_dict_sorted.keys())
            top_k = int(len(datasets_filenames) * ratio)
            
            sel_dataset = datasets_filenames[:top_k]
            sel_vectors = [vec for filename_, vec in self.vectors.items() if filename_ in sel_dataset]
            
            ret_dict[dataset_name] = [sel_dataset, sel_vectors]
        ret_dict = {'selected_dataset': ret_dict,
                    'Pred_Dataset': vectors}
        # print(ret_dict)
        return ret_dict

    def save_dict(self, dict_, filename):
        file_name = os.path.join(self.output_path, f'rel_data_{filename}_{time.time()}.pickle')
        with open(file_name, 'wb') as handle:
            pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dict(self, path_):
        with open(path_, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def convert_vectors_to_out_list(self, vectors, vectors_dataset_idx, data_builder):
        out_data = {}
        for vector, dataset_idx in zip(vectors, vectors_dataset_idx):
            dataset_name = data_builder.get_dataset_name(dataset_idx)
            out_data[dataset_name] = vector
        return out_data

    def plot_representation(self, dict_vectors, new_vectors_, new_dataset_names, dataset_builder, most_relevant, ratio=0.3):
        colors_lst = list(colors.TABLEAU_COLORS.keys())
        plt.set_loglevel('info')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for new_vector_, data_idx in zip(new_vectors_, new_dataset_names):
            data_name = dataset_builder.get_dataset_name(data_idx)
            most_rel = most_relevant[data_name]
            vec_np = np.array(list(dict_vectors.values()))
            
            labels_ = ['Relevant Dataset' if i in most_rel else 'Data Lake' for i in dict_vectors.keys()]            
            
            vec_concat_np = np.concatenate((vec_np, [new_vector_]), axis=0)

            pca = PCA(n_components=3)
            result = pca.fit_transform(vec_concat_np)
            unique = list(set(labels_))
            for i, u in enumerate(unique):

                vec_ = [result[j, :] for j in range(len(labels_)) if labels_[j] == u]
                np_arr_vec = np.asarray(vec_, dtype=np.float64)
           
                ax.scatter(np_arr_vec[:, 0], np_arr_vec[:, 1], np_arr_vec[:, 2], c=colors_lst[i], label=u)
            ax.scatter(result[vec_np.shape[0]:, 0], result[vec_np.shape[0]:, 1], result[vec_np.shape[0]:, 2], c='purple', label='New Data')
                
        ax.legend()
        ax.grid(True)

        ax.set(xlabel=(r' $x$'), ylabel=(r'$y$'), zlabel=(r'$z$'))
        plt.savefig(os.path.join(self.output_path, 'plot_representation_2.png'), dpi=100)

    def vector_DB_Search(self, collection_name, vectors, select_percent):
        out_dict = {}
        res_dict = {}
        for dataset_name, vector in vectors.items():
            sim_search_vecs = self.qdrant_controller.similarity_search_(collection_name=collection_name, vectors=vector, select_percent=select_percent)

            print(f'Similarity search for {dataset_name} done')
            # sel_dataset = datasets_filenames[:top_k]
            sel_vectors = [vec for filename_, vec in sim_search_vecs.items()]
            sel_datasets = [filename_ for filename_, vec in sim_search_vecs.items()]
            res_dict[dataset_name] = [sel_datasets, sel_vectors]
        out_dict = {'selected_dataset': res_dict, 'Pred_Dataset': vectors}
        return out_dict

    def find_relevant_datasets(self, type_of_selection, data_ratio_selection, plot_representation=True, saved_sim_search=True):

        start_time = time.time()
        self.data_type
        if self.data_type == 1:
            vectors_output = []
            vectors_dataset_idx = []

            DataBuilder = self.dataset.get_Dataloader()
            dataloader = DataLoader(DataBuilder, batch_size=1, shuffle=False)
            self.model.eval()

            with torch.no_grad():
                for batch_idx, batch_ in enumerate(dataloader):
                    
                    features_, dataset_idx = batch_
                    features_ = features_.to(self.model.device)
                    
                    _, vectors, _ = self.model(features_)

                    vectors_output.append(vectors.detach().cpu().numpy())
                    vectors_dataset_idx.append(dataset_idx.numpy())
                    del vectors, features_
                    torch.cuda.empty_cache()

            vectors_out_tensors = np.concatenate(vectors_output, axis=0)
            vectors_dataset_idx_tensors = np.concatenate(vectors_dataset_idx, axis=0)
            
            datasets_dict = self.convert_vectors_to_out_list(data_builder=DataBuilder,
                                                            vectors=vectors_out_tensors,
                                                            vectors_dataset_idx=vectors_dataset_idx_tensors)
        elif self.data_type == 2:
            DataBuilder = self.dataset.get_Dataloader()
            data_graphs = DataBuilder.get_all_data()
            vectors_out_tensors = self.model.infer(graphs=data_graphs)
            vectors_dataset_idx_tensors = np.asarray(list(DataBuilder.idx_to_filename.keys()))

            datasets_dict = self.convert_vectors_to_out_list(data_builder=DataBuilder,
                                                            vectors=vectors_out_tensors,
                                                            vectors_dataset_idx=vectors_dataset_idx_tensors)
            
            self.d_tokens = vectors_out_tensors[0].shape[0]
            print(self.d_tokens)
            
        elif self.data_type == 3:
            DataBuilder = self.dataset.get_Dataloader()
            imgs_filepath = DataBuilder.get_all_data()
            vectors_out_tensors = self.model(imgs_filepath)
            vectors_dataset_idx_tensors = np.asarray(list(DataBuilder.idx_to_images.keys()))

            vectors_out_np = [np.asarray(embd[0], dtype=np.float64) for embd in vectors_out_tensors]
            vectors_out_np = np.asarray(vectors_out_np)
            print(vectors_out_np.shape)
            
            projector = GaussianRandomProjection(n_components=self.d_tokens, random_state=42)
            vectors_out_tensors = projector.fit_transform(vectors_out_np)

            datasets_dict = self.convert_vectors_to_out_list(data_builder=DataBuilder,
                                                            vectors=vectors_out_tensors,
                                                            vectors_dataset_idx=vectors_dataset_idx_tensors)

        if type_of_selection.lower() == 'cosine':
            compute_data = self.compute_similarity(vect_dict=self.vectors,
                                                   vectors=datasets_dict)
            most_relevant_data = self.select_most_relevants_data(dict_=compute_data,
                                                                 ratio=data_ratio_selection,
                                                                 vectors=datasets_dict,
                                                                 compare='asc')    
                   
        elif type_of_selection.lower() == 'distance':
            compute_data = self.compute_distance(vect_dict=self.vectors,
                                                 vectors=datasets_dict)

            most_relevant_data = self.select_most_relevants_data(dict_=compute_data,
                                                                 ratio=data_ratio_selection,
                                                                 vectors=datasets_dict,
                                                                 compare='desc')
        elif type_of_selection.lower() == 'random':
            most_relevant_data = self.select_random(vect_dict=self.vectors, vectors=datasets_dict)
        elif type_of_selection.lower() == 'k-means':
            max_s = 0.5
            if data_ratio_selection > max_s:
                max_s = data_ratio_selection + 0.1
            most_relevant_data = self.compute_KMeans_cluster(vect_dict=self.vectors, vectors=datasets_dict, min_s=data_ratio_selection, max_s=max_s)
        elif type_of_selection.lower() == 'vectordb':
            if self.use_vec_db:
                
                most_relevant_data = self.vector_DB_Search(collection_name=self.qdrant_collection_name,
                                                                               vectors=datasets_dict,
                                                                               select_percent=data_ratio_selection)
                # most_relevant_data = self.qdrant_controller.similarity_search_(collection_name=self.qdrant_collection_name,
                #                                                                vectors=datasets_dict,
                #                                                                select_percent=data_ratio_selection)
            else:
                raise AssertionError('You must select the --vector-DB during the input')
        else:
            raise AssertionError('Wrong type of dataset selection!!')
        logger.info(f'Execution time for {type_of_selection} : {time.time() - start_time}')
        if plot_representation:
            self.plot_representation(dict_vectors=self.vectors,
                                    most_relevant=most_relevant_data['selected_dataset'],
                                    new_dataset_names=vectors_dataset_idx_tensors,
                                    dataset_builder=DataBuilder, 
                                    new_vectors_=vectors_out_tensors,)
        if saved_sim_search:
            self.save_dict(dict_=most_relevant_data, filename=type_of_selection.lower())
        else:
            return most_relevant_data
