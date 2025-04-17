import sys

from flask.cli import F
sys.path.append('../')
from models.vectorEmbedding import Num2Vec
from utils_.vectors import Vectorise
import os 
import time
from server_utils.qdrant_controller import qdrant_controller
import pickle
from utils_.select_datasets import Data_selection
from utils_.exp_accuracy import EXP_Accuracy
from utils_.create_operator import Create_Operator
import pandas as pd

scores_out_file = '/home/ubuntu/Projects/VectorEmbedding-Representation-/server/tmp_score_labels/'
operators_out_file = '/home/ubuntu/Projects/VectorEmbedding-Representation-/server/tmp_operators/'

class control_Framework:

    def __init__(self):
        self.dataset_name = None
        self.vec_size = None
        self.save_Vectors = None
        self.Vectorise = None

    def set_selected_datasets(self, selected_Dataset):
        self.dataset_name = selected_Dataset

    def set_selected_Vec_Size(self, selected_Vector_Size):
        self.vec_size = selected_Vector_Size

    def set_selected_save_vecs(self, save_Vectors):
        self.save_Vectors = save_Vectors

    def create_Vectorisation(self, model_path, data_path, out_path, data_type, show_progress=False):

        self.Vectorise = Vectorise(model_path, 
                                   data_path, 
                                   out_path, 
                                   data_type)
        vectors_dict = self.Vectorise.compute_vectors(batch_size=4, ret_vec=True, show_progress=show_progress)

        return vectors_dict
    
    def store_vectors(self, save_to, folder, vectors_dict, dataset_name):
        if save_to.lower() == "local store":
            filename = os.path.join(folder, f'vectors_{dataset_name}_{time.time()}_{self.vec_size}.pickle')
            self.Vectorise.save_vectors(file_name=filename, dict_=vectors_dict)

            return filename
        elif save_to.lower() == "qdrant":
            self.qdrant_controller = qdrant_controller()
            collection_name = f'vectors_{dataset_name}_{time.time()}_{self.vec_size}'
            self.qdrant_controller.create_collection(name=collection_name, size=self.vec_size)
            self.qdrant_controller.insert_vectors(collection_name=collection_name,
                                                  data_vectors=vectors_dict)
            
            return collection_name
        elif save_to.lower() == 'chromadb':
            pass
    
    def similarity_search(self, database_, collection_name, vectors, select_percent):
        out_dict = {}
        res_dict = {}
        for dataset_name, vector in vectors.items():
            
            if database_.lower() == "qdrant":
                try:
                    self.qdrant_controller
                except:
                    self.qdrant_controller = qdrant_controller()
                sim_search_vecs = self.qdrant_controller.similarity_search_(collection_name=collection_name,
                                                                            vectors=vector,
                                                                            select_percent=select_percent)
                
                print(f'Similarity search for {dataset_name} done')
                # sel_dataset = datasets_filenames[:top_k]
                sel_vectors = [vec for filename_, vec in sim_search_vecs.items()]
                sel_datasets = [filename_ for filename_, vec in sim_search_vecs.items()]
                res_dict[dataset_name] = [sel_datasets, sel_vectors]
            
               
            elif database_.lower() == "chromadb":
                pass
            
        out_dict = {'selected_dataset': res_dict, 'Pred_Dataset': vectors}
        return out_dict
    
    def store_similarity_search_locally(self, out_path, vectors_dict, name):
        file_name = os.path.join(out_path, f'rel_data_{name}_{time.time()}.pickle')
        with open(file_name, 'wb') as handle:
            pickle.dump(vectors_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return file_name
    
    def simimilarity_search_localy(self, data_path, out_path, vectors_path, model_path, data_selection, data_radio, data_type):
        if data_radio > 1:
            data_radio = data_radio / 100
        print(f'Ratio of data selected: {data_radio}')
        clustering = Data_selection(data_path=data_path,
                                    out_path=out_path, 
                                    vectors_path=vectors_path, 
                                    model_path=model_path, 
                                    data_type=data_type)
        return clustering.find_relevant_datasets(type_of_selection=data_selection, data_ratio_selection=data_radio, plot_representation=False, saved_sim_search=False)

    def get_vector_size(self, database_, collection_name):
        if database_.lower() == "qdrant":
            try:
                self.qdrant_controller
            except:
                self.qdrant_controller = qdrant_controller()
            return self.qdrant_controller.get_collection_vec_dim(collection_name=collection_name)
        elif database_.lower() == "chromadb":
            pass
        elif database_.lower() == "local store":
            with open(collection_name, 'rb') as handle:
                b = pickle.load(handle)
            return list(b.values())[0].shape[0]   
        
    def get_vectors_from_vectorDB(self, database_, collection_name):
        if database_.lower() == "qdrant":
            try:
                self.qdrant_controller
            except:
                self.qdrant_controller = qdrant_controller()
            vectors_ = self.qdrant_controller.get_vectors(collection_name=collection_name)
            return vectors_
        elif database_.lower() == "chromadb":
            pass


class Operator_Modelling:

    def __init__(self, operator_name, operator_outpath, operator, sim_search_vectors, prediction_data_path, directory_datapath, data_type):
        self.operator_name = operator_name
        self.out_path = operator_outpath

        self.selected_operator = operator
        self.sim_search_vectors_path = sim_search_vectors
        self.prediction_dataset_path = prediction_data_path
        self.directory_datapath = directory_datapath
        self.data_type = data_type
        self.load_vectors()
    
    def load_vectors(self):
        with open(self.sim_search_vectors_path, 'rb') as handle:
            b = pickle.load(handle)
        
        self.sim_search_vectors_dict = b
        
    def create_labels(self):
        new_folder_name = f'op_modelling_{time.time()}'
        new_path = os.path.join(scores_out_file, new_folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        exp_acc = EXP_Accuracy(input_data_dir=self.directory_datapath,
                               out_path_dir=new_path,
                               operator_name=self.selected_operator)
            
        exp_acc.create_op()
        
        return new_path
            
    def model_operator(self, path_labels):
        new_folder_name = f'operator_{time.time()}'
        new_path = os.path.join(self.out_path, new_folder_name)
        
        if not os.path.exists(new_path):
                os.makedirs(new_path)
        if self.data_type == 2:
            is_graph = True
        else:
            is_graph = False
        labels_path_full = os.path.join(path_labels, 'scores.csv')
        labels_dict = pd.read_csv(labels_path_full)
        c_op = Create_Operator(operator_dir=None,
                               out_path=new_path)
        
        print('Start Operator Modelling')
        results_ = c_op.create_operator(operator_name=self.selected_operator,
                                        most_relevant_data_path=self.sim_search_vectors_dict,
                                        threshold=0.7,
                                        repetitions=5, 
                                        use_vectors=True, 
                                        load_rel_data=False,
                                        return_res=True,
                                        labels_dict=labels_dict,
                                        is_graph=is_graph)
        
        file_name = f'metrics.csv'
        out_file = os.path.join(new_path, file_name)

        return results_, out_file
            
            
            
            