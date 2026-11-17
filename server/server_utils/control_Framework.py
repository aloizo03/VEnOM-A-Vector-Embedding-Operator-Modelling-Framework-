import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time
import pickle
import numpy as np
import pandas as pd
from server_utils.qdrant_controller import qdrant_controller
from server_utils.utils import resolve_path, TABFM_TASK
from utils_.vectors import Vectorise
from utils_.select_datasets import Data_selection
from utils_.exp_accuracy import EXP_Accuracy
from utils_.create_operator import Create_Operator
from utils_.utils import get_file_modified_time

scores_out_file = os.environ.get('VENOM_SCORES_DIR', os.path.join(_REPO_ROOT, 'server', 'tmp_score_labels', ''))
operators_out_file = os.environ.get('VENOM_OPERATORS_DIR', os.path.join(_REPO_ROOT, 'server', 'tmp_operators', ''))


def _tab_model(embedder):
    return embedder if embedder in ('num2vec', 'tabfm', 'dataset2vec') else 'num2vec'


def _modified_time(name):
    try:
        return get_file_modified_time(name)
    except OSError:
        return time.time()


def _model_path(embedder, model_path):
    if embedder in ('num2vec', 'graph2vec'):
        return resolve_path(model_path)
    return None


class control_Framework:

    def __init__(self):
        self.qdrant_controller = None

        if scores_out_file is not None:
            if not os.path.exists(scores_out_file):
                os.makedirs(scores_out_file)
        if operators_out_file is not None:
            if not os.path.exists(operators_out_file):
                os.makedirs(operators_out_file)

    def _qdrant(self):
        if self.qdrant_controller is None:
            self.qdrant_controller = qdrant_controller()
        return self.qdrant_controller

    def create_Vectorisation(self, data_path, out_path, data_type, embedder, model_path=None, vec_dim=None, show_progress=False):
        os.makedirs(out_path, exist_ok=True)
        vectorise = Vectorise(model_path=_model_path(embedder, model_path),
                              data_path=resolve_path(data_path),
                              out_path=out_path,
                              data_type=data_type,
                              save_to='local',
                              collection_name='ui',
                              d_token=vec_dim if vec_dim else 100,
                              tab_model=_tab_model(embedder),
                              tabfm_task=TABFM_TASK)
        vectors_dict = vectorise.compute_vectors(batch_size=4, ret_vec=True, show_progress=show_progress)
        return {name: np.asarray(vector, dtype=np.float64) for name, vector in vectors_dict.items()}

    def store_vectors(self, save_to, folder, vectors_dict, dataset_name):
        vec_dim = int(len(next(iter(vectors_dict.values()))))
        if save_to.lower() == "local store":
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f'vectors_{dataset_name}_{time.time()}_{vec_dim}.pickle')
            with open(filename, 'wb') as handle:
                pickle.dump(vectors_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return filename, vec_dim
        elif save_to.lower() == "qdrant":
            collection_name = f'vectors_{dataset_name}_{time.time()}_{vec_dim}'
            data_vectors = {name: vector.tolist() for name, vector in vectors_dict.items()}
            times_dict = {name: _modified_time(name) for name in data_vectors}
            self._qdrant().create_collection(name=collection_name, size=vec_dim)
            self._qdrant().insert_vectors_in_batches(collection_name=collection_name,
                                                     data_vectors=data_vectors,
                                                     times_dict=times_dict)
            return collection_name, vec_dim
        raise ValueError(f'Unsupported vector store: {save_to}')

    def store_similarity_search_locally(self, out_path, vectors_dict, name):
        os.makedirs(out_path, exist_ok=True)
        file_name = os.path.join(out_path, f'rel_data_{name}_{time.time()}.pickle')
        with open(file_name, 'wb') as handle:
            pickle.dump(vectors_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return file_name

    def similarity_search(self, data_path, out_path, vectors, vector_info, data_selection, data_ratio):
        if data_ratio > 1:
            data_ratio = data_ratio / 100
        print(f'Ratio of data selected: {data_ratio}')
        os.makedirs(out_path, exist_ok=True)
        embedder = vector_info['embedder']
        use_vector_db = vector_info['store'].lower() == 'qdrant'
        if data_selection == 'vectordb' and not use_vector_db:
            raise ValueError('Vector DB Search needs vectors stored in Qdrant')
        selection = Data_selection(data_path=resolve_path(data_path),
                                   out_path=out_path,
                                   vectors_path=vectors,
                                   model_path=_model_path(embedder, vector_info['model_path']),
                                   data_type=vector_info['data_type'],
                                   use_vector_DB=use_vector_db,
                                   d_token=vector_info['vec_dim'] or 100,
                                   tab_model=_tab_model(embedder),
                                   tabfm_task=TABFM_TASK)
        return selection.find_relevant_datasets(type_of_selection=data_selection,
                                                data_ratio_selection=data_ratio,
                                                plot_representation=False,
                                                saved_sim_search=False)

    def get_vector_size(self, database_, collection_name):
        if database_.lower() == "qdrant":
            return self._qdrant().get_collection_vec_dim(collection_name=collection_name)
        elif database_.lower() == "local store":
            with open(collection_name, 'rb') as handle:
                b = pickle.load(handle)
            return len(list(b.values())[0])

    def get_vectors_from_vectorDB(self, database_, collection_name):
        if database_.lower() == "qdrant":
            return self._qdrant().get_vectors(collection_name=collection_name)
        elif database_.lower() == "local store":
            with open(collection_name, 'rb') as handle:
                return pickle.load(handle)


class Operator_Modelling:

    def __init__(self, operator_name, operator_outpath, operator, sim_search_vectors, prediction_data_path, directory_datapath, data_type, target_labels=None):
        self.operator_name = operator_name
        self.out_path = resolve_path(operator_outpath)

        self.selected_operator = operator
        self.sim_search_vectors_path = sim_search_vectors
        self.prediction_dataset_path = resolve_path(prediction_data_path)
        self.directory_datapath = resolve_path(directory_datapath)
        self.data_type = data_type
        self.target_labels = resolve_path(target_labels) if target_labels else None
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
                               operator_name=self.selected_operator,
                               data_type=self.data_type,
                               out_filename='scores')
        exp_acc.create_op()

        if self.target_labels is None and self.data_type in (2, 3):
            target_acc = EXP_Accuracy(input_data_dir=self.prediction_dataset_path,
                                      out_path_dir=new_path,
                                      operator_name=self.selected_operator,
                                      data_type=self.data_type,
                                      out_filename='target_scores')
            target_acc.create_op()
            self.target_labels = os.path.join(new_path, 'target_scores.csv')

        return new_path

    def model_operator(self, path_labels):
        new_folder_name = f'operator_{time.time()}'
        new_path = os.path.join(self.out_path, new_folder_name)

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        labels_path_full = os.path.join(path_labels, 'scores.csv')
        labels_dict = pd.read_csv(labels_path_full)
        c_op = Create_Operator(operator_dir=None,
                               out_path=new_path,
                               out_file='results.csv')

        print('Start Operator Modelling')
        results_ = c_op.create_operator(operator_name=self.selected_operator,
                                        most_relevant_data_path=self.sim_search_vectors_dict,
                                        target_labels=self.target_labels,
                                        repetitions=5,
                                        use_vectors=True,
                                        load_rel_data=False,
                                        return_res=True,
                                        labels_dict=labels_dict,
                                        data_type=self.data_type)

        file_name = f'metrics.csv'
        out_file = os.path.join(new_path, file_name)

        return results_, out_file
