from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import matplotlib.colors as colors
from PIL import Image
import pandas as pd
import os 
import torch
import sys
import joblib

sys.path.append('../')

from models.vectorEmbedding import Num2Vec
from data.Dataset import Pipeline_Dataset, graph_Pipeline_Dataset
from torch.utils.data import DataLoader


def check_dataset(filepath):
    if os.path.isdir(filepath):
        print('dataset_exits')
        return True
    else:
        print('dataset does not exits')
        return False
    
def check_dataset_file(filepath):
    if os.path.exists(filepath):
        print('dataset_exits')
        return True
    else:
        print('dataset does not exits')
        return False

def create_model(model_path):
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
    
    model = Num2Vec(d_numerical=input_dim,
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
    
    return model

def get_dataset_vector(model_path, data_path, data_type):
    if data_type == 1:
        ckpt = torch.load(model_path)
        model_info = ckpt['model_info']
        
        model = create_model(model_path=model_path)
        dataset_dataloader = Pipeline_Dataset(data_path=data_path,
                                            norm=True,
                                            ret_table=model_info['ret_table'],
                                            k_dim=model_info['dim_k'])
        
        DataBuilder = dataset_dataloader.get_Dataloader()
        dataloader = DataLoader(DataBuilder, batch_size=1, shuffle=False)
        vectors_output = []
        vectors_dataset_idx = []
        with torch.no_grad():
            for batch_idx, batch_ in enumerate(dataloader):
                
                features_, dataset_idx = batch_
                features_ = features_.to(model.device)
                
                _, vectors, _ = model(features_)
                
                vectors_output.append(vectors.detach().cpu().numpy())
                vectors_dataset_idx.append(dataset_idx.numpy())
                
        vectors_out_tensors = np.concatenate(vectors_output, axis=0)
        vectors_dataset_idx_tensors = np.concatenate(vectors_dataset_idx, axis=0)
    elif data_type == 2:

        model = joblib.load(model_path)
        model.epochs = 1
        dataset = graph_Pipeline_Dataset(data_path=data_path,
                                         return_label=False)
        DataBuilder = dataset.get_Dataloader()
        data_graphs = DataBuilder.get_all_data()
        vectors_out_tensors = model.infer(graphs=data_graphs)
        vectors_dataset_idx_tensors = np.asarray(list(DataBuilder.idx_to_filename.keys()))


    return convert_vectors_to_out_list(data_builder=DataBuilder, vectors=vectors_out_tensors,vectors_dataset_idx=vectors_dataset_idx_tensors)

def convert_vectors_to_out_list(vectors, vectors_dataset_idx, data_builder):
    out_data = {}
    for vector, dataset_idx in zip(vectors, vectors_dataset_idx):
        dataset_name = data_builder.get_dataset_name(dataset_idx)
        out_data[dataset_name] = vector
    return out_data
    
    
def create_vec_repr_plot(vectors_):
    img = BytesIO()
    vec = list(vectors_.values())
    vec_np = np.asarray(vec, dtype=np.float64)
    plt.set_loglevel('info')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pca = PCA(n_components=3)
    result = pca.fit_transform(vec_np)

    ax.scatter(result[:, 0], result[:, 1], result[:, 2], c='blueviolet')
    ax.set(xlabel=(r' $x$'), ylabel=(r'$y$'), zlabel=(r'$z$'))
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url

def create_sim_search_vector_plot(dict_vectors, sim_search_datasets):
    img = BytesIO()
    colors_lst = list(colors.TABLEAU_COLORS.keys())
    plt.set_loglevel('info')
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    # print(list(dict_vectors.keys()))
    # print(list(sim_search_datasets['selected_dataset'].keys()))
    # print(list(sim_search_datasets['Pred_Dataset'].keys()))

    most_relevant = sim_search_datasets['selected_dataset']
    query_datasets = sim_search_datasets['Pred_Dataset']
    for data_name, query_vector in query_datasets.items():
        most_rel_names = most_relevant[data_name][0]
        vec_np = np.array(list(dict_vectors.values()))

        labels_ = ['Relevant Dataset' if i in most_rel_names else 'Data Lake' for i in dict_vectors.keys()]

        vec_concat_np = np.concatenate((vec_np, [query_vector]), axis=0)
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
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url

    

def read_vectors(dir_path):
    with open(dir_path, 'rb') as handle:
            b = pickle.load(handle)

    return b

def create_res_plot(res_loss, labels):
    img = BytesIO()
    colours_ = []
    for i in range(0, len(labels)+1):
        colours_.append(tuple(np.random.choice(range(0, 2), size=3)))
    plt.bar(labels, res_loss, color=colours_, width = 0.4)
    plt.ylabel("Value")
    plt.xlabel("Metric")

    plt.savefig(img, format='png')
    plt.close()

    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url

def execute_query(conn, query, values):
    if values is None:
        conn.execute(query)
    else:
        return conn.execute(query, values)

def insert_dataset(conn, dataset_name, data_path, data_type):
    query = "SELECT * FROM datasets d where d.dataset_name = ?"
    try:
        data_res = execute_query(conn=conn, query=query, values=(dataset_name,)).fetchall()
    except Exception as err:
            print('Query Failed: %s\nError: %s' % (query, str(err)))
    
    if len(data_res) > 0:
        results = [tuple(row) for row in data_res]
        
        return results[0]
    else:
        try:
            query = "INSERT INTO DATASETS(dataset_name, datapath, data_type) VALUES(?,?,?)"
            execute_query(conn=conn, query=query, values=(dataset_name, data_path, data_type))
            conn.commit()
            return []
        except Exception as err:
            print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_all_datasets(conn):
    query_ = "SELECT * FROM datasets d"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query_, values=None)
        data_res = cursor.fetchall()
        results = [tuple(row) for row in data_res]
        return results
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query_, str(err)))
        return []

def get_all_vec_dimension(conn):
    query = "SELECT * FROM VECTOR_DIMENSION"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=None)
        data_res = cursor.fetchall()
        results = [list(tuple(row))[0] for row in data_res]
        results.sort()
        return results
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))
        return []

def get_dataset(conn, dataset_name):
    query = "SELECT * FROM datasets d where d.dataset_name = ?"
    try:
        data_res = execute_query(conn=conn, query=query, values=(dataset_name,)).fetchall()
        results = [tuple(row) for row in data_res]
        
        return results[0] 
    except Exception as err:
            print('Query Failed: %s\nError: %s' % (query, str(err)))
            return None
    
def get_avail_store_options(conn):
    query = "SELECT * FROM STORE_VECTORS"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=None)
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        return results
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))
        return []
    
def insert_vector_location(conn, filename, store_into, data_type, dataset_id):
    query_1 = "SELECT * FROM STORE_VECTORS WHERE store_name=?"
    try:
        data_res = execute_query(conn=conn, query=query_1, values=(store_into,)).fetchall()
    except Exception as err:
            print('Query Failed: %s\nError: %s' % (query_1, str(err)))
    results = [tuple(row) for row in data_res]
    store_id, _ = results[0]

    query_2 = "INSERT INTO DATASET_VECTORS(vector_name, store_id, data_type, dataset_id) VALUES(?, ?, ?, ?)"
    try:
        execute_query(conn=conn, query=query_2, values=(filename, store_id, data_type, dataset_id, ))
        conn.commit()
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query_2, str(err)))

def get_model_path(conn, dataset_name, vec_size):
    query = "SELECT * FROM datasets d where d.dataset_name = ?"
    try:
        data_res = execute_query(conn=conn, query=query, values=(dataset_name,)).fetchall()
        id_, dataset_name, data_path, data_type = data_res[0]

        query = "SELECT mt.id, mt.model_name, mt.vec_dim, mt.model_path, mt.data_type FROM MODELS_TABLE mt INNER JOIN VECTOR_DIMENSION vd ON mt.vec_dim = vd.vec_dim WHERE mt.data_type = ? AND mt.vec_dim = ?"
        data_res = execute_query(conn=conn, query=query, values=(data_type, vec_size)).fetchall()
        results = [tuple(row) for row in data_res]
        idx, model_name, vec_dim, model_path, _ = results[0]
        return  (model_name, vec_dim, model_path, data_type)
    except Exception as err:
            print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_model_path_from_size(conn, vec_dim, data_type):
    query = "SELECT model_path FROM Models_Table WHERE vec_dim = ? and data_type=?"
    try:
        data_res = execute_query(conn=conn, query=query, values=(vec_dim, data_type, )).fetchall()
        results = [tuple(row) for row in data_res]
        print(results)
        
        return results[0][0]
    except Exception as err:
            print('Query Failed: %s\nError: %s' % (query, str(err)))

def insert_task(conn, thread_id, description):
    query = "INSERT INTO TASKS(task_number, is_complete, description) VALUES(?,?,?)"
    try:
        execute_query(conn=conn, query=query, values=(thread_id, 0, description))
        conn.commit()
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def update_task_complete(conn, thread_id, exec_time):
    query = "UPDATE TASKS SET is_complete = ?, execution_time = ? WHERE task_number = ?"
    try:
        execute_query(conn=conn, query=query, values=(1, exec_time, thread_id,))
        conn.commit()
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_all_Tasks(conn):
    query = "SELECT * FROM TASKS"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=None)
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        return results
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_all_vector_datasets(conn):
    query = "SELECT DT.id, DT.vector_name, DT.store_id, SV.store_name, DT.dataset_id FROM DATASET_VECTORS DT JOIN STORE_VECTORS SV ON DT.store_id=SV.store_id"

    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=None)
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        if len(results) <= 0:
            return None
        else:
            return results    
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))
        
def find_database_store(conn, vector_name):
    query = "SELECT SV.store_name, DT.data_type FROM DATASET_VECTORS DT JOIN STORE_VECTORS SV ON DT.store_id=SV.store_id WHERE DT.vector_name =?"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=(vector_name, ))
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        if len(results) <= 0:
            return None
        else:
            store_name, data_type = results[0]
            return store_name, data_type
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))
        
    return None, None

def get_vectors_idx_from_name(conn, vector_name):
    query = 'SELECT id, DATA_TYPE FROM DATASET_VECTORS WHERE vector_name = ?'
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=(vector_name, ))
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        if len(results) <= 0:
            return None
        else:
            idx, data_type = results[0]
            return idx, data_type
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def insert_sim_search_vectors(conn, vector_name, dataset_path, saved_datapath, vectors_idx):
    query = "INSERT INTO Sim_Search(name, pred_datapath, stored_sim_search, vector_id) VALUES(?,?,?, ?)"
    try:
        execute_query(conn=conn, query=query, values=(vector_name, dataset_path, saved_datapath, vectors_idx))
        conn.commit()
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_similarity_search(conn):
    query = "SELECT * FROM SIM_SEARCH"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=None)
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        return results
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_dataset_id_from_sim_search(conn, sim_search):
    query = "SELECT DV.Dataset_id, DV.Data_type FROM SIM_SEARCH SS JOIN DATASET_VECTORS DV ON SS.VECTOR_ID=DV.ID where SS.NAME=?"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=(sim_search, ))
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        dataset_id, data_type = results[0]

        query = "SELECT DATASET_NAME, DATAPATH FROM DATASETS WHERE ID=?"
        data_res = execute_query(conn=cursor, query=query, values=(dataset_id, ))
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]

        dataset_name, datapath = results[0]
        return datapath, data_type
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_sim_search_by_name(conn, sim_search):
    query = "SELECT pred_datapath, stored_sim_search FROM SIM_SEARCH where name=?"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=(sim_search, ))
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        prediction_dataset_path, sim_search_vec = results[0]
        return prediction_dataset_path, sim_search_vec
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def add_operator_model_results(conn, name, output_path, sim_search_id, metrics, operator):
    query = "INSERT INTO Operator_Modelling(name, output_path, sim_search_id, metrics, operator) VALUES(?,?,?,?,?)"
    try:
        execute_query(conn=conn, query=query, values=(name, output_path, sim_search_id, metrics, operator))
        conn.commit()
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_all_operators(conn):
    query = "SELECT ID, NAME FROM Operator_Modelling"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=None)
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        return results
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_metric_path(conn, name):
    query = 'SELECT METRICS, SIM_SEARCH_ID, OPERATOR FROM Operator_Modelling WHERE NAME = ?'
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=(name, ))
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        print(results)
        metric_path, sim_search_idx, operator_name = results[0]
        return metric_path, sim_search_idx, operator_name 
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

def get_sim_search_name_by_id(conn, idx):
    # query = 'SELECT METRICS, SIM_SEARCH_ID, OPERATOR FROM Operator_Modelling WHERE NAME = ?'
    query = "SELECT pred_datapath, name FROM SIM_SEARCH where id=?"
    try:
        cursor = conn.cursor()
        data_res = execute_query(conn=cursor, query=query, values=(idx, ))
        data_res = cursor.fetchall()
        results = [list(tuple(row)) for row in data_res]
        print(results)
        pred_datapath, name  = results[0]
        return pred_datapath, name  
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))

