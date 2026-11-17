from sklearn.decomposition import PCA
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import matplotlib.colors as colors
import pandas as pd
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_TYPES = {1: 'Tabular', 2: 'Graph', 3: 'Image'}

TABFM_TASK = 'regression'

OPERATORS = {
    1: [('linear_regression', 'Linear Regression'),
        ('mlp_regression', 'MLP Regression'),
        ('svm_sgd', 'SVM (SGD)'),
        ('svm', 'SVM'),
        ('logistic_regression', 'Logistic Regression'),
        ('mlp', 'MLP (Perceptron)')],
    2: [('bc', 'Betweenness Centrality'),
        ('ebc', 'Edge Betweenness Centrality'),
        ('cc', 'Closeness Centrality'),
        ('ec', 'Eigenvector Centrality'),
        ('pr', 'PageRank'),
        ('sr', 'Spectral Radius')],
    3: [('svm_sgd', 'SVM (SGD)'),
        ('svm', 'SVM'),
        ('logistic_regression', 'Logistic Regression'),
        ('mlp', 'MLP (Perceptron)'),
        ('knn', 'K-Nearest Neighbours')],
}

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS Store_Vectors(store_id INTEGER PRIMARY KEY AUTOINCREMENT, store_name TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS Vector_Dimension(VEC_DIM INT PRIMARY KEY NOT NULL)",
    "CREATE TABLE IF NOT EXISTS Dataset_Vectors(id INTEGER PRIMARY KEY AUTOINCREMENT, vector_name TEXT NOT NULL, store_id INTEGER NOT NULL, data_type INT NOT NULL default 1, Dataset_id INT, embedder TEXT, model_path TEXT, vec_dim INT, FOREIGN KEY(store_id) REFERENCES Store_Vectors(store_id))",
    "CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_name TEXT NOT NULL UNIQUE, datapath TEXT NOT NULL, data_type INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS Models_Table(id INT PRIMARY KEY NOT NULL, model_name TEXT NOT NULL, vec_dim INT NOT NULL, model_path TEXT NOT NULL, data_type INT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS Operator_Modelling(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, output_path TEXT NOT NULL, sim_search_id INT NOT NULL, metrics TEXT NOT NULL, operator TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS Sim_Search(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, pred_datapath TEXT NOT NULL, stored_sim_search TEXT NOT NULL, vector_id INTEGER REFERENCES Dataset_Vectors(id))",
    "CREATE TABLE IF NOT EXISTS tasks(task_id INTEGER PRIMARY KEY AUTOINCREMENT, task_number INTEGER UNIQUE NOT NULL, execution_time DOUBLE, is_complete INTEGER NOT NULL, description TEXT)",
]


def ensure_schema(conn):
    for statement in SCHEMA:
        conn.execute(statement)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(Dataset_Vectors)").fetchall()]
    for column, col_type in (('embedder', 'TEXT'), ('model_path', 'TEXT'), ('vec_dim', 'INT')):
        if column not in columns:
            conn.execute(f"ALTER TABLE Dataset_Vectors ADD COLUMN {column} {col_type}")
    if conn.execute("SELECT COUNT(*) FROM Store_Vectors").fetchone()[0] == 0:
        conn.executemany("INSERT INTO Store_Vectors(store_id, store_name) VALUES(?, ?)", [(1, 'Local Store'), (2, 'Qdrant')])
    conn.commit()


def resolve_path(path):
    if path is None or path == '':
        return path
    path = os.path.expanduser(str(path).strip())
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def check_dataset(filepath):
    if os.path.exists(resolve_path(filepath)):
        print('dataset_exits')
        return True
    else:
        print('dataset does not exits')
        return False

def check_dataset_file(filepath):
    if os.path.exists(resolve_path(filepath)):
        print('dataset_exits')
        return True
    else:
        print('dataset does not exits')
        return False


def get_embedding_models(conn):
    models = {}
    groups = {name: [] for name in DATA_TYPES.values()}
    try:
        rows = conn.execute("SELECT id, model_name, vec_dim, model_path, data_type FROM Models_Table ORDER BY data_type, vec_dim").fetchall()
    except Exception as err:
        print('Query Failed: %s' % str(err))
        rows = []
    for idx, model_name, vec_dim, model_path, data_type in rows:
        if data_type not in (1, 2):
            continue
        embedder = 'graph2vec' if data_type == 2 else 'num2vec'
        key = f'{embedder}:{idx}'
        models[key] = {'data_type': data_type, 'embedder': embedder, 'model_path': model_path, 'vec_dim': vec_dim}
        label = 'Num2Vec' if embedder == 'num2vec' else 'Graph2Vec'
        groups[DATA_TYPES[data_type]].append((key, f'{label} {model_name} ({vec_dim}-d)'))

    models['tabfm'] = {'data_type': 1, 'embedder': 'tabfm', 'model_path': None, 'vec_dim': None}
    groups['Tabular'].append(('tabfm', 'TabFM (google/tabfm-1.0.0-pytorch)'))
    models['dataset2vec'] = {'data_type': 1, 'embedder': 'dataset2vec', 'model_path': None, 'vec_dim': 32}
    groups['Tabular'].append(('dataset2vec', 'Dataset2Vec (32-d)'))

    for vec_dim in get_all_vec_dimension(conn):
        key = f'vit:{vec_dim}'
        models[key] = {'data_type': 3, 'embedder': 'vit', 'model_path': None, 'vec_dim': vec_dim}
        groups['Image'].append((key, f'ViT google/vit-base-patch16-224 ({vec_dim}-d)'))

    choices = {group: options for group, options in groups.items() if options}
    return models, choices


def get_operator_choices():
    choices = {}
    for data_type, operators in OPERATORS.items():
        choices[DATA_TYPES[data_type]] = [(f'{data_type}:{code}', label) for code, label in operators]
    return choices


def _project_3d(vec_np):
    vec_np = np.asarray(vec_np, dtype=np.float64)
    if vec_np.ndim == 1:
        vec_np = vec_np.reshape(1, -1)
    n_components = min(3, vec_np.shape[0], vec_np.shape[1])
    if vec_np.shape[0] > 1:
        result = PCA(n_components=n_components).fit_transform(vec_np)
    else:
        result = vec_np[:, :n_components]
    if result.shape[1] < 3:
        result = np.hstack([result, np.zeros((result.shape[0], 3 - result.shape[1]))])
    return result


def create_vec_repr_plot(vectors_):
    img = BytesIO()
    vec = list(vectors_.values())
    plt.set_loglevel('info')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    result = _project_3d(vec)

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
        vec_np = np.array(list(dict_vectors.values()), dtype=np.float64)

        labels_ = ['Relevant Dataset' if i in most_rel_names else 'Data Lake' for i in dict_vectors.keys()]

        vec_concat_np = np.concatenate((vec_np, [np.asarray(query_vector, dtype=np.float64)]), axis=0)
        result = _project_3d(vec_concat_np)
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
    
def insert_vector_location(conn, filename, store_into, data_type, dataset_id, embedder=None, model_path=None, vec_dim=None):
    query_1 = "SELECT * FROM STORE_VECTORS WHERE store_name=?"
    try:
        data_res = execute_query(conn=conn, query=query_1, values=(store_into,)).fetchall()
    except Exception as err:
            print('Query Failed: %s\nError: %s' % (query_1, str(err)))
    results = [tuple(row) for row in data_res]
    store_id, _ = results[0]

    query_2 = "INSERT INTO DATASET_VECTORS(vector_name, store_id, data_type, dataset_id, embedder, model_path, vec_dim) VALUES(?, ?, ?, ?, ?, ?, ?)"
    try:
        execute_query(conn=conn, query=query_2, values=(filename, store_id, data_type, dataset_id, embedder, model_path, vec_dim, ))
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

def update_task_failed(conn, thread_id, error):
    query = "UPDATE TASKS SET is_complete = ?, description = description || ? WHERE task_number = ?"
    try:
        execute_query(conn=conn, query=query, values=(-1, f' | Failed: {error}', thread_id,))
        conn.commit()
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
        return results
    except Exception as err:
        print('Query Failed: %s\nError: %s' % (query, str(err)))
        return []

def get_vector_info(conn, vector_name):
    query = "SELECT SV.store_name, DT.data_type, DT.embedder, DT.model_path, DT.vec_dim FROM DATASET_VECTORS DT JOIN STORE_VECTORS SV ON DT.store_id=SV.store_id WHERE DT.vector_name = ?"
    row = conn.execute(query, (vector_name, )).fetchone()
    if row is None:
        return None
    store_name, data_type, embedder, model_path, vec_dim = tuple(row)
    if embedder is None:
        embedder = {1: 'num2vec', 2: 'graph2vec', 3: 'vit'}.get(data_type)
    return {'store': store_name, 'data_type': data_type, 'embedder': embedder, 'model_path': model_path, 'vec_dim': vec_dim}

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

