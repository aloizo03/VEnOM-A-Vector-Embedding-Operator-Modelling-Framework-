from flask import Flask, render_template, request, redirect, flash
from server_utils.utils import *
from server_utils.control_Framework import control_Framework, Operator_Modelling
from utils_flask import *
from threading import Thread
import threading
import sqlite3
# from flask_socketio import SocketIO

from flask import render_template, Response
import time


app = Flask(__name__)
app.config['SECRET_KEY'] = "IAMDEAD"

# Initialize SocketIO with async_mode='eventlet'
# socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")
controller_ = control_Framework()

# @socketio.on("connect")
# def handle_connect():
#     print("Client connected!")
database_file = '/home/ubuntu/Projects/VectorEmbedding-Representation-/server/server_utils/vector_OM.db'
save_locally_folder = '/home/ubuntu/Projects/VectorEmbedding-Representation-/server/tmp_locally_store'
save_sim_search_locally = "/home/ubuntu/Projects/VectorEmbedding-Representation-/server/sim_search_store"

def get_db_connection():
    conn = sqlite3.connect(database_file)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def home():
    name = "null"
    return render_template("home.html", page_name=name)


@app.route("/display_Datasets", methods=['GET', 'POST'])
def display_datasets():
    page_name = 'show_datasets'
    conn = get_db_connection()
    avail_datasets = get_all_datasets(conn=conn)
    
    result = []
    for dataset_ in avail_datasets:
        dataset_idx, dataset_name, file_path, dataset_type = dataset_
        try:
            # print(os.listdir(file_path))
            total_avil_files = len([name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name))])
        except:
            total_avil_files = 0

        dict_ = {'dataset_name_text': dataset_name,
                 'filepath_text': file_path,
                 'total_datasets_text': str(total_avil_files),
                 'dataset_type_text': 'Tabular Dataset' if dataset_type == 1 else 'Graphs'}
        result.append(dict_)

    
    table_form = updatedDatasetForm()
    table_form.process(data={'field_list': result})

    return render_template("home.html",
                           page_name=page_name,
                           task_form=table_form)


@app.route("/onclick_load_data", methods=['GET', 'POST'])
def load_data():
    page_name = "load_data"
    name = None
    dataset_name = None
    load_data_form = LoadDataForm()
    if load_data_form.validate_on_submit():
        data_path = load_data_form.Dataset_DIR.data
        dataset_name = load_data_form.Dataset_Name.data
        data_type = int(load_data_form.Dataset_type.data)
        #TODO: Check if exists the dataset
        if check_dataset(data_path):

            conn = get_db_connection()
            insert_dataset(conn=conn,
                        data_path=data_path,
                        dataset_name=dataset_name,
                        data_type=data_type)
            conn.close()
        else:
            #TODO: Sent message with socket IO
            pass            
        load_data_form.Dataset_DIR.data = ''
        load_data_form.Dataset_Name.data = ''

    return render_template("home.html",
                           page_name=page_name,
                           dataset_name=dataset_name,
                           name=name,
                           form=load_data_form)

@app.route("/show_tasks", methods=['GET', 'POST'])
def show_tasks():
    page_name = "display_tasks"
    conn = get_db_connection()
    all_tasks = get_all_Tasks(conn=conn)
    result = []
    for task in all_tasks:
        task_id, task_number, exec_time, is_complete, description = task
        is_complete = bool(is_complete)
        dict_ = {'TaskId_text': str(task_number),
                 'Status_text': 'Complete' if is_complete else 'On progress',
                 'Exec_Time_text': str(exec_time) if is_complete else '',
                 'Description_text': description}
        result.append(dict_)

    
    table_form = updatedTaskForm()
    table_form.process(data={'field_list': result})
    # print(table_form.data)

    conn.close()
    return render_template("home.html",
                           page_name=page_name,
                           task_form=table_form)
    

# @app.route("/onclick_create_vec/<socketid>", methods=['POST'])
# def vec_dataset(socketid):
#     pass

# @app.route("/onclick_vec/<socketid>", methods=['POST'])
@app.route("/onclick_vec/", methods=['GET', 'POST'])
def vec_dataset():
    #todo: Fix there the URL Not found
    name = "vectorisation"
    conn = get_db_connection()
    dataset_names_lst = get_all_datasets(conn=conn)
    dataset_names = {str(id_): dataset_name for id_, dataset_name, data_path, data_type in dataset_names_lst}

    vec_dim_lst = get_all_vec_dimension(conn=conn)
    
    vec_sizes = {idx: vec_dim for idx, vec_dim in enumerate(vec_dim_lst)}

    saves_option = get_avail_store_options(conn=conn)
    save_into = {str(id_): save_to for id_, save_to in saves_option}
    conn.close()
    vector_form = VectorisationForm(vector_sizes=vec_sizes, avail_datasets=dataset_names, save_dbs=save_into)

    if request.method == 'POST':
        conn = get_db_connection()

        # print(dataset_names)
        # print(str(vector_form.select_dataset.data))
        # print(vector_form.select_vec_size.data)
        dataset_name = dataset_names[str(vector_form.select_dataset.data)]
        
        vec_dim = vec_sizes[vector_form.select_vec_size.data]
        save_to = save_into[str(vector_form.save_into.data)]

        ret_data = get_model_path(conn=conn, dataset_name=dataset_name, vec_size=vec_dim)

        if ret_data is not None:
            model_name, vec_dim, model_path, data_type = ret_data
            controller_.set_selected_datasets(selected_Dataset=dataset_name)
            controller_.set_selected_Vec_Size(selected_Vector_Size=vec_dim)
            controller_.set_selected_save_vecs(save_Vectors=save_to)

            ret_data_dataset = get_dataset(conn=conn, dataset_name=dataset_name)
            idx_, dataset_name, dataset_path, data_type = ret_data_dataset
            
            Thread(target=back_task_create_vect, args=(model_path, dataset_path, dataset_name, save_locally_folder, save_to, data_type, vector_form.select_dataset.data)).start()
            
        conn.close()
    return render_template("home.html", page_name=name, form=vector_form)

def back_task_create_vect(model_path, data_path, dataset_name, out_path, save_to, data_type, dataset_id):
    thread_id = threading.current_thread().native_id
    desc_ = f"Create vectorization for dataset {dataset_name} and save into {save_to}"
    conn = get_db_connection()
    
    # socketio.emit("task_started", {"message": "Vectorisation started!"})
    insert_task(conn=conn, thread_id=thread_id, description=desc_)
    start_time = time.time()
    vectors_ = controller_.create_Vectorisation(data_path=data_path,
                                                model_path=model_path,
                                                out_path=out_path,
                                                show_progress=False, 
                                                data_type=data_type)
    
    filename = controller_.store_vectors(save_to=save_to, folder=save_locally_folder, vectors_dict=vectors_, dataset_name=dataset_name)

    exec_time = time.time() - start_time
    print(f'Finish vectorisation on {exec_time} seconds')
    insert_vector_location(conn=conn, store_into=save_to, filename=filename, data_type=data_type, dataset_id=dataset_id)
    update_task_complete(conn=conn, thread_id=thread_id, exec_time=exec_time)

    conn.close()
    # socketio.emit("task_end", {"message": "Vectorisation ended!"})


@app.route("/display_vecs", methods=['GET', 'POST'])
def display_vec_dataset():
    name = "display_vectors"
    data_vectorisation = {'1': 'data/hpc_vec.pkl', '2': 'data/hpc_weather.pkl'}
    conn = get_db_connection()

    data_vectors = get_all_vector_datasets(conn=conn)
    
    avail_data_vectorisation = {str(id_): None for id_, vec_name, store_id, store_name, dataset_id in data_vectors}
    avail_data_vectors_full = {str(id_): vec_name for id_, vec_name, store_id, store_name, dataset_id in data_vectors}
    
    for id_, vec_name, store_id, store_name, _ in data_vectors:
        if store_name.lower() == 'local store':
            vec_name = vec_name.split('/')[-1].split('.')[0].lower()
        else:
            vec_name = f"{store_name} {vec_name}"
        avail_data_vectorisation[str(id_)] = vec_name

    displ_vec = DisplayVectorEmbeddingRepr(vec_emb_reprs=avail_data_vectorisation)
    vec_dir = '/opt/dlami/nvme/Vector_Embedding/Results_Experiments_2/HPC_2K/200V/vectors_1731669493.3926575.pickle'
    conn.close()
    if request.method == 'POST':
        selected_data_vectors = avail_data_vectors_full[str(displ_vec.select_ver.data)]
        conn = get_db_connection()
        stored_into, data_type = find_database_store(conn=conn, vector_name=selected_data_vectors)
        conn.close()
        if 'Qdrant' == stored_into:
            print(selected_data_vectors)
            vectors_ = controller_.get_vectors_from_vectorDB(database_='Qdrant', collection_name=selected_data_vectors)
        else:
            vectors_ = read_vectors(dir_path=vec_dir)
        
        plot_url = create_vec_repr_plot(vectors_=vectors_)
        return render_template("home.html", page_name=name, form=displ_vec, display_vecs='True', plot_url=plot_url)
    return render_template("home.html", page_name=name, form=displ_vec, display_vecs='False')

@app.route('/sim_search', methods=['GET', 'POST'])
def sim_search():
    name = "similarity_search"
    sim_search_methods = {'1': 'Vector DB Search', '2': 'Cosine Similarity', '3': 'K-Means', '4': 'Euclidean Distance',}

    sim_search_idx = {'1': 'Vector DB Search', '2': 'cosine', '3': 'K-Means', '4': 'distance',}
    conn = get_db_connection()

    data_vectors = get_all_vector_datasets(conn=conn)
    
    avail_data_vectorisation = {str(id_): None for id_, vec_name, store_id, store_name, dataset_id in data_vectors}
    avail_data_vectors_full = {str(id_): vec_name for id_, vec_name, store_id, store_name, dataset_id in data_vectors}
    
    for id_, vec_name, store_id, store_name, dataset_id in data_vectors:
        if store_name.lower() == 'local store':
            vec_name = vec_name.split('/')[-1].split('.')[0].lower()
        else:
            vec_name = f"{store_name} {vec_name}"
        avail_data_vectorisation[str(id_)] = vec_name
    conn.close()
    sim_search_form = SimilaritySearchForm(avail_sim_search=sim_search_methods, avail_vec_emb_reprs=avail_data_vectorisation)
    
    if request.method == 'POST':
       
        sim_search_technique = sim_search_idx[str(sim_search_form.select_sim_search.data)]
        
        selected_data_vectors = avail_data_vectors_full[str(sim_search_form.select_avail_vec_repr.data)]
        sim_search_data_name = sim_search_form.sim_search_name.data
        data_path = sim_search_form.pred_data_dir_path.data
        data_selection = float(sim_search_form.select_dataset.data)
        
        conn = get_db_connection()
        stored_into, data_type = find_database_store(conn=conn, vector_name=selected_data_vectors)
        conn.close()
        Thread(target=back_task_get_vectors, args=(data_path, sim_search_data_name, sim_search_technique, selected_data_vectors, stored_into, data_type, data_selection)).start()
        # sim_search_form = SimilaritySearchForm(avail_sim_search=sim_search_methods, avail_vec_emb_reprs=avail_data_vectorisation)
        return render_template("home.html", page_name=name, form=sim_search_form)
    
    return render_template("home.html", page_name=name, form=sim_search_form)

def back_task_get_vectors(data_path, sim_search_data_name, sim_search_technique, data_vectors, stores_into, data_type, select_percent):
    
    flag_check = check_dataset_file(filepath=data_path)
    if not flag_check:
        # Send that did not exists the dataset with socket
        return
       
    
    conn = get_db_connection()
    thread_id = threading.current_thread().native_id
    desc_ = f"Similarity search from Qdrant collection {data_vectors}"
    
    vectors_idx, _ = get_vectors_idx_from_name(conn, data_vectors)
    insert_task(conn=conn, thread_id=thread_id, description=desc_)

    if sim_search_technique == 'Vector DB Search':
        # Apply vector db search
        if 'Qdrant' == stores_into:
        #    
            start_time = time.time()
            vec_dim = controller_.get_vector_size(database_=stores_into, collection_name=data_vectors)
            print(vec_dim, '    ', data_type)
            model_path = get_model_path_from_size(conn=conn, vec_dim=vec_dim, data_type=data_type)
            vectors = get_dataset_vector(model_path=model_path, data_path=data_path, data_type=data_type)
            
            sim_search_vectors = controller_.similarity_search(database_=stores_into, collection_name=data_vectors, vectors=vectors, select_percent=select_percent)
            
            stored_filepath = controller_.store_similarity_search_locally(out_path=save_sim_search_locally,
                                                                          name=sim_search_data_name,
                                                                          vectors_dict=sim_search_vectors)
            exec_time = time.time() - start_time
            insert_sim_search_vectors(conn=conn, vector_name=sim_search_data_name, dataset_path=data_path, saved_datapath=stored_filepath, vectors_idx=vectors_idx)
            update_task_complete(conn=conn, thread_id=thread_id, exec_time=exec_time)
    else:
        start_time = time.time()

        vec_dim = controller_.get_vector_size(database_=stores_into, collection_name=data_vectors)
        model_path = get_model_path_from_size(conn=conn, vec_dim=vec_dim, data_type=data_type)

        sim_search_vectors = controller_.simimilarity_search_localy(data_path=data_path,
                                                                    vectors_path=data_vectors, 
                                                                    data_radio=select_percent,
                                                                    data_selection=sim_search_technique, 
                                                                    model_path=model_path,
                                                                    out_path=save_sim_search_locally, 
                                                                    data_type=data_type)
        stored_filepath = controller_.store_similarity_search_locally(out_path=save_sim_search_locally,
                                                                          name=sim_search_data_name,
                                                                          vectors_dict=sim_search_vectors)

        exec_time = time.time() - start_time
        insert_sim_search_vectors(conn=conn, vector_name=sim_search_data_name, dataset_path=data_path, saved_datapath=stored_filepath, vectors_idx=vectors_idx)
        update_task_complete(conn=conn, thread_id=thread_id, exec_time=exec_time)
        # vectors = controller_.get_vectors_from_vectorDB(stores_into, data_vectors)
        
    conn.close()

@app.route("/display_sim_vecs", methods=['GET', 'POST'])
def display_sim_vec_dataset():
    #TODO: Add on the db the id of the vectors that used for the vector embedding
    name = "display_sim_vectors"
    data_vectorisation = {'1': 'data/vectors_hpc_SimSearch.pkl', '2': 'data/vectors_k-means.pkl'}

    conn = get_db_connection()

    avail_sim_search_vec = get_similarity_search(conn=conn)

    data_vectors = get_all_vector_datasets(conn=conn)
    
    avail_data_vectorisation = {str(id_): None for id_, vec_name, store_id, store_name, dataset_id in data_vectors}
    avail_data_vectors_full = {str(id_): vec_name for id_, vec_name, store_id, store_name, dataset_id in data_vectors}
    for id_, vec_name, store_id, store_name, dataset_id in data_vectors:
        if store_name.lower() == 'local store':
            vec_name = vec_name.split('/')[-1].split('.')[0].lower()
        else:
            vec_name = f"{store_name} {vec_name}"
        avail_data_vectorisation[str(id_)] = vec_name
    
    avil_vec_sim = {str(idx): name for idx, name, pred_path, stored_sim_search, vec_id in avail_sim_search_vec}
    avail_vec_sim_vectors_id = {str(idx): vec_id for idx, name, pred_path, stored_sim_search, vec_id in avail_sim_search_vec}

    # data_vectorisation = {'1': 'data/hpc_vec.pkl', '2': 'data/hpc_weather.pkl'}
    displ_vec = DisplayVectorEmbeddingRepr(vec_emb_reprs=avil_vec_sim)

    vec_ss_dir = '/opt/dlami/nvme/Vector_Embedding/Results_Experiments_2/HPC_2K/200V/Dataset_Selection/rel_data_cosine_1733225609.908662.pickle'
    conn.close()
    if request.method == 'POST':
        conn = get_db_connection()
        selected_vec = avil_vec_sim[str(displ_vec.select_ver.data)]
        vector_id = avail_vec_sim_vectors_id[str(displ_vec.select_ver.data)]

        prediction_dataset_path, sim_search_vec = get_sim_search_by_name(conn=conn, sim_search=selected_vec)
        
        selected_data_vectors = avail_data_vectors_full[str(vector_id)]
        conn = get_db_connection()
        stored_into, data_type = find_database_store(conn=conn, vector_name=selected_data_vectors)
        conn.close()
        if 'Qdrant' == stored_into:
            vectors_ = controller_.get_vectors_from_vectorDB(database_='Qdrant', collection_name=selected_data_vectors)
        else:
            vectors_ = read_vectors(dir_path=selected_data_vectors)
        
        plot_url = create_vec_repr_plot(vectors_=vectors_)

        vectors_similarity_search = read_vectors(sim_search_vec)
        plot_url = create_sim_search_vector_plot(vectors_, vectors_similarity_search)
        conn.close()
        return render_template("home.html", page_name=name, form=displ_vec, display_vecs='True', plot_url=plot_url)
    return render_template("home.html", page_name=name, form=displ_vec)

@app.route("/op_model", methods=['GET', 'POST'])
def operator_modelling():
    name = 'operator_modelling'
    avail_operators = {'1': 'SVM', 
                       '9': 'SVM SGD',
                       '2': 'MLP', 
                       '8': 'MLP Regression',
                       '3': 'Linear Regression', 
                       '4': 'Arima', 
                       '5': 'Holt-Winter',
                       '6': 'Logistic Regression', 
                       '7': 'DBSCAN',
                       '10': 'Betweenness Centrality',
                       '11': 'Edge Betweenness Centrality',
                       '12': 'Closeness Centrality',
                       '13': 'Eigenvector Centrality',
                       '14': 'PageRank'
                       }
    
    idx_to_code_operator = {'1': 'svm', 
                            '2':'mlp_sgd', 
                            '3': 'linear_regression', 
                            '4': 'arima', 
                            '5': 'holt_winter',
                            '6': 'logistic_regression', 
                            '7': 'dbscan', 
                            '8':'mlpregression',
                            '9': 'svm_sgd',
                            '10': 'bc',
                            '11': 'ebc', 
                            '12': 'cc', 
                            '13': 'ec',
                            '14': 'pr'}
    
    conn = get_db_connection()
    avail_sim_search_vec = get_similarity_search(conn=conn)

    
    avil_vec_sim = {str(idx): name for idx, name, pred_path, stored_sim_search, vec_idx in avail_sim_search_vec}

    form = OperatorModellingForm(operator_algorithms=avail_operators, vectors=avil_vec_sim)
    if request.method == 'POST':
        model_op = True
        selected_vec_sim = avil_vec_sim[str(form.select_avail_vec_repr.data)]
        selected_operator = idx_to_code_operator[str(form.select_avail_oper.data)]
        
        selected_op_name = avail_operators[str(form.select_avail_oper.data)]
        operator_name = form.operator_name.data
        operator_outpath = form.operator_dir.data

        Thread(target=back_task_operator_modelling, args=(operator_name, operator_outpath, selected_operator, selected_vec_sim, str(form.select_avail_vec_repr.data), selected_op_name)).start()
    conn.close()
    return render_template("home.html", page_name=name, form=form)

def back_task_operator_modelling(oper_model_name, out_path, operator, sim_search, sim_search_idx, selected_op_name):
    thread_id = threading.current_thread().native_id
    conn = get_db_connection()
    
    desc_ = f"Operator Modelling for {operator} in similarity search vectors {sim_search}"
    
    prediction_dataset_path, sim_search_vec = get_sim_search_by_name(conn=conn,
                                                                     sim_search=sim_search)
    
    datapath, data_type = get_dataset_id_from_sim_search(conn=conn, sim_search=sim_search)

    insert_task(conn=conn, thread_id=thread_id, description=desc_)
    op_Modelling = Operator_Modelling(operator_name=oper_model_name,
                                      operator_outpath=out_path,
                                      operator=operator,
                                      sim_search_vectors=sim_search_vec, 
                                      prediction_data_path=prediction_dataset_path, 
                                      directory_datapath=datapath, 
                                      data_type=data_type)
    
    path_labels = op_Modelling.create_labels()
    start_time = time.time()

    results_, out_file = op_Modelling.model_operator(path_labels=path_labels)
    
    print('Finish operator modelling')
    exec_time = time.time() - start_time
    results_['Exec time'] = exec_time
    df_res = pd.DataFrame([results_])
    df_res.to_csv(out_file, index=False) 

    add_operator_model_results(conn=conn, name=oper_model_name, output_path=out_path, sim_search_id=sim_search_idx, metrics=out_file, operator=selected_op_name)
    update_task_complete(conn=conn, thread_id=thread_id, exec_time=exec_time)
    conn.close()
    # insert_task(conn=conn, thread_id=thread_id, description=desc_)

@app.route("/display_op", methods=['GET', 'POST'])
def operator_modelling_display():
    conn = get_db_connection()
    name = 'operator_modelling_display'
    # avail_op_models = {'1': 'LR_HPC',
    #                    '2': 'ARIMA_HPC',
    #                    '3': 'MLP_Regression_HPC'}
    avail_op_models = get_all_operators(conn=conn)
    avail_op_models_dict = {str(idx): name for idx, name in avail_op_models}

    form = OperatorModellingDisplayForm(avail_op_models_select=avail_op_models_dict)

    if request.method == 'POST':
        selected_operator = avail_op_models_dict[str(form.avail_op_models.data)]
        metric_path, sim_search_idx, operator_name = get_metric_path(conn, selected_operator)
        
        pred_datapath, sim_search_name = get_sim_search_name_by_id(conn=conn, idx=sim_search_idx)
        description = f'Operator Modelling for similarity search {sim_search_name} to model operator {operator_name}'
        df = pd.read_csv(metric_path, index_col=False)
        print(df)
        df = df.drop(columns=['Acc', 'r^2', 'NRMSE', 'MaPE'])
        df_tmp = df
        df = df.iloc[0]
        results = {'rmse': df['RMSE'],
                   'mae': df['MAE'], 
                   'mad': df['MAD'],
                   'exec_time': df['Exec time']}
        table_form = updatedOpModelForm()
        table_form.process(data={'field_list': [results]})
        print(results)
        conn.close()
        # plot_url = create_res_plot(res_loss=loss_res, labels=labels)
        return render_template("home.html", page_name=name, form=form, task_form=table_form, desc_p=description, display_plot='True')
    conn.close()
    return render_template("home.html", page_name=name, form=form)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)