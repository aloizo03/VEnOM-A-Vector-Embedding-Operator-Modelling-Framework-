from flask import Flask, render_template, request, redirect, flash
from server_utils.utils import *
from server_utils.control_Framework import control_Framework, Operator_Modelling
from utils_flask import *
from threading import Thread
import threading
import traceback
import uuid
import pandas as pd
import sqlite3
import os
# from flask_socketio import SocketIO

from flask import render_template, Response
import time


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('VENOM_SECRET_KEY', "IAMDEAD")

# Initialize SocketIO with async_mode='eventlet'
# socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")
controller_ = control_Framework()

# @socketio.on("connect")
# def handle_connect():
#     print("Client connected!")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_file = os.environ.get('VENOM_DB_FILE', os.path.join(_REPO_ROOT, 'server', 'server_utils', 'vector_OM.db'))
save_locally_folder = os.environ.get('VENOM_LOCAL_STORE', os.path.join(_REPO_ROOT, 'server', 'tmp_locally_store'))
save_sim_search_locally = os.environ.get('VENOM_SIM_SEARCH_STORE', os.path.join(_REPO_ROOT, 'server', 'sim_search_store'))

SIM_SEARCH_METHODS = {'1': ('Vector DB Search', 'vectordb'),
                      '2': ('Cosine Similarity', 'cosine'),
                      '3': ('K-Means', 'k-means'),
                      '4': ('Euclidean Distance', 'distance'),
                      '5': ('Random', 'random')}

def get_db_connection():
    conn = sqlite3.connect(database_file)
    conn.row_factory = sqlite3.Row
    return conn

_conn = sqlite3.connect(database_file)
ensure_schema(_conn)
_conn.close()

def new_task_id():
    return uuid.uuid4().int >> 66

def run_task(description, target):
    task_id = new_task_id()
    conn = get_db_connection()
    insert_task(conn=conn, thread_id=task_id, description=description)
    conn.close()
    start_time = time.time()
    try:
        target()
    except Exception as err:
        traceback.print_exc()
        conn = get_db_connection()
        update_task_failed(conn=conn, thread_id=task_id, error=f'{type(err).__name__}: {err}')
        conn.close()
        return
    conn = get_db_connection()
    update_task_complete(conn=conn, thread_id=task_id, exec_time=time.time() - start_time)
    conn.close()

def start_task(description, target):
    Thread(target=run_task, args=(description, target), daemon=True).start()

def vector_display_names(data_vectors):
    names = {}
    for id_, vec_name, store_id, store_name, _ in data_vectors:
        if store_name.lower() == 'local store':
            vec_name = vec_name.replace('\\', '/').split('/')[-1].rsplit('.', 1)[0].lower()
        else:
            vec_name = f"{store_name} {vec_name}"
        names[str(id_)] = vec_name
    return names

@app.route("/")
def home():
    name = "null"
    return render_template("home.html", page_name=name)


@app.route("/display_Datasets", methods=['GET', 'POST'])
def display_datasets():
    page_name = 'show_datasets'
    conn = get_db_connection()
    avail_datasets = get_all_datasets(conn=conn)
    conn.close()

    result = []
    for dataset_ in avail_datasets:
        dataset_idx, dataset_name, file_path, dataset_type = dataset_
        full_path = resolve_path(file_path)
        if os.path.isdir(full_path):
            total_avil_files = len([name for name in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, name))])
        elif os.path.isfile(full_path):
            total_avil_files = 1
        else:
            total_avil_files = 0

        dict_ = {'dataset_name_text': dataset_name,
                 'filepath_text': file_path,
                 'total_datasets_text': str(total_avil_files),
                 'dataset_type_text': DATA_TYPES.get(dataset_type, str(dataset_type))}
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
        data_path = load_data_form.Dataset_DIR.data.strip()
        dataset_name = load_data_form.Dataset_Name.data.strip()
        data_type = int(load_data_form.Dataset_type.data)
        if check_dataset(data_path):
            conn = get_db_connection()
            existing = insert_dataset(conn=conn,
                                      data_path=data_path,
                                      dataset_name=dataset_name,
                                      data_type=data_type)
            conn.close()
            if existing:
                flash(f'A dataset named "{dataset_name}" already exists.', 'warning')
            else:
                flash(f'{DATA_TYPES[data_type]} dataset "{dataset_name}" added.', 'success')
        else:
            flash(f'Path not found: {resolve_path(data_path)}', 'danger')
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
    all_tasks = get_all_Tasks(conn=conn) or []
    result = []
    for task in all_tasks:
        task_id, task_number, exec_time, is_complete, description = task
        if is_complete == 1:
            status = 'Complete'
        elif is_complete == -1:
            status = 'Failed'
        else:
            status = 'On progress'
        dict_ = {'TaskId_text': str(task_number),
                 'Status_text': status,
                 'Exec_Time_text': str(exec_time) if is_complete == 1 else '',
                 'Description_text': description}
        result.append(dict_)


    table_form = updatedTaskForm()
    table_form.process(data={'field_list': result})

    conn.close()
    return render_template("home.html",
                           page_name=page_name,
                           task_form=table_form)


@app.route("/onclick_vec/", methods=['GET', 'POST'])
def vec_dataset():
    name = "vectorisation"
    conn = get_db_connection()
    dataset_rows = get_all_datasets(conn=conn)
    datasets = {str(id_): (dataset_name, data_path, data_type) for id_, dataset_name, data_path, data_type in dataset_rows}
    dataset_names = {id_: f'{dataset_name} ({DATA_TYPES.get(data_type, data_type)})' for id_, (dataset_name, _, data_type) in datasets.items()}

    embedding_models, model_choices = get_embedding_models(conn=conn)

    saves_option = get_avail_store_options(conn=conn)
    save_into = {str(id_): save_to for id_, save_to in saves_option}
    conn.close()
    vector_form = VectorisationForm(embedding_models=model_choices, avail_datasets=dataset_names, save_dbs=save_into)

    if request.method == 'POST':
        dataset_id = str(vector_form.select_dataset.data)
        model_key = vector_form.select_model.data
        save_to = save_into.get(str(vector_form.save_into.data))

        if dataset_id not in datasets or model_key not in embedding_models or save_to is None:
            flash('Select a dataset, an embedding model and where to save the vectors.', 'danger')
            return render_template("home.html", page_name=name, form=vector_form)

        dataset_name, dataset_path, data_type = datasets[dataset_id]
        model = embedding_models[model_key]
        if model['data_type'] != data_type:
            flash(f'"{dataset_name}" is a {DATA_TYPES[data_type]} dataset; choose a {DATA_TYPES[data_type]} embedding model.', 'danger')
            return render_template("home.html", page_name=name, form=vector_form)

        start_task(f"Create vectorization for dataset {dataset_name} with {model['embedder']} and save into {save_to}",
                   lambda: back_task_create_vect(model, dataset_path, dataset_name, save_to, data_type, dataset_id))
        flash(f'Vectorisation of "{dataset_name}" started. Follow it in Tasks.', 'info')

    return render_template("home.html", page_name=name, form=vector_form)

def back_task_create_vect(model, data_path, dataset_name, save_to, data_type, dataset_id):
    vectors_ = controller_.create_Vectorisation(data_path=data_path,
                                                out_path=save_locally_folder,
                                                data_type=data_type,
                                                embedder=model['embedder'],
                                                model_path=model['model_path'],
                                                vec_dim=model['vec_dim'],
                                                show_progress=False)

    filename, vec_dim = controller_.store_vectors(save_to=save_to, folder=save_locally_folder, vectors_dict=vectors_, dataset_name=dataset_name)

    conn = get_db_connection()
    insert_vector_location(conn=conn, store_into=save_to, filename=filename, data_type=data_type, dataset_id=dataset_id,
                           embedder=model['embedder'], model_path=model['model_path'], vec_dim=vec_dim)
    conn.close()


@app.route("/display_vecs", methods=['GET', 'POST'])
def display_vec_dataset():
    name = "display_vectors"
    conn = get_db_connection()

    data_vectors = get_all_vector_datasets(conn=conn)
    avail_data_vectorisation = vector_display_names(data_vectors)
    avail_data_vectors_full = {str(id_): vec_name for id_, vec_name, store_id, store_name, dataset_id in data_vectors}

    displ_vec = DisplayVectorEmbeddingRepr(vec_emb_reprs=avail_data_vectorisation)
    conn.close()
    if request.method == 'POST' and str(displ_vec.select_ver.data) in avail_data_vectors_full:
        selected_data_vectors = avail_data_vectors_full[str(displ_vec.select_ver.data)]
        conn = get_db_connection()
        stored_into, data_type = find_database_store(conn=conn, vector_name=selected_data_vectors)
        conn.close()
        try:
            vectors_ = controller_.get_vectors_from_vectorDB(database_=stored_into, collection_name=selected_data_vectors)
            plot_url = create_vec_repr_plot(vectors_=vectors_)
        except Exception as err:
            flash(f'Could not load the vectors: {err}', 'danger')
            return render_template("home.html", page_name=name, form=displ_vec, display_vecs='False')
        return render_template("home.html", page_name=name, form=displ_vec, display_vecs='True', plot_url=plot_url)
    return render_template("home.html", page_name=name, form=displ_vec, display_vecs='False')

@app.route('/sim_search', methods=['GET', 'POST'])
def sim_search():
    name = "similarity_search"
    sim_search_methods = {idx: label for idx, (label, _) in SIM_SEARCH_METHODS.items()}

    conn = get_db_connection()
    data_vectors = get_all_vector_datasets(conn=conn)
    avail_data_vectorisation = vector_display_names(data_vectors)
    avail_data_vectors_full = {str(id_): vec_name for id_, vec_name, store_id, store_name, dataset_id in data_vectors}
    conn.close()
    sim_search_form = SimilaritySearchForm(avail_sim_search=sim_search_methods, avail_vec_emb_reprs=avail_data_vectorisation)

    if request.method == 'POST':
        method = SIM_SEARCH_METHODS.get(str(sim_search_form.select_sim_search.data))
        vectors_id = str(sim_search_form.select_avail_vec_repr.data)
        if method is None or vectors_id not in avail_data_vectors_full or sim_search_form.select_dataset.data is None:
            flash('Fill in every field.', 'danger')
            return render_template("home.html", page_name=name, form=sim_search_form)

        selected_data_vectors = avail_data_vectors_full[vectors_id]
        sim_search_data_name = sim_search_form.sim_search_name.data.strip()
        data_path = sim_search_form.pred_data_dir_path.data.strip()
        data_selection = float(sim_search_form.select_dataset.data)

        conn = get_db_connection()
        vector_info = get_vector_info(conn=conn, vector_name=selected_data_vectors)
        conn.close()

        if vector_info is None:
            flash(f'No vectors named {selected_data_vectors} in the database.', 'danger')
        elif not check_dataset_file(filepath=data_path):
            flash(f'Path not found: {resolve_path(data_path)}', 'danger')
        elif method[1] == 'vectordb' and vector_info['store'].lower() != 'qdrant':
            flash('Vector DB Search needs vectors stored in Qdrant; choose another method for local vectors.', 'danger')
        else:
            start_task(f"Similarity search ({method[0]}) for {data_path} on {DATA_TYPES.get(vector_info['data_type'])} vectors {selected_data_vectors}",
                       lambda: back_task_get_vectors(data_path, sim_search_data_name, method[1], selected_data_vectors, vector_info, data_selection))
            flash(f'Similarity search "{sim_search_data_name}" started. Follow it in Tasks.', 'info')

    return render_template("home.html", page_name=name, form=sim_search_form)

def back_task_get_vectors(data_path, sim_search_data_name, sim_search_technique, data_vectors, vector_info, select_percent):
    sim_search_vectors = controller_.similarity_search(data_path=data_path,
                                                       out_path=save_sim_search_locally,
                                                       vectors=data_vectors,
                                                       vector_info=vector_info,
                                                       data_selection=sim_search_technique,
                                                       data_ratio=select_percent)

    stored_filepath = controller_.store_similarity_search_locally(out_path=save_sim_search_locally,
                                                                  name=sim_search_data_name,
                                                                  vectors_dict=sim_search_vectors)

    conn = get_db_connection()
    vectors_idx, _ = get_vectors_idx_from_name(conn, data_vectors)
    insert_sim_search_vectors(conn=conn, vector_name=sim_search_data_name, dataset_path=data_path, saved_datapath=stored_filepath, vectors_idx=vectors_idx)
    conn.close()

@app.route("/display_sim_vecs", methods=['GET', 'POST'])
def display_sim_vec_dataset():
    name = "display_sim_vectors"

    conn = get_db_connection()

    avail_sim_search_vec = get_similarity_search(conn=conn) or []

    data_vectors = get_all_vector_datasets(conn=conn)
    avail_data_vectors_full = {str(id_): vec_name for id_, vec_name, store_id, store_name, dataset_id in data_vectors}

    avil_vec_sim = {str(idx): name for idx, name, pred_path, stored_sim_search, vec_id in avail_sim_search_vec}
    avail_vec_sim_vectors_id = {str(idx): vec_id for idx, name, pred_path, stored_sim_search, vec_id in avail_sim_search_vec}

    displ_vec = DisplayVectorEmbeddingRepr(vec_emb_reprs=avil_vec_sim)
    conn.close()
    if request.method == 'POST' and str(displ_vec.select_ver.data) in avil_vec_sim:
        conn = get_db_connection()
        selected_vec = avil_vec_sim[str(displ_vec.select_ver.data)]
        vector_id = avail_vec_sim_vectors_id[str(displ_vec.select_ver.data)]

        prediction_dataset_path, sim_search_vec = get_sim_search_by_name(conn=conn, sim_search=selected_vec)

        selected_data_vectors = avail_data_vectors_full[str(vector_id)]
        stored_into, data_type = find_database_store(conn=conn, vector_name=selected_data_vectors)
        conn.close()
        try:
            vectors_ = controller_.get_vectors_from_vectorDB(database_=stored_into, collection_name=selected_data_vectors)
            vectors_similarity_search = read_vectors(sim_search_vec)
            plot_url = create_sim_search_vector_plot(vectors_, vectors_similarity_search)
        except Exception as err:
            flash(f'Could not load the similarity search: {err}', 'danger')
            return render_template("home.html", page_name=name, form=displ_vec)
        return render_template("home.html", page_name=name, form=displ_vec, display_vecs='True', plot_url=plot_url)
    return render_template("home.html", page_name=name, form=displ_vec)

@app.route("/op_model", methods=['GET', 'POST'])
def operator_modelling():
    name = 'operator_modelling'
    operator_choices = get_operator_choices()
    operator_labels = {key: label for options in operator_choices.values() for key, label in options}

    conn = get_db_connection()
    avail_sim_search_vec = get_similarity_search(conn=conn) or []
    conn.close()

    avil_vec_sim = {str(idx): name for idx, name, pred_path, stored_sim_search, vec_idx in avail_sim_search_vec}

    form = OperatorModellingForm(operator_algorithms=operator_choices, vectors=avil_vec_sim)
    if request.method == 'POST':
        sim_search_idx = str(form.select_avail_vec_repr.data)
        operator_key = form.select_avail_oper.data
        if sim_search_idx not in avil_vec_sim or operator_key not in operator_labels or not form.operator_name.data or not form.operator_dir.data:
            flash('Fill in the name, the output directory, the operator and the similarity search.', 'danger')
            return render_template("home.html", page_name=name, form=form)

        selected_vec_sim = avil_vec_sim[sim_search_idx]
        operator_type, selected_operator = operator_key.split(':', 1)
        selected_op_name = operator_labels[operator_key]

        conn = get_db_connection()
        sim_search_dataset = get_dataset_id_from_sim_search(conn=conn, sim_search=selected_vec_sim)
        conn.close()

        target_labels = (form.target_labels.data or '').strip() or None
        if sim_search_dataset is None:
            flash(f'The dataset behind "{selected_vec_sim}" is no longer in the database.', 'danger')
        elif int(operator_type) != sim_search_dataset[1]:
            flash(f'"{selected_op_name}" is a {DATA_TYPES[int(operator_type)]} operator but "{selected_vec_sim}" holds {DATA_TYPES.get(sim_search_dataset[1])} data.', 'danger')
        elif target_labels and not check_dataset_file(target_labels):
            flash(f'Target labels file not found: {resolve_path(target_labels)}', 'danger')
        else:
            operator_name = form.operator_name.data.strip()
            operator_outpath = form.operator_dir.data.strip()
            start_task(f"Operator Modelling for {selected_operator} in similarity search vectors {selected_vec_sim}",
                       lambda: back_task_operator_modelling(operator_name, operator_outpath, selected_operator, selected_vec_sim, sim_search_idx, selected_op_name, target_labels))
            flash(f'Operator modelling "{operator_name}" started. Follow it in Tasks.', 'info')
    return render_template("home.html", page_name=name, form=form)

def back_task_operator_modelling(oper_model_name, out_path, operator, sim_search, sim_search_idx, selected_op_name, target_labels=None):
    conn = get_db_connection()
    prediction_dataset_path, sim_search_vec = get_sim_search_by_name(conn=conn,
                                                                     sim_search=sim_search)

    datapath, data_type = get_dataset_id_from_sim_search(conn=conn, sim_search=sim_search)
    conn.close()

    op_Modelling = Operator_Modelling(operator_name=oper_model_name,
                                      operator_outpath=out_path,
                                      operator=operator,
                                      sim_search_vectors=sim_search_vec,
                                      prediction_data_path=prediction_dataset_path,
                                      directory_datapath=datapath,
                                      data_type=data_type,
                                      target_labels=target_labels)

    path_labels = op_Modelling.create_labels()
    start_time = time.time()

    results_, out_file = op_Modelling.model_operator(path_labels=path_labels)

    print('Finish operator modelling')
    exec_time = time.time() - start_time
    results_['Exec time'] = exec_time
    df_res = pd.DataFrame([results_])
    df_res.to_csv(out_file, index=False)

    conn = get_db_connection()
    add_operator_model_results(conn=conn, name=oper_model_name, output_path=out_path, sim_search_id=sim_search_idx, metrics=out_file, operator=selected_op_name)
    conn.close()

@app.route("/display_op", methods=['GET', 'POST'])
def operator_modelling_display():
    conn = get_db_connection()
    name = 'operator_modelling_display'
    avail_op_models = get_all_operators(conn=conn) or []
    avail_op_models_dict = {str(idx): name for idx, name in avail_op_models}

    form = OperatorModellingDisplayForm(avail_op_models_select=avail_op_models_dict)

    if request.method == 'POST' and str(form.avail_op_models.data) in avail_op_models_dict:
        selected_operator = avail_op_models_dict[str(form.avail_op_models.data)]
        metric_path, sim_search_idx, operator_name = get_metric_path(conn, selected_operator)

        pred_datapath, sim_search_name = get_sim_search_name_by_id(conn=conn, idx=sim_search_idx)
        description = f'Operator Modelling for similarity search {sim_search_name} to model operator {operator_name}'
        df = pd.read_csv(metric_path, index_col=False).iloc[0]
        results = {key: '' if pd.isna(df.get(column)) else str(df.get(column))
                   for key, column in (('r2', 'r^2'), ('rmse', 'RMSE'), ('mae', 'MAE'), ('mad', 'MAD'), ('exec_time', 'Exec time'))}
        table_form = updatedOpModelForm()
        table_form.process(data={'field_list': [results]})
        conn.close()
        return render_template("home.html", page_name=name, form=form, task_form=table_form, desc_p=description, display_plot='True')
    conn.close()
    return render_template("home.html", page_name=name, form=form)


if __name__ == "__main__":
    if not os.path.exists(save_locally_folder):
        os.makedirs(save_locally_folder)
    if not os.path.exists(save_sim_search_locally):
        os.makedirs(save_sim_search_locally)
    app.run(debug=True, host="0.0.0.0", port=5000)
