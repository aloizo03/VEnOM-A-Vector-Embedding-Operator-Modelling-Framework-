from flask import Flask, render_template, request, redirect
from server_utils.utils import create_vec_repr_plot, read_vectors, create_sim_search_vector_plot, create_res_plot
from utils_flask import *

app = Flask(__name__)
app.config['SECRET_KEY'] = "IAMDEAD"

# @app.route("/onclick_upload_data", methods=['GET', 'POST'])
# def upload_data():
#     name = ''
#     dir_filepath = request.form.get("filepath")
#     processed_text = dir_filepath.upper()
#     print(processed_text)
#     print('------------------')
#     return render_template("home.html", page_name=name) 

@app.route("/")
def home():
    name = "null"
    return render_template("home.html", page_name=name)

@app.route("/onclick_load_data", methods=['GET', 'POST'])
def load_data():
    page_name = "load_data"
    name = None
    dataset_name = None
    load_data_form = LoadDataForm()
    if load_data_form.validate_on_submit():
        name = load_data_form.Dataset_DIR.data
        dataset_name = load_data_form.Dataset_Name.data
        load_data_form.Dataset_DIR.data = ''
        load_data_form.Dataset_Name.data = ''

    return render_template("home.html",
                           page_name=page_name,
                           dataset_name=dataset_name,
                           name=name,
                           form=load_data_form)

@app.route("/onclick_vec", methods=['GET', 'POST'])
def vec_dataset():
    name = "vectorisation"
    dataset_names = {'1': 'HPC', '2': 'Weather'}
    # dataset_names = ['data_A', 'data_B']
    # vec_sizes = ['40', '100', '200', '300']
    vec_sizes = {1: '40', 2: '100', 3: '200', 4: '300'}

    save_into = {'1': 'Localy', '2': 'Qdrant', '3': 'Milvus'}
    
    vector_form = VectorisationForm(vector_sizes=vec_sizes, avail_datasets=dataset_names, save_dbs=save_into)

    if request.method == 'POST':
        print('Create Vectorisation')
        print(f"data: {vector_form.select_dataset.data}, vec_sizes: {vector_form.select_vec_size.data}")
    return render_template("home.html", page_name=name, form=vector_form)

@app.route("/display_vecs", methods=['GET', 'POST'])
def display_vec_dataset():
    name = "display_vectors"
    data_vectorisation = {'1': 'data/hpc_vec.pkl', '2': 'data/hpc_weather.pkl'}
    displ_vec = DisplayVectorEmbeddingRepr(vec_emb_reprs=data_vectorisation)
    vec_dir = '/opt/dlami/nvme/Vector_Embedding/Results_Experiments_2/HPC_2K/200V/vectors_1731669493.3926575.pickle'
    if request.method == 'POST':
        vectors_ = read_vectors(dir_path=vec_dir)
        plot_url = create_vec_repr_plot(vectors_=vectors_)
        return render_template("home.html", page_name=name, form=displ_vec, display_vecs='True', plot_url=plot_url)
    return render_template("home.html", page_name=name, form=displ_vec, display_vecs='False')

@app.route('/sim_search', methods=['GET', 'POST'])
def sim_search():
    name = "similarity_search"
    sim_search_methods = {'1': 'Vector DB Search', '2': 'Cosine Similarity', '3': 'K-Means', '4': 'Euclidean Distance',}
    avail_data_vectorisation = {'1': 'HPC Vectors', '2': 'Weather Vectors'}
    sim_search_form = SimilaritySearchForm(avail_sim_search=sim_search_methods, avail_vec_emb_reprs=avail_data_vectorisation)
    if request.method == 'POST':
        print('create_sim_search')
        sim_search_form = SimilaritySearchForm(avail_sim_search=sim_search_methods, avail_vec_emb_reprs=avail_data_vectorisation)
        return render_template("home.html", page_name=name, form=sim_search_form)
    return render_template("home.html", page_name=name, form=sim_search_form)

@app.route("/display_sim_vecs", methods=['GET', 'POST'])
def display_sim_vec_dataset():
    name = "display_sim_vectors"
    data_vectorisation = {'1': 'data/vectors_hpc_SimSearch.pkl', '2': 'data/vectors_k-means.pkl'}
    # data_vectorisation = {'1': 'data/hpc_vec.pkl', '2': 'data/hpc_weather.pkl'}
    displ_vec = DisplayVectorEmbeddingRepr(vec_emb_reprs=data_vectorisation)
    vec_dir = "/opt/dlami/nvme/Vector_Embedding/Results_Experiments_2/HPC_2K/200V/vectors_1731669493.3926575.pickle"
    vec_ss_dir = '/opt/dlami/nvme/Vector_Embedding/Results_Experiments_2/HPC_2K/200V/Dataset_Selection/rel_data_cosine_1733225609.908662.pickle'
    if request.method == 'POST':
        vectors = read_vectors(vec_dir)
        vectors_similarity_search = read_vectors(vec_ss_dir)
        plot_url = create_sim_search_vector_plot(vectors, vectors_similarity_search)
        return render_template("home.html", page_name=name, form=displ_vec, display_vecs='True', plot_url=plot_url)
    return render_template("home.html", page_name=name, form=displ_vec)

@app.route("/op_model", methods=['GET', 'POST'])
def operator_modelling():
    name = 'operator_modelling'
    avail_operators = {'1': 'SVM', 
                       '2':'MLP', 
                       '3': 'Linear Regression', 
                       '4': 'Arima', 
                       '5': 'Holt-Winter',
                       '6': 'Logistic Regression', 
                       '7': 'DBSCAN'}
    avil_vec_sim = {'1': 'HPC_SimSearch',
                    '2': 'HPC_VectorDB'}
    form = OperatorModellingForm(operator_algorithms=avail_operators, vectors=avil_vec_sim)
    if request.method == 'POST':
        model_op = True
        form = OperatorModellingForm(operator_algorithms=avail_operators, vectors=avil_vec_sim)
        return render_template("home.html", page_name=name, form=form)
    return render_template("home.html", page_name=name, form=form)

@app.route("/display_op", methods=['GET', 'POST'])
def operator_modelling_display():
    name = 'operator_modelling_display'
    avail_op_models = {'1': 'LR_HPC',
                       '2': 'ARIMA_HPC',
                       '3': 'MLP_Regression_HPC'}
    form = OperatorModellingDisplayForm(avail_op_models_select=avail_op_models)

    if request.method == 'POST':
        NRMSE = 0.023
        rmse_loss = 8.5
        MAE_loss = 5.24
        MaPE = 3.2
        SpeedUp = 2.8
        loss_res = [NRMSE, rmse_loss, MAE_loss, MaPE, SpeedUp]
        labels = ['NRMSE Loss', 'RMSE Loss', 'MAE Loss', 'MaPE Loss', 'Speedup']
        plot_url = create_res_plot(res_loss=loss_res, labels=labels)
        return render_template("home.html", page_name=name, form=form, plot_url=plot_url, display_plot='True')

    return render_template("home.html", page_name=name, form=form)


if __name__ == "__main__":
    app.run(debug=True)