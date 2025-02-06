from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import matplotlib.colors as colors
import cv2
from PIL import Image

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
    filepath = '/home/ubuntu/Projects/Vector Embedding/server/static/images/plot_representation_2.png'
    with open(filepath, "rb") as fh:
        img = BytesIO(fh.read())
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
        # colors_lst = list(colors.TABLEAU_COLORS.keys())
        # plt.set_loglevel('info')

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # for dataset, rel_datasets in sim_search_datasets.items():
        #     labels_ = ['Relevant Dataset' if i in rel_datasets else 'Data Lake' for i in dict_vectors.keys()]
        #     vec_np = np.array(list(dict_vectors.values()))


        # for new_vector_, data_idx in zip(new_vectors_, new_dataset_names):
        #     data_name = dataset_builder.get_dataset_name(data_idx)
        #     most_rel = most_relevant[data_name]
        #     vec_np = np.array(list(dict_vectors.values()))
            
        #     labels_ = ['Relevant Dataset' if i in most_rel else 'Data Lake' for i in dict_vectors.keys()]
        #     # colors_ = [colors_lst[data_idx] if i in most_rel else 'blueviolet' for i in dict_vectors.keys()]
            
            
        #     vec_concat_np = np.concatenate((vec_np, [new_vector_]), axis=0)

        #     pca = PCA(n_components=3)
        #     result = pca.fit_transform(vec_concat_np)
        #     unique = list(set(labels_))
        #     for i, u in enumerate(unique):

        #         vec_ = [result[j, :] for j in range(len(labels_)) if labels_[j] == u]
        #         np_arr_vec = np.asarray(vec_, dtype=np.float64)
           
        #         ax.scatter(np_arr_vec[:, 0], np_arr_vec[:, 1], np_arr_vec[:, 2], c=colors_lst[i], label=u)
        #     ax.scatter(result[vec_np.shape[0]:, 0], result[vec_np.shape[0]:, 1], result[vec_np.shape[0]:, 2], c='purple', label='New Data')
                
        # ax.legend()
        # ax.grid(True)

        # ax.set(xlabel=(r' $x$'), ylabel=(r'$y$'), zlabel=(r'$z$'))
        # plt.savefig(os.path.join(self.output_path, 'plot_representation_2.png'), dpi=100)


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


# def find_each_dataset_vector(datasets_, vectors):
#     out_dict = {}
#     for pred_data, dataset_dirs in datasets_.items():
#           out_dict[pred_data] = []
#           for dir in dataset_dirs:
#             out_dict[pred_data].append(vectors[dir])
#     return out_dict
