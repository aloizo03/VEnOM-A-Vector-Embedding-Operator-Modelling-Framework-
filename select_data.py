import argparse

from utils_.select_datasets import Data_selection
from utils_.utils import check_path, get_vec_DB_collections
import os
import sys
from server.server_utils.qdrant_controller import qdrant_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--model-weight', type=str, default=None,
                        help="Model Weights for the Vector Embeddings")
    parser.add_argument("-i", "--data-input", type=str, help="Input dataset path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-v', '--vectors', type=str, help='vectors path or VectorDB collection name')
    parser.add_argument('-s', '--data-selection', type=str, default='distance',
                        help='Type of data selection (sim, distance, random)')
    parser.add_argument('-r', '--data-ratio', type=float, default=0.3,
                        help='Percentage of who many datasets will be selectes which are relevant to data input .csv file')
    parser.add_argument("-dt", "--data-type", type=str, default='tabular',
                        help='Available Data Type:\n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    parser.add_argument("-vb", "--vector-db", action="store_true", help="Use the vector from the vector database")
    parser.add_argument('-vs', '--vector-size', type=int, default=100, help='The vector embedding token dimension')
    parser.add_argument('-sDB', '--show-DBs', action="store_true",
                        help="Show all the available vector Dabase colections with their description")
    parser.add_argument("-tm", "--tab-model", type=str, default='num2vec', choices=['num2vec', 'tabfm', 'dataset2vec'],
                        help="Model used to vectorise the input tabular dataset (must match the model used to create the stored vectors):\n\t-num2vec: the trained Num2Vec model (needs --model-weight)\n\t-tabfm: Google's TabFM foundation model (google/tabfm-1.0.0-pytorch); --model-weight is optional and may point to a local TabFM checkpoint dir\n\t-dataset2vec: the pretrained Dataset2Vec meta-feature model (hadijomaa/dataset2vec, 32-d vectors); --model-weight is optional and may point to a checkpoint dir")
    parser.add_argument("--tabfm-task", type=str, default='classification', choices=['classification', 'regression'],
                        help='Which TabFM backbone weights to use for the embeddings')
    parser.add_argument("--tabfm-max-rows", type=int, default=512,
                        help='Maximum number of rows per dataset sampled as TabFM context')
    parser.add_argument("--d2v-split", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Which pretrained Dataset2Vec checkpoint split to use')
    parser.add_argument("--d2v-batches", type=int, default=10,
                        help='Number of sampled batches averaged into each Dataset2Vec vector')
    args = parser.parse_args()

    show_Vec_DB_collections = args.show_DBs
    if show_Vec_DB_collections:
        qdrant_ctrl = qdrant_controller()
        all_collections = qdrant_ctrl.get_all_colections()
        print('Available Collections: ')
        for collection in all_collections.collections:
            info = qdrant_ctrl.get_collection_info(collection.name)
            print(f"\n=== Collection: {collection.name} ===")
            print(f"Vector size: {info.config.params.vectors.size}")
            print(f"Distance: {info.config.params.vectors.distance}")
            print(f"Shard number: {info.config.params.shard_number}")
        sys.exit()

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    vectors = args.vectors
    data_ratio_selection = args.data_ratio
    d_token = args.vector_size
    data_selection = args.data_selection
    data_type_str = args.data_type
    use_vector_DB = args.vector_db

    if data_type_str.lower() == 'tabular':
        data_type = 1
    elif data_type_str.lower() == 'graph':
        data_type = 2
    elif data_type_str.lower() == 'image':
        data_type = 3
    else:
        AssertionError(
            'Wrong Data type available data types: \n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')

    clustering = Data_selection(data_path=data_input_path,
                                out_path=out_path,
                                vectors_path=vectors,
                                model_path=model_path,
                                data_type=data_type,
                                use_vector_DB=use_vector_DB,
                                d_token=d_token,
                                tab_model=args.tab_model,
                                tabfm_task=args.tabfm_task,
                                tabfm_max_rows=args.tabfm_max_rows,
                                d2v_split=args.d2v_split,
                                d2v_batches=args.d2v_batches)

    clustering.find_relevant_datasets(type_of_selection=data_selection,
                                      data_ratio_selection=data_ratio_selection,
                                      plot_representation=False)


if __name__ == '__main__':
    main()


