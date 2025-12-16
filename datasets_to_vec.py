import argparse
from utils_.vectors import Vectorise
from utils_.utils import check_path
import os

# python datasets_to_vec.py --model-weight "/opt/dlami/nvme/Vector Embedding/results/Opt_Adam_HPC_in_all_dataset_tuples_e200_v300/weights/best.pt" --data-input "data/train_data_big_dataset" --out-path "results/test_big_datasets/" --batch-size 8
# python select_data.py --model-weight "/opt/dlami/nvme/Vector Embedding/results/Opt_Adam_HPC_in_all_dataset_e200_v300/weights/best.pt" --data-input "data/test_big_data/dataset_0_286.csv" --out-path "results/test_big_datasets" --vectors "results/test_big_datasets/vectors_1723" --data-selection 'distance'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--model-weight', type=str, default=None, help="Model Weights for the Vector Embeddings")
    parser.add_argument("-i", "--data-input", type=str, help="Input datasets path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-bz', '--batch-size', type=int, default=1, help='total batch size')
    parser.add_argument('-cn', '--collection-name', type=str, default='collection_name', help='Set the vectors collection name')
    parser.add_argument('-vs', '--vector-size', type=int, default=100, help='The vector embedding token dimension')
    parser.add_argument("-dt", "--data-type", type=str, default='tabular', help='Available Data Type:\n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    parser.add_argument("-s", "--save-to", type=str, default='local', help="Where do you want to save the vector embedding representation:\n\t-s local: save to local repository output\n\t-s vectorDB: save to Qdrant Vector Database")

    args = parser.parse_args()

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    batch_size = args.batch_size
    save_to = args.save_to
    data_type_str = args.data_type
    d_token = args.vector_size
    collection_name = args.collection_name
    if data_type_str.lower() == 'tabular':
        data_type = 1
    elif data_type_str.lower() == 'graph':
        data_type = 2
    elif data_type_str.lower() == 'image':
        data_type = 3
    else:
        AssertionError('Wrong Data type available data types: \n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    
    clustering = Vectorise(data_path=data_input_path,
                            model_path=model_path,
                            out_path=out_path,
                            data_type=data_type,
                            save_to=save_to, 
                            d_token=d_token, 
                            collection_name=collection_name)

    clustering.compute_vectors(batch_size=batch_size)

if __name__ == '__main__':
    main()


