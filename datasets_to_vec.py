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
    parser.add_argument("-g", "--graph", action="store_true")

    args = parser.parse_args()

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    batch_size = args.batch_size

    is_graph_dataset = args.graph
    if is_graph_dataset:
        data_type = 2
    else:
        data_type = 1

    clustering = Vectorise(data_path=data_input_path,
                            model_path=model_path,
                            out_path=out_path,
                            data_type=data_type,)

    clustering.compute_vectors(batch_size=batch_size)

if __name__ == '__main__':
    main()


