import argparse
from utils_.select_datasets import Data_selection
from utils_.utils import check_path
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--model-weight', type=str, default=None,
                        help="Model Weights for the Vector Embeddings")
    parser.add_argument("-i", "--data-input", type=str, help="Input dataset path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-v', '--vectors', type=str, help='total batch size')
    parser.add_argument('-s', '--data-selection', type=str, default='distance',
                        help='Type of data selection (sim, distance, random)')

    args = parser.parse_args()

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    vectors = args.vectors

    data_selection = args.data_selection

    clustering = Data_selection(data_path=data_input_path,
                                out_path=out_path,
                                vectors_path=vectors,
                                model_path=model_path)
    clustering.find_relevant_datasets(type_of_selection=data_selection)


if __name__ == '__main__':
    main()
