import argparse

from flask.cli import F
from utils_.select_datasets import Data_selection
from utils_.utils import check_path
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--model-weight', type=str, default=None, help="Model Weights for the Vector Embeddings")
    parser.add_argument("-i", "--data-input", type=str, help="Input dataset path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-v', '--vectors', type=str, help='total batch size')
    parser.add_argument('-s', '--data-selection', type=str, default='distance', help='Type of data selection (sim, distance, random)')
    parser.add_argument('-r', '--data-ratio', type=float, default=0.3, help='Percentage of who many datasets will be selectes which are relevant to data input .csv file')
    parser.add_argument("-g", "--graph", action="store_true")

    args = parser.parse_args()

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    vectors = args.vectors
    data_ratio_selection = args.data_ratio

    data_selection = args.data_selection
    
    is_graph_dataset = args.graph
    if is_graph_dataset:
        data_type = 2
    else:
        data_type = 1

    clustering = Data_selection(data_path=data_input_path,
                              out_path=out_path, 
                              vectors_path=vectors, 
                              model_path=model_path,
                              data_type=data_type,)
    
    clustering.find_relevant_datasets(type_of_selection=data_selection,
                                      data_ratio_selection=data_ratio_selection, 
                                      graph_dataset=is_graph_dataset, 
                                      plot_representation=False)

if __name__ == '__main__':
    main()


