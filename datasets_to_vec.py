import argparse
from utils_.vectors import Vectorise
from utils_.utils import check_path
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--model-weight', type=str, default=None, help="Model Weights for the Vector Embeddings")
    parser.add_argument("-i", "--data-input", type=str, help="Input datasets path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-bz', '--batch-size', type=int, default=8, help='total batch size')

    args = parser.parse_args()

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    batch_size = args.batch_size

    clustering = Vectorise(data_path=data_input_path,
                            model_path=model_path,
                            out_path=out_path)

    clustering.compute_vectors(batch_size=batch_size)

if __name__ == '__main__':
    main()

