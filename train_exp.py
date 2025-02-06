import argparse
from utils_.exp_accuracy import EXP_Accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-input", type=str, help="Datasets vectorisation directory")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-o', '--operator', type=str, default='svm_sgd', help='Type of the operator')
    parser.add_argument('-q', '--query',  type=str, default='all', help='Type of the operator')
    args = parser.parse_args()

    data_input = args.data_input
    out_path = args.out_path
    operator_name = args.operator
    query = args.query

    operator_creation = EXP_Accuracy(input_data_dir=data_input, 
                                     out_path_dir=out_path, 
                                     operator_name=operator_name)

    operator_creation.create_op(query=query)


if __name__ == '__main__':
    main()