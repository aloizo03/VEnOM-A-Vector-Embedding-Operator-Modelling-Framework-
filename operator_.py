from utils_.create_operator import Create_Operator
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dict-input", type=str, help="Input dataset path f.e data/path/dir")
    parser.add_argument("-oi", "--operator-input", default=None, type=str, help="Directory to input operator operator/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument("-t", "--threshold", type=float, default=0.7, help="Threshold for the Operator F")
    parser.add_argument('-o', '--operator', type=str, default='svm_sgd',
                        help='Type of data selection (sim, distance, random)')
    # parser.add_argument('-o', '--use-vectors', type=str, default='svm_sgd',
    #                     help='Type of data selection (sim, distance, random)')
    parser.add_argument('-r', '--repetitions', type=int, default=2, help='The amount of repetition of the experiment')
        

    args = parser.parse_args()

    data_input_dict_path = args.dict_input
    input_operator_dir = args.operator_input
    out_path = args.out_path
    threshold = args.threshold
    operator_name = args.operator

    operator_class = Create_Operator(out_path=out_path, operator_dir=input_operator_dir)
    operator_class.create_operator(operator_name=operator_name,
                                   most_relevant_data_path=data_input_dict_path,
                                   threshold=threshold, 
                                   repetitions=1)


if __name__ == '__main__':
    main()
