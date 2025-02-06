from utils_.sr_operator import SR_Operator_Creation
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--vec-input", type=str, help="Datasets vectorisation directory")
    parser.add_argument("-sr", "--sampling-ratio", type=float,
                        help="The percentage of the random datasets that will be used")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-o', '--operator', type=str, default='svm_sgd', help='Type of the operator')

    args = parser.parse_args()

    vec_input_path = args.vec_input
    sr = args.sampling_ratio
    out_path = args.out_path
    operator_name = args.operator

    sr_operator_creation = SR_Operator_Creation(vec_input_path=vec_input_path,
                                                out_path=out_path,
                                                sr=sr)
    sr_operator_creation.create_operator(operator_name=operator_name)


if __name__ == '__main__':
    main()
