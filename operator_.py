from utils_.create_operator import Create_Operator
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dict-input", type=str, help="Input dataset path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-o', '--operator', type=str, default='svm_sgd', help='Type of data selection (sim, distance, random)')

    args = parser.parse_args() 

    data_input_dict_path = args.dict_input
    out_path = args.out_path
    operator_name = args.operator

    operator_class = Create_Operator(out_path=out_path)
    operator_class.create_operator(operator_name=operator_name,
                                   most_relevant_data_path=data_input_dict_path)

if __name__ == '__main__':
    main()