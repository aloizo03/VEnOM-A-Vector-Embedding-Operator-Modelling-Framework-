from utils_.create_operator import Create_Operator
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dict-input", type=str, help="Input dataset path f.e data/path/dir")
    parser.add_argument("-oi", "--operator-input", default=None, type=str, help="Directory to input operator operator/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-o', '--operator', type=str, default='svm_sgd',
                        help='Type of data selection (sim, distance, random)')
    parser.add_argument('-r', '--repetitions', type=int, default=2, help='The amount of repetition of the experiment')
    parser.add_argument("-dt", "--data-type", type=str, default='tabular', help='Available Data Type:\n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    parser.add_argument("-tl", "--target-labels", type=str, default=None, help='Labels to the target dataset/datasets')

    args = parser.parse_args()

    data_input_dict_path = args.dict_input
    input_operator_dir = args.operator_input
    out_path = args.out_path
    operator_name = args.operator
    repetitions = args.repetitions
    data_type_str = args.data_type
    target_labels = args.target_labels

    if data_type_str.lower() == 'tabular':
        data_type = 1
    elif data_type_str.lower() == 'graph':
        data_type = 2
    elif data_type_str.lower() == 'image':
        data_type = 3
    else:
        AssertionError('Wrong Data type available data types: \n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    

    # operator_class = Create_Operator(out_path=out_path, operator_dir=input_operator_dir)
    operator_class = Create_Operator(out_path=out_path, operator_dir=None)
    operator_class.create_operator(operator_name=operator_name,
                                   most_relevant_data_path=data_input_dict_path,
                                   repetitions=repetitions,
                                   labels_dict=input_operator_dir, 
                                   target_labels=target_labels,
                                   data_type=data_type)


if __name__ == '__main__':
    main()
