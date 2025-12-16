import argparse
from utils_.exp_accuracy import EXP_Accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-input", type=str, help="Datasets vectorisation directory")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-o', '--operator', type=str, default='svm_sgd', help='Type of the operator')
    parser.add_argument('-q', '--query',  type=str, default='all', help='Type of the operator')
    parser.add_argument("-dt", "--data-type", type=str, default='tabular', help='Available Data Type:\n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    parser.add_argument("-fn", "--file-name", type=str, default='scores', help='The name of the output file with the labels')



    args = parser.parse_args()

    data_input = args.data_input
    out_path = args.out_path
    operator_name = args.operator
    query = args.query
    output_filename = args.file_name

    data_type_str = args.data_type
    if data_type_str.lower() == 'tabular':
        data_type = 1
    elif data_type_str.lower() == 'graph':
        data_type = 2
    elif data_type_str.lower() == 'image':
        data_type = 3
    else:
        AssertionError('Wrong Data type available data types: \n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    
    operator_creation = EXP_Accuracy(input_data_dir=data_input, 
                                     out_path_dir=out_path, 
                                     operator_name=operator_name, 
                                     data_type=data_type,
                                     out_filename=output_filename)

    operator_creation.create_op(query=query)


if __name__ == '__main__':
    main()