import argparse
from utils_.train_sr_op import Train_Operator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vectors-input", type=str, help="Datasets vectorisation directory")
    parser.add_argument("-p", "--pred-input", type=str, help="Prediction dataset/datasets directory")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-ckpt', '--model-ckpt', type=str, default='svm_sgd', help='Type of the operator')
    parser.add_argument('-m', '--model', type=str, default='perceptron', help='Type of the operator')
    parser.add_argument('-sr', '--sampling-ratio', type=float, default=0.1, help='Type of the operator')
    parser.add_argument('-r', '--repitition', type=int, default=5, help='Type of the operator')
    parser.add_argument('-q', '--query',  type=str, default='all', help='Type of the operator')
    parser.add_argument('-f', '--score-file',  type=str, default=None, help='Type of the operator')
    parser.add_argument('-o', '--op-name',  type=str, default=None, help='The name of the operator that we want to build the ML Model')
    args = parser.parse_args()

    vectors_path = args.vectors_input
    pred_path = args.pred_input
    
    out_path = args.out_path
    ckpt_model_path = args.model_ckpt

    model = args.model
    sr = args.sampling_ratio
    repititions = args.repitition
    query = args.query
    score_file =args.score_file
    op_name = args.op_name
    # vec_input_path, pred_path, out_path, sr, model_path
    
    train_operator = Train_Operator(model_path=ckpt_model_path,
                                    pred_path=pred_path,
                                    vec_input_path=vectors_path,
                                    out_path=out_path,
                                    sr=sr)
    train_operator.train_operator(ML_model_name=model, 
                                  query=query, 
                                  scores_file=score_file, 
                                  repititions=repititions,
                                  op_name=op_name)

if __name__ == '__main__':
    main()

