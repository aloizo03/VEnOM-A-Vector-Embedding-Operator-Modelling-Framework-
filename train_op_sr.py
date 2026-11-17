import argparse
from utils_.train_sr_op import Train_Operator
from utils_.utils import set_global_seed
import sys
from server.server_utils.qdrant_controller import qdrant_controller

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vectors-input", type=str, help="Datasets vectorisation directory")
    parser.add_argument("-p", "--pred-input", type=str, help="Prediction dataset/datasets directory")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-ckpt', '--model-ckpt', type=str, default='svm_sgd', help='Type of the operator')
    parser.add_argument('-m', '--model', type=str, default='perceptron', help='Type of the operator')
    parser.add_argument('-sr', '--sampling-ratio', type=float, default=0.2, help='Type of the operator')
    parser.add_argument('-r', '--repitition', type=int, default=20, help='Type of the operator')
    parser.add_argument('-q', '--query',  type=str, default='all', help='Type of the operator')
    parser.add_argument('-f', '--score-file',  type=str, default=None, help='Type of the operator')
    parser.add_argument('-o', '--op-name',  type=str, default=None, help='The name of the operator that we want to build the ML Model')
    parser.add_argument("-dt", "--data-type", type=str, default='tabular', help='Available Data Type:\n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    parser.add_argument("-tl", "--target-labels", type=str, default=None, help='Labels to the target dataset/datasets')
    parser.add_argument("-vb", "--vector-db", action="store_true", help="Use the vector from the vector database")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help='One independent run per seed (fresh seeded lake sample + seeded surrogate, '
                             'repititions=1 each); per-run metrics are appended to <out>/results_sr_runs.csv')
    parser.add_argument('-sDB', '--show-DBs', action="store_true", help="Show all the available vector Dabase colections with their description")
    parser.add_argument("--seed", type=int, default=0,
                        help='Global random seed (numpy/random/torch), set once before the experiments start')

    args = parser.parse_args()
    set_global_seed(args.seed)
    show_Vec_DB_collections = args.show_DBs
    if show_Vec_DB_collections:
        qdrant_ctrl = qdrant_controller()
        all_collections = qdrant_ctrl.get_all_colections()
        print('Available Collections: ')
        for collection in all_collections.collections:
            info = qdrant_ctrl.get_collection_info(collection.name)
            print(f"\n=== Collection: {collection.name} ===")
            print(f"Vector size: {info.config.params.vectors.size}")
            print(f"Distance: {info.config.params.vectors.distance}")
            print(f"Shard number: {info.config.params.shard_number}")
        sys.exit()

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
    target_labels = args.target_labels
    data_type_str = args.data_type
    use_vector_DB = args.vector_db

    if data_type_str.lower() == 'tabular':
        data_type = 1
    elif data_type_str.lower() == 'graph':
        data_type = 2
    elif data_type_str.lower() == 'image':
        data_type = 3
    else:
        AssertionError('Wrong Data type available data types: \n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    # vec_input_path, pred_path, out_path, sr, model_path
    
    if args.seeds is not None:
        import csv, os, time
        runs_csv = os.path.join(out_path, 'results_sr_runs.csv')
        for seed in args.seeds:
            t0 = time.time()
            # fresh object per run: train_operator replaces self.dataset, so it is not re-entrant
            train_operator = Train_Operator(model_path=ckpt_model_path, pred_path=pred_path,
                                            vec_input_path=vectors_path, out_path=out_path, sr=sr,
                                            data_type=data_type, use_vector_DB=use_vector_DB)
            m = train_operator.train_operator(ML_model_name=model, query=query, scores_file=score_file,
                                              repititions=1, op_name=op_name,
                                              score_target_file=target_labels, seed=seed)
            new = not os.path.isfile(runs_csv)
            with open(runs_csv, 'a', newline='') as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(['op_name', 'model', 'sr', 'seed', 'RMSE', 'MAE', 'NRMSE', 'MAD', 'MaPE', 'Acc',
                                'r^2', 'wall_time_s'])
                w.writerow([op_name, model, sr, seed, m['RMSE'], m['MAE'], m['NRMSE'], m['MAD'], m['MaPE'],
                            m['Acc'], m['r^2'], time.time() - t0])
        return

    train_operator = Train_Operator(model_path=ckpt_model_path,
                                    pred_path=pred_path,
                                    vec_input_path=vectors_path,
                                    out_path=out_path,
                                    sr=sr,
                                    data_type=data_type,
                                    use_vector_DB=use_vector_DB)

    train_operator.train_operator(ML_model_name=model, 
                                  query=query, 
                                  scores_file=score_file, 
                                  repititions=repititions,
                                  op_name=op_name,
                                  score_target_file=target_labels)

if __name__ == '__main__':
    main()

