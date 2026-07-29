import os

os.environ.setdefault('HF_HOME', os.path.join(os.getcwd(), '.hf_cache'))

import argparse
from utils_.vectors import Vectorise
from utils_.utils import check_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--model-weight', type=str, default=None,
                        help="Model Weights for the Vector Embeddings")
    parser.add_argument("-i", "--data-input", type=str, help="Input datasets path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-bz', '--batch-size', type=int, default=1, help='total batch size')
    parser.add_argument('-cn', '--collection-name', type=str, default='collection_name',
                        help='Set the vectors collection name')
    parser.add_argument('-vs', '--vector-size', type=int, default=100, help='The vector embedding token dimension')
    parser.add_argument("-dt", "--data-type", type=str, default='tabular',
                        help='Available Data Type:\n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    parser.add_argument("-s", "--save-to", type=str, default='local',
                        help="Where do you want to save the vector embedding representation:\n\t-s local: save to local repository output\n\t-s vectorDB: save to Qdrant Vector Database")
    parser.add_argument("-tm", "--tab-model", type=str, default='num2vec', choices=['num2vec', 'tabfm', 'dataset2vec'],
                        help="Model used to vectorise tabular datasets:\n\t-num2vec: the trained Num2Vec model (needs --model-weight)\n\t-tabfm: Google's TabFM foundation model (google/tabfm-1.0.0-pytorch); --model-weight is optional and may point to a local TabFM checkpoint dir\n\t-dataset2vec: the pretrained Dataset2Vec meta-feature model (hadijomaa/dataset2vec, 32-d vectors); --model-weight is optional and may point to a checkpoint dir")
    parser.add_argument("--tabfm-task", type=str, default='regression', choices=['classification', 'regression'],
                        help='Which TabFM backbone weights to use for the embeddings')
    parser.add_argument("--tabfm-max-rows", type=int, default=512,
                        help='Maximum number of rows per dataset sampled as TabFM context')
    parser.add_argument("--d2v-split", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Which pretrained Dataset2Vec checkpoint split to use')
    parser.add_argument("--d2v-batches", type=int, default=10,
                        help='Number of sampled batches averaged into each Dataset2Vec vector')

    args = parser.parse_args()

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    batch_size = args.batch_size
    save_to = args.save_to
    data_type_str = args.data_type
    d_token = args.vector_size
    collection_name = args.collection_name
    if data_type_str.lower() == 'tabular':
        data_type = 1
    elif data_type_str.lower() == 'graph':
        data_type = 2
    elif data_type_str.lower() == 'image':
        data_type = 3
    else:
        AssertionError(
            'Wrong Data type available data types: \n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')

    clustering = Vectorise(data_path=data_input_path,
                           model_path=model_path,
                           out_path=out_path,
                           data_type=data_type,
                           save_to=save_to,
                           d_token=d_token,
                           collection_name=collection_name,
                           tab_model=args.tab_model,
                           tabfm_task=args.tabfm_task,
                           tabfm_max_rows=args.tabfm_max_rows,
                           d2v_split=args.d2v_split,
                           d2v_batches=args.d2v_batches)
    print('Start vector calculation')
    clustering.compute_vectors(batch_size=batch_size)


if __name__ == '__main__':
    main()


