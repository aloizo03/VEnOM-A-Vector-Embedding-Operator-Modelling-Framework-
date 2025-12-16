import torch

import joblib

from tqdm import tqdm
import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

from utils_.dataset_update import Dataset_Evolution
import matplotlib.pyplot as plt
import logging
import os
import time
import pickle
from server.server_utils.qdrant_controller import qdrant_controller

 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--model-weight', type=str, default=None, help="Model Weights for the Vector Embeddings")
    parser.add_argument("-i", "--data-input", type=str, help="Input dataset path f.e data/path/dir")
    parser.add_argument("-out", "--out-path", type=str, default="results/test",
                        help='Output path for the saving of the embeddings')
    parser.add_argument('-v', '--vectors', type=str, help='vectors path or VectorDB collection name')
    parser.add_argument('-vt', '--vectors-time', type=str, default='', help='Dataset latest modified time path')
    parser.add_argument('-bz', '--batch-size', type=int, default=1, help='total batch size')
    parser.add_argument('-vs', '--vector-size', type=int, default=100, help='The vector embedding token dimension')
    parser.add_argument("-dt", "--data-type", type=str, default='tabular', help='Available Data Type:\n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    parser.add_argument("-s", "--save-to", type=str, default='local', help="Where do you want to save the vector embedding representation:\n\t-s local: save to local repository output\n\t-s vectorDB: save to Qdrant Vector Database")
    parser.add_argument('-sDB', '--show-DBs', action="store_true", help="Show all the available vector Dabase colections with their description")


    args = parser.parse_args()

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

    model_path = args.model_weight
    data_input_path = args.data_input
    out_path = args.out_path
    batch_size = args.batch_size
    save_to = args.save_to
    data_type_str = args.data_type
    d_token = args.vector_size
    collection_name = args.vectors
    dataset_latest_time = args.vectors_time
    if save_to.lower() == 'vectordb':
        print('Use vector DB')
        use_vectordb=True
    else:
        use_vectordb=False

    if data_type_str.lower() == 'tabular':
        data_type = 1
    elif data_type_str.lower() == 'graph':
        data_type = 2
    elif data_type_str.lower() == 'image':
        data_type = 3
    else:
        AssertionError('Wrong Data type available data types: \n\t-tabular: For tabular Dataset\n\t-graph: For Graph Dataset\n\t-Image: For Image Dataset')
    
    dataset_evolution = Dataset_Evolution(data_name=collection_name, 
                                            data_path=data_input_path, 
                                            model_path=model_path, 
                                            out_path=out_path, 
                                            data_type=data_type, 
                                            data_times_path=dataset_latest_time, 
                                            use_vectorDB=use_vectordb, 
                                            d_token=d_token)
    
    dataset_evolution.update_vectors(batch_size=batch_size)

if __name__ == '__main__':
    main()