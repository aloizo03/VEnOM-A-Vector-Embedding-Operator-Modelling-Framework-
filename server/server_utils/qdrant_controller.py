from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import json
import numpy as np 

class qdrant_controller:

    def __init__(self):
        self.client =  QdrantClient(url="http://localhost:6333")

    def restart_qdrant(self):
        self.client =  QdrantClient(url="http://localhost:6333")

    def return_client(self):
        return self.client

    def create_collection(self, name, size):
        flag = self.client.create_collection(collection_name=name, 
                                          vectors_config=VectorParams(size=size, distance=Distance.DOT))
        print(flag)

    def insert_vectors(self, collection_name, data_vectors):
        points_struct = []
        idx = 0
        
        for dataset_name, vectors_ in data_vectors.items():
            points_struct.append(PointStruct(id=idx, vector=vectors_, payload={"dataset_name": dataset_name}))
            idx+=1

        operation_info = self.client.upsert(collection_name=collection_name,
                                            wait=True,
                                            points=points_struct,)
        
        print(operation_info)

    def get_collection_vec_dim(self, collection_name):
        try:
            print(f'Collection existance: {self.client.collection_exists(collection_name=collection_name)}')
            collection = self.client.get_collection(collection_name=collection_name)
        except:
            return 0
        json_collection = json.loads(collection.model_dump_json())
        
        return json_collection["config"]["params"]["vectors"]["size"]
        
    def get_collection_size(self, collection_name):
        try:
            print(f'Collection existance: {self.client.collection_exists(collection_name=collection_name)}')
            collection = self.client.get_collection(collection_name=collection_name)
        except:
            return 0
        
        return collection.points_count
        
    def get_vectors(self, collection_name):
        # self.client.get
        
        try:
            collection = self.client.get_collection(collection_name=collection_name)
            
        except:
            return None
        size_of_collection = self.get_collection_size(collection_name=collection_name)
        vectors = self.client.retrieve(collection_name=collection_name,
                             ids=[i for i in range(size_of_collection)],
                             with_vectors=True)
        vec_dic = {}
        for vec_ in vectors:
            vec_json = json.loads(vec_.model_dump_json())
            vec_dic[vec_json["payload"]["dataset_name"]] = vec_json["vector"]
        return vec_dic
    
    def similarity_search_(self, collection_name, vectors, select_percent):
        if select_percent > 1:
            select_percent = select_percent / 100
 
        top_k = np.ceil(select_percent * self.get_collection_size(collection_name=collection_name))
        sim_search_vectors = self.client.search(
                                                collection_name=collection_name,
                                                query_vector=vectors,  # type: ignore
                                                limit=top_k,
                                                with_vectors=True)
        out_dict = {}
        for vec in sim_search_vectors:
            vec_json = json.loads(vec.model_dump_json())
            out_dict[vec_json["payload"]["dataset_name"]] = vec_json["vector"]
        return out_dict
