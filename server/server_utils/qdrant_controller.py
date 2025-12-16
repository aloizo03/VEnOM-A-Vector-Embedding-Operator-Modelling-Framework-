from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import json
import numpy as np 
from tqdm import tqdm
import time
import uuid

# run docker: 
# docker run -p 6333:6333 -p 6334:6334     -v "$(pwd)/qdrant_storage:/qdrant/storage:z"     qdrant/qdrant
class qdrant_controller:

    def __init__(self):
        self.client =  QdrantClient(url="http://localhost:6333", timeout=120.0)

    def restart_qdrant(self):
        self.client =  QdrantClient(url="http://localhost:6333", timeout=120.0)

    def return_client(self):
        return self.client

    def get_all_colections(self):
        avail_col = self.client.get_collections()
        return avail_col

    def get_collection_info(self, collection_name):
        info = self.client.get_collection(collection_name)
        return info

    def create_collection(self, name, size, distance=Distance.EUCLID):
        flag = self.client.create_collection(collection_name=name, 
                                          vectors_config=VectorParams(size=size, distance=distance))
        print(flag)

    
    def insert_vectors(self, collection_name, data_vectors):
        points_struct = []
        idx = 0
    
        for dataset_name, vectors_ in data_vectors.items():
            points_struct.append(PointStruct(id=idx, vector=vectors_, payload={"dataset_name": dataset_name}))
            idx+=1

        operation_info = self.client.upsert(collection_name=collection_name,
                                            wait=False,
                                            points=points_struct,)
        
        print(operation_info)

    def insert_vectors_in_batches(self, collection_name, data_vectors, times_dict, batch_size=100):
        points_struct = []
        idx = 0
    
        for dataset_name, vectors_ in data_vectors.items():
            time_create = times_dict[dataset_name]
            points_struct.append(PointStruct(id=idx, vector=vectors_, payload={"dataset_name": dataset_name,
                                                                               'create_data': time_create}))
            idx+=1

        total = len(points_struct)
        print(f"Uploading {total} vectors to collection '{collection_name}' in batches of {batch_size}...")
        for start in tqdm(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            batch = points_struct[start:end]

            try:
                operation_info = self.client.upsert(
                    collection_name=collection_name,
                    wait=False,          # Don’t block until indexing is complete
                    points=batch
                )
            except Exception as e:
                print(f"Warning: Batch {start // batch_size + 1} failed: {e}")
                time.sleep(3)  # small delay before retry
                continue

        print("✅ All vectors uploaded successfully.")

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

    def get_vectors_time(self, collection_name):
        try:
            collection = self.client.get_collection(collection_name=collection_name) 
        except:
            return None
        size_of_collection = self.get_collection_size(collection_name=collection_name)
        vectors = self.client.retrieve(collection_name=collection_name,
                             ids=[i for i in range(size_of_collection)],
                             with_vectors=True)

        time_dic = {}
        for vec_ in vectors:
            vec_json = json.loads(vec_.model_dump_json())
            time_dic[vec_json["payload"]["dataset_name"]] = vec_json["payload"]["create_data"]
        return time_dic
        

    def get_vectors(self, collection_name):        
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
    

    def upsert_collection(self, collection_name, vectors, times_):
        for dataset_name, vector_ in vectors.items():
            search = self.client.scroll(collection_name=collection_name,
                                        scroll_filter={"must": [{"key": "dataset_name", "match": {"value": dataset_name}}]},
                                        limit=1)

            existing_point = search[0] if search else None
            new_time = times_[dataset_name]

            if search[0]:
                existing_point = search[0][0]

            if existing_point:
                point_id = existing_point.id
                print(f"[UPDATE] Updating dataset {dataset_name} with point_id {point_id}")
            else:
                # Not found → insert new point with new ID
                point_id = str(uuid.uuid4())
                print(f"[INSERT] Creating new dataset {dataset_name} with point_id {point_id}")

            self.client.upsert(collection_name=collection_name,
                               points=[
                                   PointStruct(id=point_id,
                                               vector=vector_,
                                               payload={"dataset_name": dataset_name, "create_data": new_time})
                               ])
