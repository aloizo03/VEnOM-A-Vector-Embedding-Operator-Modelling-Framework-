import chromadb

class chroma_db_controler:
    
    def __init__(self):
        self.client = chromadb.Client()
        
    def create_collection(self, collection_name):
        self.client.create_collection(name=collection_name)
        
    def insert_data(self, vectors_dict, collection):
        embeddings = []
        metatada = []
        idx = []
        
        collection = self.client.get_collection(name=collection)
        
        
    
    def get_collection(self, collection_name):
        return self.client.get_or_create_collection(name=collection_name) 
        
        