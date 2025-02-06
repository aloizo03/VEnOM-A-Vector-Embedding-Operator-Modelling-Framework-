import ismlldataset

dataset_id = 31 # (between 0-119)

metadataset = ismlldataset.datasets.get_metadataset(dataset_id=dataset_id)

print(metadataset)