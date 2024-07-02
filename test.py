# import numpy as np
#
# # a = np.load("data/california_housing/N_train.npy")
# # print(a[0])
# from ucimlrepo import fetch_ucirepo
#
# # fetch dataset
# wine = fetch_ucirepo(id=109)
#
# # data (as pandas dataframes)
# X = wine.data.features
# y = wine.data.targets
#
# # metadata
# print(wine.metadata)
#
# # variable information
# print(wine.variables)
import numpy as np
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("cspencergo/llama-2-7b-tabular")
# model = AutoModelForCausalLM.from_pretrained("cspencergo/llama-2-7b-tabular")
# model.tokenizer(10)
# model = AutoModelForSequenceClassification.from_pretrained("zeon8985army/bert-base-bible3-numbers")
# model.tokenizer(torch.Tensor(10))
# from transformers import TableTransformerModel
# print(TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST)
# model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
model = AutoModel.from_pretrained("facebook/data2vec-text-base")
tokens = tokenizer('Hello world')
print(tokens)
out = model(tokens['input_ids'])
print(out)
