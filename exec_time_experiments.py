from data.Dataset import Pipeline_Dataset
from utils_.utils import create_operator, predict_operator, predict_linear_regression_operator, predict_time_series_model, predict_dbscan
import time
import os

filepath = '/home/ubuntu/Projects/Vector Embedding/data/HPC_2Krows/test'
print('HPC')
start_time = time.time()
files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)
print(len(paths_))
dataset_HPC = Pipeline_Dataset(paths_, norm=True, create_operator=True, ret_class=True)
data_builder_test = dataset_HPC.get_Dataloader()
X_train, y_train = data_builder_test.get_all_dataset_data()


operator_ = create_operator(name='linear_regression', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator linear_regression: {exec_time}')

start_time = time.time()
start_time = time.time()
files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)
print(len(paths_))
operator_ = create_operator(name='mlpregression', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator mlpregression: {exec_time}')
print('--------------------------------------------------------\n\n')

filepath = '/home/ubuntu/Projects/Vector Embedding/data/Stock/test_'
print('Stock')
start_time = time.time()

files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)

dataset_Stock = Pipeline_Dataset(paths_, norm=True, create_operator=True, ret_class=True)
data_builder_test = dataset_Stock.get_Dataloader()
X_train, y_train = data_builder_test.get_all_dataset_data()


operator_ = create_operator(name='linear_regression', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator linear_regression: {exec_time}')

start_time = time.time()

files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)

dataset_Stock = Pipeline_Dataset(paths_, norm=True, create_operator=True, ret_class=True)
data_builder_test = dataset_Stock.get_Dataloader()
X_train, y_train = data_builder_test.get_all_dataset_data()
operator_ = create_operator(name='mlpregression', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator mlpregression: {exec_time}')
print('--------------------------------------------------------\n\n')


filepath = '/home/ubuntu/Projects/Vector Embedding/data/Adult/test'
start_time = time.time()

print('Adult')
files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)

dataset_Stock = Pipeline_Dataset(paths_, norm=True, create_operator=True, ret_class=True)
data_builder_test = dataset_Stock.get_Dataloader()
X_train, y_train = data_builder_test.get_all_dataset_data()

operator_ = create_operator(name='perceptron', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator perceptron: {exec_time}')

start_time = time.time()

print('Adult')
files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)

dataset_Stock = Pipeline_Dataset(paths_, norm=True, create_operator=True, ret_class=True)
data_builder_test = dataset_Stock.get_Dataloader()
X_train, y_train = data_builder_test.get_all_dataset_data()

operator_ = create_operator(name='svm_sgd', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator svm_sgd: {exec_time}')

print('--------------------------------------------------------\n\n')

filepath = '/home/ubuntu/Projects/Vector Embedding/data/Weather/test'
start_time = time.time()

files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)

dataset_Stock = Pipeline_Dataset(paths_, norm=True, create_operator=True, ret_class=True)
data_builder_test = dataset_Stock.get_Dataloader()
X_train, y_train = data_builder_test.get_all_dataset_data()

operator_ = create_operator(name='perceptron', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator perceptron: {exec_time}')

start_time = time.time()

files_ = os.listdir(filepath)
paths_ = []
for file in files_:
    filepath_ = os.path.join(filepath, file)
    paths_.append(filepath_)

dataset_Stock = Pipeline_Dataset(paths_, norm=True, create_operator=True, ret_class=True)
data_builder_test = dataset_Stock.get_Dataloader()
X_train, y_train = data_builder_test.get_all_dataset_data()

operator_ = create_operator(name='svm_sgd', X=X_train, y=y_train)
exec_time = time.time() - start_time
print(f'Execution time for operator svm_sgd: {exec_time}')