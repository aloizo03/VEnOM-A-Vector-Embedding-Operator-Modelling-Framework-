from data.Dataset import Pipeline_Dataset
from utils_.utils import create_operator, predict_operator
import pickle
import logging
import os
import time

logger = logging.getLogger(__name__)


class Create_Operator:

    def __init__(self, out_path):
        self.out_path = out_path

        logging.basicConfig(filename=os.path.join(self.out_path, 'log_file.log'), encoding='utf-8',
                            level=logging.DEBUG)
        logger.setLevel(logging.INFO)

    def load_dict(self, path_):
        with open(path_, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def save_model(self, filename, operator):
        file_name = os.path.join(self.out_path, f'operator_{filename}_{time.time()}.pickle')
        with open('model.pkl', 'wb') as f:
            pickle.dump(operator, f)

    def create_operator(self, operator_name, most_relevant_data_path):
        most_relevant_data = self.load_dict(path_=most_relevant_data_path)
        dict_operators = {}
        for dataset_name, dataset_list in most_relevant_data.items():
            dataset_train = Pipeline_Dataset(dataset_list, norm=True, create_operator=True, ret_class=True)
            data_builder_train = dataset_train.get_Dataloader()
            X_train, y_train = data_builder_train.get_all_dataset_data()
            operator_ = create_operator(name=operator_name,
                                        X=X_train,
                                        y=y_train)
            dataset_predict = Pipeline_Dataset(dataset_name, norm=True, create_operator=True, ret_class=True)
            data_builder_train = dataset_predict.get_Dataloader()
            X_test, y_test = data_builder_train.get_all_dataset_data()
            acc = predict_operator(operator=operator_, X=X_test, y=y_test)
            logger.info(f'Operator name: {operator_name}, for dataset {dataset_name} accuracy is: {acc}')
            self.save_model(filename=operator_name, operator=operator_)

