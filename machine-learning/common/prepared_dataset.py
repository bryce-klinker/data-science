from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.data_sets import get_dataset_path

class PreparedDataset:
    def __init__(self, name, dependent_variable_index, test_size):
        self._dataset_name = name
        self._dependent_variable_index = dependent_variable_index
        self._test_size = test_size
        self._scaler = StandardScaler()
        self._dataset = None
        self._features_training_set = None
        self._features_test_set = None
        self._dependents_training_set = None
        self._dependents_test_set = None
        self._has_split_dataset = False

    def get_dataset(self):
        if (self._dataset is None):
            dataset_path = get_dataset_path(self._dataset_name)
            self._dataset = read_csv(dataset_path)
        
        return self._dataset

    def get_features_set(self):
        return self.get_dataset().iloc[:, :-1].values

    def get_dependents_set(self):
        return self.get_dataset().iloc[:, self._dependent_variable_index].values

    def get_features_training_set(self):
        self._split_dataset()
        return self._features_training_set

    def get_features_test_set(self):
        self._split_dataset()
        return self._features_test_set

    def get_scaled_features_training_set(self):
        self._fit_scaler()
        return self._scaler.transform(self.get_features_training_set())

    def get_scaled_features_test_set(self):
        self._fit_scaler()
        return self._scaler.transform(self.get_features_test_set())

    def get_dependents_training_set(self):
        self._split_dataset()
        return self._dependents_training_set

    def get_dependents_test_set(self):
        self._split_dataset()
        return self._dependents_test_set

    def _split_dataset(self):
        if (self._has_split_dataset):
            return

        features_set = self.get_features_set()
        dependents_set = self.get_dependents_set()
        self._features_training_set, self._features_test_set, self._dependents_training_set, self._dependents_test_set = train_test_split(features_set, dependents_set, test_size=self._test_size, random_state=0)
        self._has_split_dataset = True

    def _fit_scaler(self):
        self._scaler.fit(self.get_features_training_set())

