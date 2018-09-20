import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from common.data_sets import get_dataset_path

def prepare_dataset(dataset_name, dependent_variable_index=3, test_size=0.2):
    # Importing the dataset
    dataset_path = get_dataset_path(dataset_name)
    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, dependent_variable_index].values

    # Splitting the dataset into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return dataset, X, y, X_train, X_test, y_train, y_test

def prepare_dataset_with_scaling(dataset_path, dependent_variable_index=3, test_size=0.2):
    dataset, X, y, X_train, X_test, y_train, y_test = prepare_dataset(dataset_path, dependent_variable_index=dependent_variable_index, test_size=test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return dataset, X, y, X_train, X_test, y_train, y_test