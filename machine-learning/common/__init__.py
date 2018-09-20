import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from common.data_sets import get_dataset_path
from common.prepared_dataset import PreparedDataset

def get_dataset(dataset_name, dependent_variable_index=3, test_size=0.2):
    # Importing the dataset
    return PreparedDataset(dataset_name, dependent_variable_index, test_size)