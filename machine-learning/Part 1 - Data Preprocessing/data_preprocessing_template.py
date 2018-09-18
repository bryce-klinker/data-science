# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Data.csv')
features = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into training and test sets.
features_train, features_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=0)

# Feature scaling
# standardScaler_Features = StandardScaler()
# features_train = standardScaler_Features.fit_transform(features_train)
# features_test = standardScaler_Features.transform(features_test)
