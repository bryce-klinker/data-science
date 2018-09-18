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

# Take care of missing data
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(features[:, 1:3])
features[:, 1:3] = imputer.transform(features[:, 1:3])

labelEncoder_Features = LabelEncoder()
features[:, 0] = labelEncoder_Features.fit_transform(features[:, 0])

oneHotEncoder = OneHotEncoder(categorical_features=[0])
features = oneHotEncoder.fit_transform(features).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into training and test sets.
features_train, features_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=0)

# Feature scaling
standardScaler_Features = StandardScaler()
features_train = standardScaler_Features.fit_transform(features_train)
features_test = standardScaler_Features.transform(features_test)

print(features)
print(features_train)
print(features_test)

print(y)
print(y_test)
print(y_train)