# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

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

print(features)
print(y)