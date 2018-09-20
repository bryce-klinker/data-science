from common import get_dataset
from sklearn.linear_model import LinearRegression

def execute():
    preparedDataset = get_dataset("salary-data", dependent_variable_index=1, test_size=1/3)
    dataset = preparedDataset.get_dataset()
    
    features_train = preparedDataset.get_features_training_set()
    features_test = preparedDataset.get_features_test_set()

    dependents_train = preparedDataset.get_dependents_training_set()
    dependents_test = preparedDataset.get_dependents_test_set()

    regressor = LinearRegression()
    regressor.fit(features_train, dependents_train)

    dependents_predictions = regressor.predict(features_test)
    print(dataset)
    print(dependents_test)
    print(dependents_predictions)