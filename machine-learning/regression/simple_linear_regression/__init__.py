from common import prepare_dataset

def execute():
    dataset = prepare_dataset("salary-data", dependent_variable_index=1, test_size=1/3)[0]
    print(dataset)