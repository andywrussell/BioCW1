import pandas as pd

def MSE(output, prediction):
    return (output - prediction)**2


def read_data(path, filename):
    data = pd.read_csv(path + 'Data/' + filename, sep='\t', header=None)
    column_names =data.columns.tolist()
    data.columns = column_names

    
    inputs = data.drop(column_names[-1:], axis=1)
    outputs = data.drop(column_names[:-1], axis=1)
    return (inputs, outputs)