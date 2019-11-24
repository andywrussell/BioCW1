import pandas as pd

def MSE(output, prediction):
    return (output - prediction)**2

def SUM(output, prediction):
    return abs(output - prediction)


def read_data(path, filename, sample = False):
    data = pd.read_csv(path + 'Data/' + filename, delim_whitespace=True, header=None)
    column_names = data.columns.tolist()
    data.columns = column_names

    
    if (sample):
        data = data.sample(10).reset_index(drop=True)

    inputs = data.drop(column_names[-1:], axis=1)
    outputs = data.drop(column_names[:-1], axis=1)
    return (inputs, outputs)