import pandas as pd

def MSE(output, prediction):
    return (output - prediction)**2


def read_data(path, filename):
    data = pd.read_csv(path + 'Data/' + filename, sep='\t', header=None)
    print(data.shape[1])
    column_names = ["input{}".format(i) for i in range(1, data.shape[1])] + ["output"]
    data.columns = column_names
    
    inputs = data.drop(column_names[-1:], axis=1)
    outputs = data.drop(column_names[:-1], axis=1)
    return (inputs, outputs)