import numpy as np

with open('./data/fer2013.csv') as fer:
    expression_data = fer.readlines()

data_lines = np.array(expression_data)
print(data_lines.size)
