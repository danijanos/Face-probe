import numpy as np
import tensorflow as tf

with open('./data/fer2013.csv') as fer:
    expression_data = fer.readlines()

data_lines = np.array(expression_data)
data_instances = data_lines.size

train_x, train_y, test_x, test_y = [], [], [], []

for i in range(1, data_instances):
    try:

        emotion, image, usage = data_lines[i].split(',')
        value = image.split(' ')
        pixels = np.array(value, 'float32')

        # one-hot encoding of the data
        # Converts a class vector (integers) to binary class matrix for use with categorical_crossentropy
        emotion = tf.keras.utils.to_categorical(emotion, num_classes=7)

        if 'Training' in usage:
            train_x.append(pixels)
            train_y.append(emotion)
        elif 'PublicTest' in usage:
            test_x.append(pixels)
            test_y.append(emotion)
    except:
        print('', end='')
