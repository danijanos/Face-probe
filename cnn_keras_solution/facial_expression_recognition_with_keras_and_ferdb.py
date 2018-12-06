import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

number_of_emotion_classes = 7
image_dimension = 48
training_batch_size = 256
learning_epochs = 1

# Data preparing and transforming:
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
        # Converts a class vector (integers) to binary class matrix for use with categorical cross entropy
        emotion = tf.keras.utils.to_categorical(emotion, num_classes=number_of_emotion_classes)

        if 'Training' in usage:
            train_x.append(pixels)
            train_y.append(emotion)
        elif 'PublicTest' in usage:
            test_x.append(pixels)
            test_y.append(emotion)
    except ValueError:
        print('', end='')

# --- train data transformation (SciPy is required):
train_x = np.array(train_x, 'float32')
train_y = np.array(train_y, 'float32')

train_x /= 255  # normalize between [0, 1]
train_x = train_x.reshape(train_x.shape[0], image_dimension, image_dimension, 1)
train_x = train_x.astype('float32')
# ---

# --- test data transformation:
test_x = np.array(test_x, 'float32')
test_y = np.array(test_y, 'float32')

test_x /= 255
test_x = test_x.reshape(test_x.shape[0], image_dimension, image_dimension, 1)
test_x = test_x.astype('float32')
# ---

# The implementation of the network:

model = tf.keras.models.Sequential()

# The CNN structure:

# The first convolution layer:
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu, input_shape=(image_dimension, image_dimension, 1)))
# model shape = 48×48×64 = 147456
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# The second convolution layer:
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# The third convolution layer:
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# Add a flattening layer (batch size remains the same):
model.add(tf.keras.layers.Flatten())

# Fully connected layer:
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))

# Classification layer:
model.add(tf.keras.layers.Dense(number_of_emotion_classes, activation=tf.nn.softmax))

# Training section:

generator = ImageDataGenerator(vertical_flip=True)
# randomly select train set instances
train_data = generator.flow(train_x, train_y, batch_size=training_batch_size)
test_data = generator.flow(test_x, test_y, batch_size=training_batch_size)

# create tb logs:
tensor_board = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])

# train for all train set:
# model.fit_generator(train_x, train_y, epochs=learning_epochs)

# Train with test validation:
model.fit_generator(
    train_data,
    validation_data=test_data,
    steps_per_epoch=training_batch_size,
    epochs=learning_epochs,
    callbacks=[tensor_board])

# train for randomly selected train sets:
# model.fit_generator(
#     train_data,
#     steps_per_epoch=training_batch_size,
#     epochs=learning_epochs,
#     callbacks=[tensor_board])

# The evaluation of the network:
test_scores = model.evaluate(test_x, test_y)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1] * 100)
