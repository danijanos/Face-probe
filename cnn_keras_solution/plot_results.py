import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot
from keras.models import model_from_json

model_in_json = open('./models/facial_expression_model_structure.json', 'r').read()
FER_model = model_from_json(model_in_json)
FER_model.load_weights('./models/fe_modelweights.h5')

test_data_path = './data/test_images/dj_1_cropped.jpg'
image_dimension = 48

# Requires Image included PIL
test_image = tf.keras.preprocessing.image.load_img(
    test_data_path,
    target_size=(image_dimension, image_dimension),
    color_mode='grayscale')

# Prepare the data from the test image for prediction:
values_from_testimage = tf.keras.preprocessing.image.img_to_array(test_image)
values_from_testimage = np.expand_dims(values_from_testimage, axis=0)
values_from_testimage /= 255

prediction_from_image = FER_model.predict(values_from_testimage)

# Drawing a bar chart which represents the confidential values of the classification
emotion_types = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
x_axis_value = np.arange(len(emotion_types))

plot.subplot(1, 2, 2)  # one row, two column, second (right side) subplot
plot.title('Emotion')
plot.ylabel('Percentage')
plot.bar(x_axis_value, prediction_from_image[0], align='center')
plot.xticks(x_axis_value, emotion_types)

# Prepare the data for showing the test image in the left subplot:
values_from_testimage = np.array(values_from_testimage, 'float32')
values_from_testimage = values_from_testimage.reshape([image_dimension, image_dimension])

plot.subplot(1, 2, 1)  # one row, two column, first (left side) subplot for showing the test image
plot.gray()
plot.imshow(values_from_testimage)

plot.show()
