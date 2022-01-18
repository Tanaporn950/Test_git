from Training import load_data, batch_size
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from configparser import ConfigParser

img_height = 180
img_width = 180
config_object = ConfigParser()
config_object.read("config.ini")
trainingmodel = config_object["TRAININGMODEL"]
train_ds,val_ds = load_data()
model = load_model(trainingmodel["model"])

class_names = train_ds.class_names
print(class_names)


def classified(filepath):
    img = keras.preprocessing.image.load_img(
    #'C:/Users/66918/Desktop/archive_problem/'+filepath 
    filepath, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    """
    print(
        "This image most likely belongs to {} with a {:.2f} % confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    """
    return "This image most likely belongs to {} with a {:.2f} % confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

classified("C:/Users/66918/Desktop/img_classification/archive_problem/ff.jpg")