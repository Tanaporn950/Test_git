import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import glob
from tensorflow.keras.models import Sequential
import tensorflow as tf
from configparser import ConfigParser

#import matplotlib.pyplot as plt 
#import numpy as np
#import os
#import tensorflow_datasets as tfds
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.callbacks import TensorBoard
#from keras.preprocessing.image import ImageDataGenerator

# set config
config_object = ConfigParser()
config_object.read("config.ini")

#path = glob.glob("C:/Users/66918/Desktop/archive_problem/*/*/*.*")
path_img = config_object["IMAGE_COUNT"]
path = glob.glob(path_img["Cpath_img"])
image_count = len(list(path))
print(image_count)

batch_size = 64
num_classes = 3
# number of training epochs
epochs = 15
img_height = 180
img_width = 180

def load_data():
    #method read config file 
    #train_ds = #get from config
    training_set = config_object["TRAINING_DATA"]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    training_set["Ctraining_set"],
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    validation_set = config_object["VALIDATION_DATA"]
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_set["Cvalidation_set"],
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    return train_ds,val_ds

def create_model(input_shape):
    model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=input_shape),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()
    return model

if __name__ == "__main__":
    # load the data
    train_ds,val_ds = load_data()
    # constructs the model
    model = create_model(input_shape=(img_height, img_width, 3))
    # some nice callbacks
    # train
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    # save the model to disk
    trainingmodel = config_object["TRAININGMODEL"]
    model.save(trainingmodel["model"])

#Test