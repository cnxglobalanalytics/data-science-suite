import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from ImageClassification.settings import BASE_DIR,MODELS_PATH


model = load_model(MODELS_PATH +'/Nike_fake_shoe_detection/nike_model.h5')

def nike_fake_detection(path):

    d={}
    img = load_img(path, target_size=(200, 200))
    x = img_to_array(img)

    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    if classes[0] < 0.5:
        d["Result"] = "The test shoe is a real Nike product"
    else:
        d["Result"] = "The test shoe is a fake Nike product"
    return d
