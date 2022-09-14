from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.image import imread
import os
from datetime import datetime
from tensorflow import keras
from ImageClassification.settings import BASE_DIR,MODELS_PATH

# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(300, 300))

    img = img_to_array(img)

    img = img.reshape(1, 300, 300, 3)

    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# load an image and predict the class
def run_example(im_file):
    d = {}
    img = load_image(im_file)

    model = load_model(MODELS_PATH+'/Truck_classification/final_model.h5')

    result = model.predict(img)
    # print(result[0])
    # print(model.predict_generator(img))
    # print("\n")
    if (result[0][0] == 1.0):
        d['Result'] = "The truck image classified is not an object of interest"
    else:
        d['Result'] = "The truck image classified is an object of interest"

    return d


