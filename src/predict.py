from keras import models
import numpy as np
import glob
from scipy import ndimage


def predict(X, model_path='./models/model.h5'):
    model = keras.models.load_model(model_path)
    return model(X, training=False, verbose=1)


# Load the three example images
images = glob.glob('./assets/*_2020_12_08_10_46_19_361.jpg')
# Make sure they are alway in the same order, center, left, right
images = sort(images)
images = [ndimage.imread(img) for img in images]

prediction = predict(nd.array(images))
print(prediction)
