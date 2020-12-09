from keras import models
import numpy as np
import glob
from scipy import ndimage
# from keras.preprocessing import image

# print(tf.__version__)


def predict(X, model_path='./models/model.h5'):
    model = models.load_model(model_path)
    return model(X)


# Load the three example images
image_paths = glob.glob('./assets/*_2020_12_08_10_46_19_361.jpg')
# Make sure they are alway in the same order, center, left, right
image_paths = sorted(image_paths)

img1 = ndimage.imread(image_paths[0])
img2 = ndimage.imread(image_paths[1])
img3 = ndimage.imread(image_paths[2])
img_batch = np.expand_dims([img1, img2, img3], axis=0)

prediction = predict(img_batch)
print("PREDICTION\n\n", prediction, "\n\n")
