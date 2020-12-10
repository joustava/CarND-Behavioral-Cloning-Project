from keras import models
import numpy as np
import glob
from scipy import ndimage


def predict(X, model_path='./models/model.h5'):
    """
    Predicts the outputs for the inputs given with the restored model.
    """
    model = models.load_model(model_path)
    # model.summary()
    return model.predict(X)


# Load the three example images
image_paths = glob.glob('./assets/*_2020_12_08_10_46_19_361.jpg')
# Make sure they are alway in the same order, center, left, right
image_paths = sorted(image_paths)

img1 = ndimage.imread(image_paths[0])
img2 = ndimage.imread(image_paths[1])
img3 = ndimage.imread(image_paths[2])
img_batch = np.array([img1, img2, img3])

prediction = predict(img_batch)
print("CENTER: \t ", prediction[0], "\n")
print("LEFT: \t", prediction[1], "\n")
print("RIGHT: \t", prediction[2], "\n")
