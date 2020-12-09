from keras import models
import numpy as np
import glob
# from scipy import ndimage
from keras.preprocessing import image_dataset_from_directory

# print(tf.__version__)


def predict(X, model_path='./models/model.h5'):
    model = models.load_model(model_path)
    return model(X)


# Load the three example images
image_paths = glob.glob('./assets/*_2020_12_08_10_46_19_361.jpg')
# Make sure they are alway in the same order, center, left, right
image_paths = sorted(image_paths)

# img = image.load_img(image_paths[0])
# img_array = image.img_to_array(img)
# img_batch = np.expand_dims(img_array, axis=0)

dataset = image_dataset_from_directory('./assets')

prediction = predict(dataset)
print("PREDICTION\n\n", prediction, "\n\n")
