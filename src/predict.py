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

img_batch = []

for file_name in image_paths:
    img = ndimage.imread(file_name)
    img_batch.append(img)

img_batch = np.array(img_batch)

prediction = predict(img_batch)

centers, lefts, rights = np.array_split(prediction, 3)

for i in range(len(centers)):
    print("\n\n", image_paths[i*3])
    print("CENTER: ", centers[i], "\n")
    print("LEFT: \t", lefts[i], "\n")
    print("RIGHT: \t", rights[i], "\n")
