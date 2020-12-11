from keras import models
import numpy as np
import glob
from scipy import ndimage


def predict(X, model_path='./models/model.h5'):
    """
    Predicts the outputs for the inputs given with the restored network model.
    """
    model = models.load_model(model_path)
    # model.summary()
    return model.predict(X)


# Load the example images
image_paths = glob.glob('./assets/cameras/*.jpg')
# Make sure they are alway in the same order, center, left, right
image_paths = sorted(image_paths)

# Read images from file and create an numpy array.
img_batch = []
for file_name in image_paths:
    img = ndimage.imread(file_name)
    img_batch.append(img)

img_batch = np.array(img_batch)

# Make example predictions
prediction = predict(img_batch)

# Display predictions
centers, lefts, rights = np.array_split(prediction, 3)
for i in range(len(centers)):
    image_path = image_paths[i]
    print("\n", image_path.rsplit('_', 1)[1])
    print("CENTER:\t", centers[i])
    print("LEFT:\t", lefts[i])
    print("RIGHT:\t", rights[i])
