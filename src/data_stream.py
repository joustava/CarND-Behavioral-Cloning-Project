import sklearn
import numpy as np
from scipy import ndimage
import os


def generator(samples, batch_size=64, correction=0.3, image_path='/opt/data/IMG/'):
    """
    ! Deprecated in favour of augmentation.py

    Lazily loads the sample driving data


    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_cam = image_path + batch_sample[0].split('/')[-1]
                left_cam = image_path + batch_sample[1].split('/')[-1]
                right_cam = image_path + batch_sample[2].split('/')[-1]
                center_img = ndimage.imread(center_cam)
                left_img = ndimage.imread(left_cam)
                right_img = ndimage.imread(right_cam)
                center_angle = float(batch_sample[3])

                images.append(center_img)
                images.append(left_img)
                images.append(right_img)

                angles.append(center_angle)
                angles.append(center_angle + correction)
                angles.append(center_angle - correction)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
