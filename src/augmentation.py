import numpy as np
import keras
import sklearn
from scipy import ndimage


class CustomDataGenerator(keras.utils.Sequence):
    """
    Custom keras data generator based on https://keras.io/api/utils/python_utils/#sequence-class

    Reads driving sensor samples and converts these samples into data we can use for e.g training of a 
    network through a models fit_generator method.

    From docs:
    'Sequence are a safer way to do multiprocessing. This structure guarantees that the network will
    only train once on each sample per epoch which is not the case with generators.'
    """

    def __init__(self, samples, batch_size=64, image_path='/opt/data/IMG/'):
        self.samples = samples
        self.base_path = image_path
        self.batch_size = batch_size
        self.correction = 0.3
        self.on_epoch_end()

    def __len__(self):
        """
        return the number of batches
        """
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        """
        return one batch
        """
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_samples = self.samples[start:end]

        return self.__create_batch(batch_samples)

    def on_epoch_end(self):
        """
        Randomzise whole dataset for every epoch.
        """
        sklearn.utils.shuffle(self.samples)

    def __create_batch(self, batch_samples):
        """
        Generates a batch of images/labels where steering angle for left and right are adjusted.
        """
        X, y = [], []

        for batch_sample in batch_samples:
            center_img, left_img, right_img = self.__load_images(batch_sample)
            center_angle = float(batch_sample[3])

            X.append(center_img)
            X.append(left_img)
            X.append(right_img)

            y.append(center_angle)
            y.append(center_angle + self.correction)
            y.append(center_angle - self.correction)

        X = np.array(X)
        y = np.array(y)
        # Radomize batch samples
        return sklearn.utils.shuffle(X, y)

    def __load_images(self, sample):
        """
        Loads each image found in sample

        return images in center, left, right order
        """
        center_cam = self.base_path + sample[0].split('/')[-1]
        left_cam = self.base_path + sample[1].split('/')[-1]
        right_cam = self.base_path + sample[2].split('/')[-1]
        center_img = ndimage.imread(center_cam)
        left_img = ndimage.imread(left_cam)
        right_img = ndimage.imread(right_cam)

        return center_img, left_img, right_img
