# 8. Training your network.
import csv
import cv2
import numpy as np
from scipy import ndimage


def load(samples_path='../data/driving_log.csv'):
    """
    Read CSV log file containing sensor data

    returns array of samples
    """
    lines = []
    with open(samples_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

# # Adjust file paths as they have been copied into another filesystem and then load the data
# images = []
# measurements = []
# for line in lines:
#   src_path = line[0]
#   filename = src_path.split('/')[-1]
#   current_path = '../data/IMG' + filename
#   image = ndimage.imread(current_path)
#   # add img
#   images.append(image)
#   # add steering angle
#   measurements.append(float(line[3]))

# # convert data to numpy arrays.
# X_train = np.array(images)
# y_train = np.array(measurements)
# return X_train, y_train
