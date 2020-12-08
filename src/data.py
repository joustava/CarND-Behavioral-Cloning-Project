# 8. Training your network.
import csv
import cv2
import numpy as np
from scipy import ndimage


def load(samples_path='/opt/data/driving_log.csv'):
    """
    Read CSV log file containing sensor data

    returns array of samples
    """
    lines = []
    with open(samples_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
