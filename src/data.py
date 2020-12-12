import csv
import time
import os
import datetime
import pickle


def save_history(history, history_path='./assets/pickles/history.pickle'):
    with open(history_path, 'wb') as file:
        pickle.dump(history.history, file)


def load_history(history_path='./assets/pickles/history.pickle'):
    file = open(history_path, 'rb')
    history = pickle.load(file)
    file.close()
    return history


def load_samples(samples_path='/opt/data/driving_log.csv'):
    """
    Read a CSV driving_log file containing sensor data.

    Returns array of all samples found.
    indices:
        0: Center image 320X160x3
        1: Left image   320X160x3
        2: Right image  320X160x3
        3: steering angle -1..1
        4: throttle       0..1
        5: brake          0
        6: speed          0..30
    """
    lines = []
    with open(samples_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    return lines


def save_model(model, model_file_name='./models/model.h5', history_file_name='./assets/pickles/history.pickle'):
    """
    Makes a backup of the network in a model.h5 file (if existing) by appending a timestamp to the filename part to then save 
    the new model under the original name.

    The created files will include the models weights, architecture, compilation details (loss and metrics)
    and optimizer state.
    """
    if os.path.isfile(model_file_name):
        modified_ts = os.path.getmtime(model_file_name)
        name, ext = model_file_name.rsplit('.', 1)
        backup_file_name = "{}_{}.{}".format(name, int(modified_ts), ext)
        os.rename(model_file_name, backup_file_name)
        print("model backed up as: ", backup_file_name)

    model.save(model_file_name)
    # save_history(history, history_file_name)
    print("model saved as: ", model_file_name)
