from data import load
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from sklearn.model_selection import train_test_split
from data_stream import generator
from math import ceil
import time
import os
import datetime

# Hyper params
epochs = 1
batch_size = 32


# Logging
verbosity = 1


# Data
input_shape = (160, 320, 3)
samples = load()
training_samples, validation_samples = train_test_split(samples, test_size=0.2)
steps_per_epoch = ceil(len(training_samples)/batch_size)
validation_steps = ceil(len(validation_samples)/batch_size)

# Stream/generator for memory efficiency
train_generator = generator(training_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Create model
model = Sequential()


# Preprocessing
# center around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=input_shape, output_shape=input_shape))


# Layers
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Training
model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    epochs=epochs, verbose=verbosity)

# Backup previous model and save new model
model_file_name = './models/model.h5'
modified_ts = os.path.getmtime(model_file_name)

# date_time = datetime.datetime.fromtimestamp(
#     modified_ts).strftime("%Y-%M-%D")
name, ext = model_file_name.rsplit('.', 1)
os.rename(model_file_name, "{}_{}.{}".format(name, modified_ts, ext))

model.save(model_file_name)
