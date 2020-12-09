from data import load_samples, save_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, AveragePooling2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from data_stream import generator
from math import ceil

print("Traing model...")

# Hyper params
epochs = 10
batch_size = 32
dropout_rate = 0.5

# Logging
verbosity = 1


# Data
input_shape = (160, 320, 3)
samples = load_samples()
training_samples, validation_samples = train_test_split(samples, test_size=0.2)
steps_per_epoch = ceil(len(training_samples) / batch_size)
validation_steps = ceil(len(validation_samples) / batch_size)

# Stream/generator for memory efficiency
train_generator = generator(training_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Create model
model = Sequential()


# Preprocessing
# center around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.0,
                 input_shape=input_shape, output_shape=input_shape))


# Layers
# model.add(Flatten())
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')

model.add(Conv2D(filters=32, kernel_size=(5, 5),
                 activation='tanh'))
model.add(AveragePooling2D())

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))
model.add(AveragePooling2D())

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))
model.add(AveragePooling2D())

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))
model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(units=1024, activation='tanh'))
model.add(Dropout(dropout_rate))

model.add(Dense(units=512, activation='tanh'))
model.add(Dropout(dropout_rate))

model.add(Dense(units=256, activation='tanh'))
model.add(Dropout(dropout_rate))

model.add(Dense(units=1, activation='tanh'))

model.compile(loss='mse', optimizer='adam')

# Training
model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    epochs=epochs, verbose=verbosity)

# Backup previous model and save new model
save_model(model)
