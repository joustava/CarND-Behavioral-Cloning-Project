from data import load_samples, save_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, AveragePooling2D, MaxPooling2D, Cropping2D
from sklearn.model_selection import train_test_split
from data_stream import generator
from math import ceil
from plotter import plot_training_history

print("Traing model...")

# Hyper params
epochs = 15
batch_size = 32
dropout_rate = 0.5

# Logging
verbosity = 1


# Data
input_shape = (160, 320, 3)
cropped_shape = (90, 320, 3)
samples = load_samples()
training_samples, validation_samples = train_test_split(
    samples, test_size=0.3, shuffle=True)
steps_per_epoch = ceil(len(training_samples) / batch_size)
validation_steps = ceil(len(validation_samples) / batch_size)

# Stream/generator for memory efficiency
train_generator = generator(training_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Create model
model = Sequential()


# Preprocessing
# Exclude hood of car and scenery above road horizon from images
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
# center around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.0,
                 input_shape=cropped_shape, output_shape=cropped_shape))


# Layers

model.add(Conv2D(filters=32, kernel_size=(5, 5),
                 activation='softsign'))
model.add(AveragePooling2D())

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='softsign'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='softsign'))

model.add(Flatten())

model.add(Dense(units=512, activation='softsign'))
model.add(Dropout(dropout_rate))

model.add(Dense(units=256, activation='softsign'))
model.add(Dropout(dropout_rate))

model.add(Dense(units=1, activation='softsign'))

model.compile(loss='rmse', optimizer='adam')

# Training
training = model.fit_generator(train_generator,
                               steps_per_epoch=steps_per_epoch,
                               validation_data=validation_generator,
                               validation_steps=validation_steps,
                               epochs=epochs, verbose=verbosity)
plot_training_history(training)
# Backup previous model and save new model
save_model(model)
