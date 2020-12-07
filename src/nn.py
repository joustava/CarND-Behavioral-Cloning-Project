from keras.models import Sequential
from keas.layers import Dense, Flatten, Lambda

from data import load_samples

# Data
input_shape = (160, 320, 3)
X_train, y_train = load_samples()

# Create model
model = Sequential()

# Preprocessing

# Normalize
modela.add(Lambda(lambda: x: x / 255.0, input_shape=input_shape)
# Mean center
modela.add(Lambda(lambda: x: x - 0.5)

# Layers
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Training
model.fit(X_train, y_train, validation_split=0.3,
          shuffle=True, nbepoch=10, verbose=1)


model.save('model.h5')
