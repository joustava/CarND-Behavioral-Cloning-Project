from keras.models import Sequential
from keas.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')