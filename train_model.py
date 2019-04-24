import numpy as np
from tensorflow.keras import models, layers, utils
from tensorflow.keras.callbacks import TensorBoard
import pickle

tensorboard = TensorBoard(log_dir="logs/nn/")

(X, Y) = pickle.load(open("train_data.pickle", "rb"));

X = np.array(X)
Y = np.array(Y)

Y_bin = utils.to_categorical(Y)

model = models.Sequential();
print(X.shape)
print(Y.shape)
print(Y_bin.shape)

model.add(layers.Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation("relu"))

model.add(layers.Dense(3))
model.add(layers.Activation("softmax"))

model.compile(loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

model.fit(X, Y_bin, batch_size=10, epochs=10, validation_split=0.1, callbacks=[tensorboard])

model.save("trained.model")






