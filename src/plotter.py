import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
from data import load_history
import sys


def plot_training_history(history, save_as_file='./assets/plots/training-plot.png'):
    """
    Plot loss over epochs from a keras model fitting history object and save as image.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = np.arange(1, len(loss) + 1, 1)

    plt.figure()
    plt.plot(epochs, loss, label="training")
    plt.plot(epochs, val_loss, label="validation")
    plt.title("Training/validation loss per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_as_file)


def plot_keras_model(model):
    plot_model(
        model,
        to_file="./assets/plots/network-model.png",
        show_shapes=True,
        #         show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        #         expand_nested=False,
        #         dpi=96,
    )


def visualize(model_file):
    #     history = load_history()
    model = load_model(model_file)
    plot_keras_model(model)
#     plot_training_history(history)
    model.summary()


# arguments = sys.argv
# visualize(arguments[1])
