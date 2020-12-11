import numpy as np
import matplotlib.pyplot as plt


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
