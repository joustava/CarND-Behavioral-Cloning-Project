import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(history, save_as_file='./assets/plots/training-plot.png'):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["acc"]
    val_accuracy = history.history["val_acc"]

    epochs = np.arange(0, len(loss))

    plt.figure()
    plt.plot(epochs, loss, label="training loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.plot(epochs, accuracy, label="training accuracy")
    plt.plot(epochs, val_accuracy, label="validation accuracy")
    plt.title("Training Loss and Accuracy Per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(save_as_file)