import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """
    Set seed for reproducibility across numpy, random, and tensorflow.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_params(model):
    """
    Print number of trainable parameters in the model.
    """
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"Trainable parameters: {trainable_params:,}")

    
def plot_training_curves(history):
    """
    Plot training and validation loss and other metrics in separate subplots.
    """
    history_dict = history.history
    loss_keys = [k for k in history_dict if 'loss' in k]
    metric_keys = [k for k in history_dict if 'loss' not in k]

    # Plot losses
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for key in loss_keys:
        plt.plot(history_dict[key], label=key)
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot other metrics
    plt.subplot(1, 2, 2)
    for key in metric_keys:
        plt.plot(history_dict[key], label=key)
    plt.title("Metric Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
