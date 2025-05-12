import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from data.image_generator import ImageGenerator
from config.config import BATCH_SIZE, EPOCHS, INPUT_SHAPE, DATA_PATH, MODEL_SAVE_PATH
from train import train
from models.build_model import *
from losses.loss import combo_loss
from metrics.metrics import iou_metric, true_positive_rate, accuracy_metric
from utils import plot_training_curves

# Load and preprocess data
def load_data():
    print("Loading data...")

    image_paths = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)]
    
    train_generator = ImageGenerator(image_paths, batch_size=BATCH_SIZE, input_size=INPUT_SHAPE)
    
    val_image_paths = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)]
    val_generator = ImageGenerator(val_image_paths, batch_size=BATCH_SIZE, input_size=INPUT_SHAPE)
    
    return train_generator, val_generator

# Build the model
def build_and_compile_model():
    print("Building and compiling model...")
    
    model = build_unet_model(INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combo_loss,
        metrics=[iou_metric, true_positive_rate, accuracy_metric]
    )
    
    return model

# Main function to orchestrate the workflow
def main():
    """
    The main entry point to load data, build the model, and start training.
    """
    train_generator, val_generator = load_data()

    # Build and compile the model
    model = build_and_compile_model()

    print("Starting training...")
    history = train(model, train_generator, val_generator, learning_rate=1e-4, epoches=EPOCHS)  # Using the train function from train.py

    print("Training complete.")
    
    plot_training_curves(history = history)
    

if __name__ == '__main__':
    main()
