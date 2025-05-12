import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from losses.loss import combo_loss
from metrics.metrics import iou_metric, true_positive_rate, accuracy_metric

# Example to set up your training
def train(model, train_generator, val_generator, learning_rate=1e4, epoches = 50):
    input_shape = (256, 256, 6)  # Example input shape
    
    callbacks_unet_with_resnet = [
        EarlyStopping(monitor="val_iou_metric", patience=10, restore_best_weights=True, verbose=1, mode="max"),
        ModelCheckpoint("best_model.keras", monitor="val_iou_metric", save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        CSVLogger("training_log.csv", append=True)
    ]
    
     # Set up your custom metrics and loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss=combo_loss, metrics=[iou_metric, true_positive_rate, accuracy_metric])
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epoches,
        verbose=1,
        callbacks=callbacks_unet_with_resnet
    )

if __name__ == '__main__':
    train()
