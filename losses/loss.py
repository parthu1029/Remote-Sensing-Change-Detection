import tensorflow.keras.backend as K
import tensorflow as tf

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_true = tf.squeeze(y_true, axis=-1)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# Focal Loss
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.squeeze(y_pred, axis=-1)
        y_true = tf.squeeze(y_true, axis=-1)        
        
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss

# Combined Loss
def combo_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(gamma=2.0, alpha=0.25)(y_true, y_pred)