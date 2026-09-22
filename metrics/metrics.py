import tensorflow as tf

# IOU
def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

# Accuracy
def accuracy_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

# True Positive Rate (Sensitivity / Recall)
def true_positive_rate(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    
    true_positives = tf.reduce_sum(y_true * y_pred)
    actual_positives = tf.reduce_sum(y_true)
    return true_positives / (actual_positives + 1e-6)

# F1 Score
def f1_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)

    true_positives = tf.reduce_sum(y_true * y_pred)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

    return (2 * true_positives) / (
        2 * true_positives + false_positives + false_negatives + 1e-6
    )