import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def weighted_binary_crossentropy(skill_weights):
    weights = tf.cast(skill_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, dtype=tf.float32)

        bce_loss = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # Apply per-skill urgency weights to penalize minority-class misses
        weight_modifier = (y_true * (weights - 1.0)) + 1.0
        weighted_bce_loss = bce_loss * weight_modifier

        return tf.reduce_mean(weighted_bce_loss, axis=-1)

    return loss

class F1Score(tf.keras.metrics.Metric):
    """F1 metric using Keras Precision/Recall."""

    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        # Prevent zero-division
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
