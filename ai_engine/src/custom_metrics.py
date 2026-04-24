import tensorflow as tf
import numpy as np

def weighted_binary_crossentropy(skill_weights):
    weights = tf.cast(skill_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Clip predictions untuk menghindari NaN
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, dtype=tf.float32)

        # komponen Binary Crossentropy
        bce_loss = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # bobot urgensi
        weight_modifier = (y_true * (weights - 1.0)) + 1.0
        weighted_bce_loss = bce_loss * weight_modifier

        # Rata-ratakan loss di seluruh skill dan batch
        return tf.reduce_mean(weighted_bce_loss, axis=-1)

    return loss
