import tensorflow as tf

def erro_quadratico_medio(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
