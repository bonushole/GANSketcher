import tensorflow as tf
import numpy as np

@tf.function()
def add_noise(img_tensor, seed=None):
    # replace the green layer with noise
    random_layer = tf.random.uniform(shape=[256,256], seed=seed)
    layers = tf.unstack(img_tensor, axis=2)
    return tf.stack([layers[0], random_layer, layers[2]], axis=2)

LABEL_DICT = {'Michelangelo':0, 'Mikhail_Vrubel':1, 'Frida_Kahlo':2, 'Titian':3, 'Pablo_Picasso':4, 'Salvador_Dali':5, 'Diego_Rivera':6, 'Vincent_van_Gogh':7}

@tf.function()
def add_label(image, label_string):
    label = LABEL_DICT[label_string]
    label_layer = [tf.ones([256]) if i==label else tf.zeros([256]) for i in range(256)]
    label_layer = tf.stack(label_layer)
    layers = tf.unstack(img_tensor, axis=2)
    return tf.stack([layers[0], layers[1], label_layer], axis=2)

def artist_from_filename(filename):
    filename = tf.strings.regex_replace(filename, '.*\/', '')
    filename = tf.strings.regex_replace(filename, '_.*\.jpg', '')
    return filename

def average_color(image):
    return tf.reduce_mean()
