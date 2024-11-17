import tensorflow as tf
print(tf.__version__)
print(dir(tf.keras))  # This should list Keras components, including `layers`

#test file to setup enviorment
from tensorflow.keras.layers import Dropout, Dense, Input
