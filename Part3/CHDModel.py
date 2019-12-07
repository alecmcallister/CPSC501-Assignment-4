from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf


#train_file_path = tf.keras.utils.get_file("train.csv", 'heart.csv')
test_file_path = 'heart.csv'

np.set_printoptions(precision=3, suppress=True)

print(test_file_path)