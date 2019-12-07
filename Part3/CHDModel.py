from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras

train_file_path = 'heart_train.csv'  # 396 rows
test_file_path = 'heart_test.csv'  # 66 rows


def prepare_dataset(path):
    df = pd.read_csv(path)
    df['famhist'] = pd.Categorical(df['famhist'])
    df['famhist'] = df.famhist.cat.codes
    df.pop('row.names')

    target = df.pop('chd')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    shuffled_dataset = dataset.shuffle(len(df)).batch(1).repeat()

    return shuffled_dataset, df


train_dataset, train_df = prepare_dataset(train_file_path)
test_dataset, test_df = prepare_dataset(test_file_path)


def get_compiled_model():
    model = keras.Sequential([
        keras.layers.Dense(9, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = get_compiled_model()
model.fit(train_dataset, epochs=7, steps_per_epoch=train_df.shape[0])

model_loss, model_acc = model.evaluate(test_dataset, steps=test_df.shape[0],  verbose=2)
print(f"Model Loss:     {model_loss:.2f}")
print(f"Model Accuracy: {model_acc*100:.1f}%")
