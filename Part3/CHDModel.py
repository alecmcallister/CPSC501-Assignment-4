from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow_core.python.keras import regularizers

train_file_path = 'heart_train.csv'  # 396 rows
test_file_path = 'heart_test.csv'  # 66 rows

columns_to_normalize = {
    'sbp': 218,
    'tobacco': 31.2,
    'ldl': 15.33,
    'adiposity': 42.49,
    'typea': 78,
    'obesity': 46.58,
    'alcohol': 147.19,
    'age': 64
}


def prepare_dataset(path):
    df = pd.read_csv(path)
    df['famhist'] = pd.Categorical(df['famhist'])
    df['famhist'] = df.famhist.cat.codes
    df.pop('row.names')

    for column, max in columns_to_normalize.items():
        df[column] = df[column] / max

    target = df.pop('chd')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    shuffled_dataset = dataset.shuffle(len(df)).batch(1).repeat()

    return shuffled_dataset, df


train_dataset, train_df = prepare_dataset(train_file_path)
test_dataset, test_df = prepare_dataset(test_file_path)


def get_compiled_model():
    model = keras.Sequential([
        keras.layers.Dense(9, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        keras.layers.Dropout(0.5),
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
