import tensorflow as tf
from tensorflow import keras
import numpy as np

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0

print("--Make model--")
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=6, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuracy: {model_acc * 100:.1f}%")

# Save
model.save("notMNIST.h5")