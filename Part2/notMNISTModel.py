import tensorflow as tf
from tensorflow import keras
import numpy as np

print("--Get data--")
with np.load("Part2/notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0

print("--Make model--")
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation='sigmoid')
])
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=1, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc * 100:.1f}%")

# Save
# model.save("Part2/notMNIST.h5")