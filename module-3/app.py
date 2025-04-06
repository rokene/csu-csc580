from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# Print TensorFlow version
print(tf.__version__)

# Fetch the Auto MPG dataset
auto_mpg = fetch_ucirepo(id=9)

# Extract features and labels
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine into a single DataFrame for easier handling
dataset = pd.concat([X, y], axis=1)

print(dataset.tail())  # Show last few rows

# Train/test split
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=0)

# Separate labels (target variable)
train_labels = train_dataset.pop('mpg')
test_labels = test_dataset.pop('mpg')

# Generate summary statistics
train_stats = train_dataset.describe().transpose()

# Normalize the features
def normalize(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = normalize(train_dataset)
normed_test_data = normalize(test_dataset)

# Visualize relationships between some features
sns.pairplot(train_dataset[["cylinders", "displacement", "weight", "acceleration"]], diag_kind="kde")
plt.show(block=False)

# Build the regression model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )
    return model

model = build_model()

# Display the model architecture
model.summary()

# Try model prediction on a small batch
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# Train the model
EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

# Review training history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# Plot Mean Absolute Error
plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show(block=False)

# Plot Mean Squared Error
plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG²]')
plt.show(block=False)

# Evaluate on test dataset (optional)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print(f"Testing set Mean Abs Error: {mae:.2f} MPG")
