import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
import argparse
import sys
import os

# Set random seeds
np.random.seed(456)
tf.random.set_seed(456)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train or Infer with a neural network on the Tox21 dataset")
parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help='Mode: train or infer')
args = parser.parse_args()

MODE = args.mode

print(f"Running in MODE={MODE}")

# Constants
input_dim = 1024
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100
dropout_rate = 0.5
model_save_path = "tox21_model.keras"

# Load the Tox21 dataset
_, (train, valid, test), _ = dc.molnet.load_tox21()

train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Remove extra tasks (only first target)
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

# Build the neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="InputLayer"),
        tf.keras.layers.Dense(n_hidden, activation='relu', name="HiddenLayer"),
        tf.keras.layers.Dropout(dropout_rate, name="DropoutLayer"),
        tf.keras.layers.Dense(1, activation='sigmoid', name="OutputLayer")
    ])
    return model

if MODE == 'train':
    model = create_model()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # TensorBoard callback
    log_dir = "logs/fcnet-tox21-tf2"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(
        train_X, train_y,
        validation_data=(valid_X, valid_y),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=[tensorboard_callback]
    )

    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate on validation set
    valid_y_pred = (model.predict(valid_X) > 0.5).astype(int).flatten()
    acc = accuracy_score(valid_y, valid_y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Plot the loss curve
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()

elif MODE == 'infer':
    if not os.path.exists(model_save_path):
        print(f"No trained model found at {model_save_path}. Please train the model first.")
        sys.exit(1)

    # Load the model
    model = tf.keras.models.load_model(model_save_path)
    print(f"Loaded model from {model_save_path}")

    # Predict on the validation set (you could also predict on test set)
    valid_y_pred_probs = model.predict(valid_X).flatten()
    valid_y_pred = (valid_y_pred_probs > 0.5).astype(int)

    # Print sample outputs
    for i in range(10):
        print(f"Sample {i}: Predicted={valid_y_pred[i]} (Prob={valid_y_pred_probs[i]:.4f}), True={int(valid_y[i])}")

    # Evaluate
    acc = accuracy_score(valid_y, valid_y_pred)
    print(f"Inference Validation Accuracy: {acc:.4f}")

else:
    print(f"Unsupported MODE: {MODE}")
    sys.exit(1)
