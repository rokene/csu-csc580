import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.gridspec as gridspec

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_model.keras")

def test_learning_rates():
    x_train, y_train, x_test, y_test = load_data()

    # Split validation set
    x_val = x_train[-5000:]
    y_val = y_train[-5000:]
    x_train = x_train[:-5000]
    y_train = y_train[:-5000]

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    test_accuracies = []

    print("📊 Testing different learning rates...\n")
    for lr in learning_rates:
        print(f"🔧 Training with learning rate: {lr}")
        model = build_model(learning_rate=lr)

        model.fit(
            x_train, y_train,
            epochs=10,  # shorter training for quick comparison
            batch_size=100,
            validation_data=(x_val, y_val),
            verbose=0
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        test_accuracies.append(test_acc)
        print(f"   → Test Accuracy: {test_acc:.4f}\n")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, test_accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test Accuracy")
    plt.title("Effect of Learning Rate on Test Accuracy")
    plt.grid(True)
    plt.savefig("lr_test_results.png", dpi=300)
    plt.show()
    print("📈 Learning rate test results saved as lr_test_results.png")

# Load and preprocess the data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

# Build model
def build_model(hidden_units=512, activation='relu', learning_rate=0.5, optimizer_type='sgd'):
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(hidden_units, activation=activation),
        layers.Dense(10, activation='softmax')
    ])

    # Choose optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_type.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train and save the model
def train_model():
    x_train, y_train, x_test, y_test = load_data()

    # Split validation set
    x_val = x_train[-5000:]
    y_val = y_train[-5000:]
    x_train = x_train[:-5000]
    y_train = y_train[:-5000]

    model = build_model()

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=100,
        validation_data=(x_val, y_val)
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"🧪 Final Test Accuracy: {test_acc:.4f}")

    # Predictions and misclassifieds
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    misclassified_indices = np.where(predicted_labels != true_labels)[0]
    print(f"❌ Misclassified samples found: {len(misclassified_indices)}")

    # Random test samples 4
    random_indices = np.random.choice(len(x_test), 4, replace=False)

    # Misclassified samples 4
    misclassified_samples = np.random.choice(
        misclassified_indices, min(4, len(misclassified_indices)), replace=False
    ) if len(misclassified_indices) > 0 else []

    # Plot layout
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 6)  # 3 rows, 6 columns

    # === Left Side: Accuracy Graph ===
    ax1 = fig.add_subplot(gs[:, :2])  # full height, first 2 columns
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Training vs. Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # === Top-Right: Random Samples ===
    for i, idx in enumerate(random_indices):
        row = 0
        col = i % 4 + 2
        ax = fig.add_subplot(gs[row, col])
        image = x_test[idx].reshape(28, 28) * 255.0
        label = int(true_labels[idx])
        pred_label = int(predicted_labels[idx])
        confidence = predictions[idx][pred_label] * 100
        color = 'green' if label == pred_label else 'red'
        ax.imshow(image, cmap='gray_r')
        ax.set_title(f"{label} ({confidence:.0f}%)", fontsize=8, color=color, fontfamily='monospace')
        ax.axis('off')

    # === Bottom-Right: Misclassified Samples ===
    for i, idx in enumerate(misclassified_samples):
        row = 1
        col = i % 4 + 2
        ax = fig.add_subplot(gs[row + 1, col])  # row 2 (index 2)
        image = x_test[idx].reshape(28, 28) * 255.0
        true_label = int(true_labels[idx])
        pred_label = int(predicted_labels[idx])
        confidence = predictions[idx][pred_label] * 100
        title_text = f"Truth:{true_label} Predicted:{pred_label} ({confidence:.0f}%)"
        ax.imshow(image, cmap='gray_r')
        ax.set_title(title_text, fontsize=8, color='red', fontfamily='monospace')
        ax.axis('off')

    fig.suptitle("Accuracy Graph, Random Samples, and Misclassified Samples", fontsize=16)
    plt.subplots_adjust(hspace=0.8, wspace=0.4)  # More space between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("training_results.png", dpi=300)
    plt.show()
    print("📷 Plot saved to training_results.png")

# Load model and run inference
def run_inference():
    _, _, x_test, y_test = load_data()
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found. Please train the model first.")
        return

    model = models.load_model(MODEL_PATH)
    print("✅ Model loaded.")

    # Pick 8 random samples to predict and display
    indices = np.random.choice(len(x_test), 8, replace=False)
    predictions = model.predict(x_test[indices])

    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        image = x_test[idx].reshape(28, 28) * 255.0
        true_label = np.argmax(y_test[idx])
        predicted_label = np.argmax(predictions[i])

        plt.subplot(2, 4, i + 1)
        plt.imshow(image, cmap='gray_r')
        plt.title(f"True: {true_label}, Pred: {predicted_label}")
        plt.axis('off')

    plt.suptitle("Inference Samples")
    plt.tight_layout()
    plt.show()

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Classifier")
    parser.add_argument("--mode", choices=["train", "infer", "lrtest"], required=True, help="Run mode: train, infer, or lrtest")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "infer":
        run_inference()
    elif args.mode == "lrtest":
        test_learning_rates()
