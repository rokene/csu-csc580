#!/usr/bin/env python3
"""
cifar_cnn.py

Train or run inference with a simple CIFAR-10 CNN in TensorFlow 2.x.
"""

import os
import argparse
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
CLASS_NAMES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------
def enable_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        logging.info(f"Enabled memory growth on {len(gpus)} GPU(s)")


def get_datasets(batch_size: int, augment: bool):
    """Load CIFAR-10, normalize to [0,1], build tf.data pipelines."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    def _augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10_000)
    if augment:
        train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


def build_model(num_classes: int = 10) -> tf.keras.Model:
    """Constructs the CNN architecture."""
    model = models.Sequential([
        layers.Input((32,32,3)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ])
    return model


# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
def train(args):
    logging.info("Starting training mode")
    train_ds, val_ds = get_datasets(args.batch_size, augment=True)

    model = build_model(num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(args.log_dir, timestamp)
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            filepath=args.model_path, save_best_only=True),
        callbacks.TensorBoard(log_dir=logdir)
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=cb_list,
        verbose=2
    )

    # final evaluation
    test_loss, test_acc = model.evaluate(val_ds, verbose=0)
    logging.info(f"Validation accuracy: {test_acc:.4f}")

    logging.info(f"Model saved to: {args.model_path}")


# ------------------------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------------------------
def infer(args):
    logging.info("Starting inference mode")
    model = tf.keras.models.load_model(args.model_path)
    logging.info(f"Loaded model from {args.model_path}")

    # load test set if no image_path
    if args.image_path:
        img = tf.keras.preprocessing.image.load_img(
            args.image_path, target_size=(32,32))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    else:
        # use CIFAR-10 test data
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        img = x_test.astype("float32") / 255.0
        args.image_index = np.clip(args.image_index, 0, len(img)-1)
        img = img[args.image_index]
        true_label = int(y_test[args.image_index][0])
        print(f"True label: {CLASS_NAMES[true_label]}")

    # predict
    probs = model.predict(img[None, ...])[0]
    preds = np.argsort(probs)[::-1][:3]

    print("Top-3 Predictions:")
    for idx in preds:
        print(f"  {CLASS_NAMES[idx]:<6} — {probs[idx]*100:5.2f}%")

    # show image
    try:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    except ImportError:
        logging.warning("matplotlib not installed, skipping image display")


# ------------------------------------------------------------------------------
# MAIN / ARGPARSE
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train or infer on CIFAR-10 with a simple CNN"
    )
    p.add_argument(
        "--mode", choices=["train", "infer"], required=True,
        help="Operation mode: train or infer"
    )
    p.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training or evaluation"
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (train mode only)"
    )
    p.add_argument(
        "--model_path", type=str, default="cifar_cnn.h5",
        help="Where to save/load the model"
    )
    p.add_argument(
        "--log_dir", type=str, default="logs",
        help="TensorBoard log directory (train mode only)"
    )
    p.add_argument(
        "--image_index", type=int, default=0,
        help="Test‐set image index for inference (infer mode only)"
    )
    p.add_argument(
        "--image_path", type=str, default=None,
        help="Path to an image file to run inference on (infer mode only)"
    )
    return p.parse_args()


def main():
    # logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    args = parse_args()
    enable_gpu_memory_growth()

    if args.mode == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
