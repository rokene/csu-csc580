from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
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
import os
import sys

MODEL_PATH = 'fuel_efficiency_model.keras'
PLOTS_DIR = 'plots'

def load_data():
    # Fetch the Auto MPG dataset
    auto_mpg = fetch_ucirepo(id=9)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets
    dataset = pd.concat([X, y], axis=1)
    return dataset

def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
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

def train():
    dataset = load_data()
    print(dataset.tail())

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=0)

    train_labels = train_dataset.pop('mpg')
    test_labels = test_dataset.pop('mpg')

    train_stats = train_dataset.describe().transpose()

    def normalize(x):
        return (x - train_stats['mean']) / train_stats['std']

    # Make normalize globally available
    globals()['normalize'] = normalize

    normed_train_data = normalize(train_dataset)
    normed_test_data = normalize(test_dataset)

    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Visualize relationships between features
    sns.pairplot(train_dataset[["cylinders", "displacement", "weight", "acceleration"]], diag_kind="kde")
    plt.savefig(os.path.join(PLOTS_DIR, 'pairplot.png'))
    plt.show()

    model = build_model(len(train_dataset.keys()))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    EPOCHS = 1000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS,
        batch_size=64,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop, tfdocs.modeling.EpochDots()]
    )

    # Save training history
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    print("History details:")
    print(hist.tail())

    # Create plotter
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

    # Plot Mean Absolute Error
    plotter.plot({'Basic': history}, metric="mae")
    plt.yscale('log')
    plt.ylabel('Mean Absolute Error [MPG]')
    plt.xlabel('Epoch')
    plt.title('Training MAE over Epochs (Log Scale)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'mae_plot.png'))
    plt.show()

    # Plot Mean Squared Error
    plotter.plot({'Basic': history}, metric="mse")
    plt.yscale('log')
    plt.ylabel('Mean Squared Error [MPG²]')
    plt.xlabel('Epoch')
    plt.title('Training MSE over Epochs (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(PLOTS_DIR, 'mse_plot.png'))
    plt.show()

    # Evaluate the model
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print(f"Testing set Mean Abs Error: {mae:.2f} MPG")

    # Save the model
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

def infer():
    # Load the trained model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load dataset to recreate normalization
    dataset = load_data()
    dataset.pop('mpg')
    train_stats = dataset.describe().transpose()

    def normalize(x):
        return (x - train_stats['mean']) / train_stats['std']

    # Inference loop
    def predict_efficiency():
        print("\n--- Fuel Efficiency Prediction ---")
        while True:
            try:
                cylinders = input("Enter number of cylinders (or type 'exit' to quit): ")
                if cylinders.lower() == 'exit':
                    print("Exiting inference mode.")
                    break
                cylinders = float(cylinders)

                displacement = float(input("Enter engine displacement (in cubic inches): "))
                horsepower = float(input("Enter horsepower: "))
                weight = float(input("Enter weight (in lbs): "))
                acceleration = float(input("Enter acceleration (0-60 mph time): "))
                model_year = float(input("Enter model year (e.g., 76 for 1976): "))
                origin = float(input("Enter origin (1=USA, 2=Europe, 3=Japan): "))
            except ValueError:
                print("Invalid input. Please enter numerical values.")
                continue

            user_input = pd.DataFrame({
                'cylinders': [cylinders],
                'displacement': [displacement],
                'horsepower': [horsepower],
                'weight': [weight],
                'acceleration': [acceleration],
                'model_year': [model_year],
                'origin': [origin]
            })

            user_input_normalized = normalize(user_input)
            prediction = model.predict(user_input_normalized)
            print(f"\nEstimated fuel efficiency (MPG): {prediction[0][0]:.2f}\n")

    predict_efficiency()

def main():
    parser = argparse.ArgumentParser(description="Fuel Efficiency Prediction Model")
    parser.add_argument('--mode', choices=['train', 'infer'], required=True, help="Mode to run the script in: train or infer")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'infer':
        infer()
    else:
        print("Invalid mode. Choose 'train' or 'infer'.")

if __name__ == '__main__':
    main()
