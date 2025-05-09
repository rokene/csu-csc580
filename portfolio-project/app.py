import os
import argparse
import datetime
import numpy as np
from numpy.random import randint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


def define_models(n_input, n_output, n_units):
    """
    Create training and inference encoder-decoder models.
    """
    # Encoder
    encoder_inputs = Input(shape=(None, n_input), name='encoder_inputs')
    encoder_lstm = LSTM(n_units, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder for training
    decoder_inputs = Input(shape=(None, n_output), name='decoder_inputs')
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='training_model')

    # Encoder model for inference
    encoder_model = Model(encoder_inputs, encoder_states, name='inference_encoder')

    # Decoder model for inference
    dec_state_input_h = Input(shape=(n_units,), name='dec_input_h')
    dec_state_input_c = Input(shape=(n_units,), name='dec_input_c')
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=dec_states_inputs)
    dec_outputs = decoder_dense(dec_outputs)
    decoder_model = Model(
        [decoder_inputs] + dec_states_inputs,
        [dec_outputs, state_h, state_c],
        name='inference_decoder'
    )

    return train_model, encoder_model, decoder_model


def generate_sequence(length, n_unique):
    """Generate a random integer sequence (1..n_unique), reserving 0 as start/pad."""
    return [randint(1, n_unique) for _ in range(length)]


def get_dataset(n_in, n_out, cardinality, n_samples):
    """Build source, shifted target input, and actual target output arrays."""
    X1, X2, y = [], [], []
    for _ in range(n_samples):
        src = generate_sequence(n_in, cardinality)
        tgt = src[:n_out][::-1]
        tgt_in = [0] + tgt[:-1]

        # one-hot encode
        src_enc = to_categorical(src, num_classes=cardinality)
        tgt_enc = to_categorical(tgt, num_classes=cardinality)
        tgt_in_enc = to_categorical(tgt_in, num_classes=cardinality)

        X1.append(src_enc)
        X2.append(tgt_in_enc)
        y.append(tgt_enc)

    return np.array(X1), np.array(X2), np.array(y)


def one_hot_decode(encoded_seq):
    """Convert one-hot encoded vectors back to integer sequence."""
    return [int(np.argmax(vec)) for vec in encoded_seq]


def predict_sequence(inf_enc, inf_dec, source, n_steps, cardinality):
    """Generate output sequence step-by-step for a source using inference models."""
    state = inf_enc.predict(source)
    target_seq = np.zeros((1, 1, cardinality))  # start token
    output = []

    for _ in range(n_steps):
        yhat, h, c = inf_dec.predict([target_seq] + state)
        output.append(yhat[0, 0, :])
        state = [h, c]
        target_seq = yhat

    return np.array(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq2Seq encoder-decoder')
    parser.add_argument('--mode', choices=['train', 'infer'], default='train',
                        help='train models or run inference')
    parser.add_argument('--samples', type=int, default=10000,
                        help='number of training samples (train mode only)')
    parser.add_argument('--n-features', type=int, default=51,
                        help='vocabulary size (including start/pad token)')
    parser.add_argument('--n-steps-in', type=int, default=6,
                        help='input sequence length')
    parser.add_argument('--n-steps-out', type=int, default=3,
                        help='output sequence length')
    parser.add_argument('--units', type=int, default=128,
                        help='LSTM cell count')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='training epochs')
    parser.add_argument('--encoder-path', type=str, default='encoder_model.h5',
                        help='path to save/load encoder model')
    parser.add_argument('--decoder-path', type=str, default='decoder_model.h5',
                        help='path to save/load decoder model')
    args = parser.parse_args()

    n_features = args.n_features
    n_steps_in, n_steps_out = args.n_steps_in, args.n_steps_out
    units, batch_size, epochs = args.units, args.batch_size, args.epochs

    if args.mode == 'train':
        train_model, inf_enc, inf_dec = define_models(n_features, n_features, units)
        train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Set up TensorBoard logging
        log_dir = os.path.join('logs', 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

        X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, args.samples)
        train_model.fit(
            [X1, X2], y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tensorboard_cb],
            verbose=2
        )
        inf_enc.save(args.encoder_path)
        inf_dec.save(args.decoder_path)
        print(f'Training complete. Models saved to {args.encoder_path} and {args.decoder_path}.')
        print(f'TensorBoard logs written to: {log_dir}')

    else:
        if not os.path.exists(args.encoder_path) or not os.path.exists(args.decoder_path):
            raise FileNotFoundError('Models not found. Run with --mode train first.')

        inf_enc = load_model(args.encoder_path, compile=False)
        inf_dec = load_model(args.decoder_path, compile=False)

        total, correct = 100, 0
        print(f'Inference accuracy: # iterations {total}')
        for _ in range(total):
            X1s, _, ys = get_dataset(n_steps_in, n_steps_out, n_features, 1)
            yhat = predict_sequence(inf_enc, inf_dec, X1s, n_steps_out, n_features)
            if one_hot_decode(ys[0]) == one_hot_decode(yhat):
                correct += 1
        print(f'Inference accuracy: {correct/total*100:.2f}%')

        print('\nSample predictions:')
        for _ in range(10):
            X1s, _, ys = get_dataset(n_steps_in, n_steps_out, n_features, 1)
            yhat = predict_sequence(inf_enc, inf_dec, X1s, n_steps_out, n_features)
            print(f"X={one_hot_decode(X1s[0])} y={one_hot_decode(ys[0])} yhat={one_hot_decode(yhat)}")

        # Interactive user input
        print('\nEnter your own sequences for prediction (type "exit" to quit):')
        while True:
            user_in = input(f'Enter {n_steps_in} integers (1-{n_features-1}) separated by spaces: ')
            if user_in.lower() in ('exit', 'quit'):
                break
            try:
                tokens = list(map(int, user_in.strip().split()))
                if len(tokens) != n_steps_in or any(t < 1 or t >= n_features for t in tokens):
                    raise ValueError
            except ValueError:
                print(f'Invalid input. Please enter {n_steps_in} integers between 1 and {n_features-1}.')
                continue
            src_enc = to_categorical(tokens, num_classes=n_features).reshape(1, n_steps_in, n_features)
            yhat = predict_sequence(inf_enc, inf_dec, src_enc, n_steps_out, n_features)
            pred = one_hot_decode(yhat)
            print(f'Predicted reversed subsequence: {pred}\n')
