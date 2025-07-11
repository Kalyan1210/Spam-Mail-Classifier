#!/usr/bin/env python3
"""
train.py

Reproducible training script for the spam classifier.
- loads prepared data from .npz
- loads tokenizer and label encoder
- sets random seeds for reproducibility
- builds a simple neural network
- trains and evaluates on validation set
- saves the trained model
"""
import argparse
import os
import random
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout


def set_seeds(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(data_dir: str):
    """Load preprocessed data and artifacts."""
    # Load arrays
    npz = np.load(os.path.join(data_dir, 'data.npz'))
    X_train = npz['X_train']
    y_train = npz['y_train']
    X_val = npz['X_val']
    y_val = npz['y_val']
    # Load tokenizer and label encoder
    with open(os.path.join(data_dir, 'tokenizer.pickle'), 'rb') as f_tok:
        tokenizer = pickle.load(f_tok)
    with open(os.path.join(data_dir, 'label_encoder.pickle'), 'rb') as f_le:
        label_encoder = pickle.load(f_le)

    return X_train, y_train, X_val, y_val, tokenizer, label_encoder


def build_model(vocab_size: int, embed_dim: int, input_length: int,
                dropout_rate: float, dense_units: int, learning_rate: float) -> tf.keras.Model:
    """Constructs and compiles the Keras model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length),
        GlobalAveragePooling1D(),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def main(data_dir: str, model_out: str, embed_dim: int, dense_units: int,
         dropout_rate: float, learning_rate: float, epochs: int, batch_size: int,
         seed: int):
    # 1. Set seeds
    set_seeds(seed)

    # 2. Load data
    X_train, y_train, X_val, y_val, tokenizer, label_encoder = load_data(data_dir)
    vocab_size = len(tokenizer.word_index) + 1
    input_length = X_train.shape[1]

    # 3. Build model
    model = build_model(vocab_size, embed_dim, input_length,
                        dropout_rate, dense_units, learning_rate)

    # 4. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 5. Save model
    model.save(model_out)
    print(f"Model saved to {model_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the spam classifier.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing data.npz and pickled artifacts')
    parser.add_argument('--model-out', default='spam_model.keras',
                        help='Output path for the trained model')
    parser.add_argument('--embed-dim', type=int, default=16,
                        help='Dimension of the embedding layer')
    parser.add_argument('--dense-units', type=int, default=24,
                        help='Number of units in the dense layer')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate after pooling')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    main(
        args.data_dir,
        args.model_out,
        args.embed_dim,
        args.dense_units,
        args.dropout_rate,
        args.learning_rate,
        args.epochs,
        args.batch_size,
        args.seed
    )
