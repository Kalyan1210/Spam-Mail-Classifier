#!/usr/bin/env python3
"""
spam_classifier.py

Script for preparing the spam dataset:
- loads and cleans spam.csv
- encodes labels (ham=0, spam=1)
- tokenizes and pads text sequences
- splits into train/val/test sets
- saves data arrays in compressed .npz
- pickles tokenizer and label encoder
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def main(csv_path, output_dir, num_words, maxlen, test_size, val_size, seed):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and clean data
    df = pd.read_csv(csv_path, encoding='latin-1')
    # Drop unused columns if present
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore', inplace=True)

    # 2. Encode labels
    label_encoder = LabelEncoder()
    df['v1'] = label_encoder.fit_transform(df['v1'])  # ham -> 0, spam -> 1
    labels = df['v1'].values

    # 3. Tokenize text
    texts = df['v2'].astype(str).tolist()
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding='post', maxlen=maxlen)

    # 4. Split into train, val, test
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        padded, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )
    # Then split remaining into train and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_temp
    )

    # 5. Save prepared data
    np.savez_compressed(
        os.path.join(output_dir, 'data.npz'),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    # 6. Save tokenizer and label encoder
    with open(os.path.join(output_dir, 'tokenizer.pickle'), 'wb') as f_tok:
        pickle.dump(tokenizer, f_tok)
    with open(os.path.join(output_dir, 'label_encoder.pickle'), 'wb') as f_le:
        pickle.dump(label_encoder, f_le)

    print(f"Data preparation complete. Files saved under '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare and split spam dataset for model training."
    )
    parser.add_argument(
        '--csv', required=True,
        help='Path to the spam.csv file'
    )
    parser.add_argument(
        '--output-dir', default='data',
        help='Directory to save the processed data'
    )
    parser.add_argument(
        '--num-words', type=int, default=10000,
        help='Maximum vocabulary size for tokenization'
    )
    parser.add_argument(
        '--maxlen', type=int, default=100,
        help='Maximum sequence length for padding'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction of data to reserve for the test set'
    )
    parser.add_argument(
        '--val-size', type=float, default=0.1,
        help='Fraction of data to reserve for the validation set (of remaining data)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    main(
        args.csv,
        args.output_dir,
        args.num_words,
        args.maxlen,
        args.test_size,
        args.val_size,
        args.seed
    )
