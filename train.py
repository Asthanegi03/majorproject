# train.py - Main training script for Audio Deepfake Detection System

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import librosa

# Configure GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Parameters
DATA_DIR = 'dataset'  # Directory containing real and fake audio files
SAMPLE_RATE = 22050
MAX_AUDIO_LENGTH = 5  # in seconds
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 50
TEST_SPLIT = 0.2
RANDOM_SEED = 42
MODELS_DIR = 'models'

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def extract_features(audio_path):
    """Extract audio features from a file"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Trim or pad to fixed length
        if len(y) > SAMPLE_RATE * MAX_AUDIO_LENGTH:
            y = y[:int(SAMPLE_RATE * MAX_AUDIO_LENGTH)]
        else:
            y = np.pad(y, (0, int(SAMPLE_RATE * MAX_AUDIO_LENGTH) - len(y)))
        
        # Extract features
        features = {}
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
        features['mfcc_delta_std'] = np.std(mfcc_delta, axis=1)
        features['mfcc_delta2_mean'] = np.mean(mfcc_delta2, axis=1)
        features['mfcc_delta2_std'] = np.std(mfcc_delta2, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
        features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)
        
        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmonic_mean'] = np.mean(np.abs(y_harmonic))
        features['harmonic_std'] = np.std(np.abs(y_harmonic))
        features['percussive_mean'] = np.mean(np.abs(y_percussive))
        features['percussive_std'] = np.std(np.abs(y_percussive))
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Flattening all features into a single vector
        feature_vector = []
        for feature_name in sorted(features.keys()):
            feature_value = features[feature_name]
            if isinstance(feature_value, np.ndarray):
                feature_vector.extend(feature_value)
            else:
                feature_vector.append(feature_value)
        
        return np.array(feature_vector)
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {str(e)}")
        return None

def load_dataset(data_dir):
    """Load and prepare dataset from directory structure"""
    # Expected structure:
    # data_dir/
    #   real/
    #     file1.wav
    #     file2.wav
    #   fake/
    #     file1.wav
    #     file2.wav
    
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))]
    
    print(f"Found {len(real_files)} real audio files and {len(fake_files)} fake audio files")
    
    # Extract features
    X = []
    y = []
    
    print("Extracting features from real audio files...")
    for i, file_path in enumerate(real_files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(real_files)}")
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 for real
    
    print("Extracting features from fake audio files...")
    for i, file_path in enumerate(fake_files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(fake_files)}")
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 for fake
    
    return np.array(X), np.array(y)

def create_ann_model(input_shape):
    """Create ANN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_cnn_model(input_shape):
    """Create CNN model (requires reshaping input)"""
    # Reshape input to 2D for CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_rnn_model(input_shape):
    """Create RNN model (requires reshaping input)"""
    # Reshape input to 2D for RNN
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_lstm_model(input_shape):
    """Create LSTM model (requires reshaping input)"""
    # Reshape input to 2D for LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_gan_detector(input_shape):
    """Create specialized GAN detector model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train a model and save it"""
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print(f"Training {model_name} model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return metrics

def main():
    print("Loading dataset...")
    X, y = load_dataset(DATA_DIR)
    
    if len(X) == 0:
        print("No data could be loaded. Please check your dataset directory.")
        return
    
    print(f"Dataset loaded: {len(X)} samples, {len(X[0])} features")
    print(f"Class distribution: {np.sum(y == 0)} real, {np.sum(y == 1)} fake")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    
    input_shape = X_train.shape[1]
    print(f"Input shape: {input_shape}")
    
    # Train models sequentially to manage memory
    metrics = {}
    
    # Train ANN
    model = create_ann_model(input_shape)
    metrics['ann'] = train_model(model, X_train, y_train, X_test, y_test, 'ann')
    del model
    
    # Train CNN
    model = create_cnn_model(input_shape)
    metrics['cnn'] = train_model(model, X_train, y_train, X_test, y_test, 'cnn')
    del model
    
    # Train RNN
    model = create_rnn_model(input_shape)
    metrics['rnn'] = train_model(model, X_train, y_train, X_test, y_test, 'rnn')
    del model
    
    # Train LSTM
    model = create_lstm_model(input_shape)
    metrics['lstm'] = train_model(model, X_train, y_train, X_test, y_test, 'lstm')
    del model
    
    # Train GAN detector
    model = create_gan_detector(input_shape)
    metrics['gan_detector'] = train_model(model, X_train, y_train, X_test, y_test, 'gan_detector')
    del model
    
    # Save metrics
    print("Training completed. Model metrics:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}: Accuracy = {model_metrics['accuracy']:.4f}, F1 = {model_metrics['f1_score']:.4f}")
    
    # Save metrics to file
    import json
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()