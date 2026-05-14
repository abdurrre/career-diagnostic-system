import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import clean_job_title
import wandb
from tracker import init_wandb, ElitePerformanceTracker
from wandb.integration.keras import WandbMetricsLogger

from architectures import GapModel
from custom_metrics import weighted_binary_crossentropy

import keras_tuner as kt

def load_artifacts(data_dir):
    with open(os.path.join(data_dir, 'dataset-metadata.json'), 'r') as f:
        metadata = json.load(f)
    with open(os.path.join(data_dir, 'job_encoder.pkl'), 'rb') as f:
        job_encoder = pickle.load(f)
    with open(os.path.join(data_dir, 'skill_binarizer.pkl'), 'rb') as f:
        skill_binarizer = pickle.load(f)
    return metadata, job_encoder, skill_binarizer

def prepare_gap_data(csv_path, job_encoder, skill_binarizer):
    df = pd.read_csv(csv_path)
    df['job_title'] = df['job_title'].apply(clean_job_title)
    df = df[df['job_title'] != 'DROP'].reset_index(drop=True)
    
    X = job_encoder.transform(df['job_title'])
    df['skill_list'] = df['cleaned_skills'].apply(lambda x: [s.strip() for s in str(x).split(',')])
    y = skill_binarizer.transform(df['skill_list'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def run_training():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CSV_PATH = os.path.join(DATA_DIR, 'final_ready_it_jobs (2).csv')

    metadata, job_encoder, skill_binarizer = load_artifacts(DATA_DIR)
    X_train, X_val, y_train, y_val = prepare_gap_data(CSV_PATH, job_encoder, skill_binarizer)

    print(f"Bentuk Data X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Handle class imbalance — y_train is binary (0/1), not gap categories
    from sklearn.utils.class_weight import compute_class_weight
    y_train_flat = y_train.flatten()
    
    weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
    
    # Boost positive class (very sparse) so probabilities can cross inference thresholds
    global_pos_weight = weights[1] / weights[0]
    skill_weights = np.full((y_train.shape[1],), global_pos_weight)

    def build_gap_model(hp):
        units = hp.Int('units', min_value=64, max_value=512, step=64)
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model = GapModel(
            num_professions=metadata['num_professions'],
            num_skills=metadata['num_skills'],
            embedding_dim=64,
            dense_units=units,
            dropout_rate=dropout
        )
        
        # Explicit build so KerasTuner can manage weights between trials
        model.build(input_shape=(None,))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=weighted_binary_crossentropy(skill_weights=skill_weights),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
        )
        return model

    # Use val_auc — val_accuracy is biased toward majority class (0) and causes model collapse
    tuner = kt.Hyperband(
        build_gap_model,
        objective=kt.Objective('val_auc', direction='max'),
        max_epochs=50,
        factor=3,
        directory=os.path.join(BASE_DIR, 'tuning'),
        project_name='gap_model_tuning_balanced'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    print("\n[INFO] Memulai Hyperparameter Tuning (Hyperband)...")
    
    init_wandb(
        project_name="career-diagnostic-system",
        run_name="gap-model-training",
        config={
            "epochs": 50,
            "batch_size": 32,
            "tuner": "Hyperband",
            "max_epochs": 50,
            "objective": "val_auc"
        }
    )

    wandb_logger = WandbMetricsLogger()
    elite_tracker = ElitePerformanceTracker(validation_data=(X_val, y_val))
    
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, wandb_logger, elite_tracker]
    )

    print("\n[INFO] Extracting best model...")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best HPs -> Units: {best_hps.get('units')}, Dropout: {best_hps.get('dropout')}, LR: {best_hps.get('learning_rate')}")

    best_model = tuner.get_best_models(num_models=1)[0]
    
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

    model_save_path = os.path.join(BASE_DIR, 'models', 'gap_model.keras')
    best_model.save(model_save_path)
    
    wandb.finish()
    print(f"\n[INFO] Model saved to: {model_save_path}")

if __name__ == "__main__":
    run_training()
