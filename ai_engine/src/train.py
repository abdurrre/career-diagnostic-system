import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from architectures import GapModel
from custom_metrics import weighted_binary_crossentropy
from tracker import init_wandb, ElitePerformanceTracker
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

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
    X = job_encoder.transform(df['job_title'])
    df['skill_list'] = df['cleaned_skills'].apply(lambda x: [s.strip() for s in str(x).split(',')])
    y = skill_binarizer.transform(df['skill_list'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def run_training():
    DATA_DIR = 'ai_engine/data'
    CSV_PATH = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/clean data/final_ready_it_jobs (1).csv'

    metadata, job_encoder, skill_binarizer = load_artifacts(DATA_DIR)
    X_train, X_val, y_train, y_val = prepare_gap_data(CSV_PATH, job_encoder, skill_binarizer)

    print(f"Bentuk Data X_train: {X_train.shape}, y_train: {y_train.shape}")

    model = GapModel(
        num_professions=metadata['num_professions'],
        num_skills=metadata['num_skills'],
        embedding_dim=64
    )

    pos_counts = y_train.sum(axis=0) 
    neg_counts = len(y_train) - pos_counts
    skill_weights = neg_counts / (pos_counts + 1e-5)
    skill_weights = np.clip(skill_weights, a_min=1.0, a_max=100.0)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_binary_crossentropy(skill_weights=skill_weights), 
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    )

    init_wandb(project_name="career-diagnostic", run_name="gap_model_final_thresh_0.2")
    
    elite_tracker = ElitePerformanceTracker(validation_data=(X_val, y_val), threshold=0.2)
    
    wandb_logger = WandbMetricsLogger()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='ai_engine/models/gap_model.keras',
        save_best_only=True,
        monitor='val_auc',
        mode='max'
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[elite_tracker, wandb_logger, checkpoint]
    )

if __name__ == "__main__":
    run_training()
