import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CSV_PATH = 'final_ready_it_jobs.csv'
ARTIFACTS_DIR = 'ai_engine/data'
MODEL_SAVE_PATH = 'ai_engine/models/scoring_model.keras'

def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, 'dataset-metadata.json'), 'r') as f:
        metadata = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'job_encoder.pkl'), 'rb') as f:
        job_encoder = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'skill_binarizer.pkl'), 'rb') as f:
        skill_binarizer = pickle.load(f)
    return metadata, job_encoder, skill_binarizer

def build_knowledge_base(df):
    df['skill_list'] = df['cleaned_skills'].apply(lambda x: [s.strip() for s in str(x).split(',')])
    knowledge_base = {}
    
    for prof in df['job_title'].unique():
        prof_data = df[df['job_title'] == prof]
        all_skills = set()
        for skills in prof_data['skill_list']:
            all_skills.update(skills)
        knowledge_base[prof] = all_skills
        
    kb_serializable = {k: sorted(list(v)) for k, v in knowledge_base.items()}
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    with open(os.path.join(ARTIFACTS_DIR, 'knowledge_base.json'), 'w') as f:
        json.dump(kb_serializable, f, indent=4)
        
    return knowledge_base

def prepare_training_data(df, knowledge_base, skill_binarizer):
    X_list, y_list = [], []
    
    for _, row in df.iterrows():
        profession = row['job_title']
        user_skills = set(row['skill_list'])
        required = knowledge_base[profession]

        if not required:
            continue

        user_vec = skill_binarizer.transform([list(user_skills)])[0].astype(np.float32)
        req_vec = skill_binarizer.transform([list(required)])[0].astype(np.float32)

        intersection = user_skills & required
        coverage = len(intersection) / len(required)

        feature = np.concatenate([user_vec, req_vec])
        X_list.append(feature)
        y_list.append(coverage)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

class ScoringModel(tf.keras.Model):
    def __init__(self, num_skills, **kwargs):
        super(ScoringModel, self).__init__(**kwargs)
        self.num_skills = num_skills
        self.dense1 = Dense(256, activation='relu')
        self.bn1 = BatchNormalization()
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(64, activation='relu')
        self.out_layer = Dense(1, activation='sigmoid', name="score_out")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.out_layer(x)

    def get_config(self):
        config = super(ScoringModel, self).get_config()
        config.update({"num_skills": self.num_skills})
        return config

def main():
    metadata, job_encoder, skill_binarizer = load_artifacts()
    df = pd.read_csv(CSV_PATH)
    knowledge_base = build_knowledge_base(df)
    
    X, y = prepare_training_data(df, knowledge_base, skill_binarizer)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = ScoringModel(num_skills=metadata['num_skills'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop]
    )

    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation MAE: {val_mae:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
