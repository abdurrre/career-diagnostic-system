import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
import wandb
from tracker import init_wandb
from wandb.integration.keras import WandbMetricsLogger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'data')
JSONL_PATH = os.path.join(BASE_DIR, 'data', 'dataset_ner_skills_bersih.jsonl')

MAX_LEN = 256  # cap token length to handle long cv texts
BATCH_SIZE = 32
EPOCHS = 30    

print("Memuat Kamus dan Metadata")
with open(os.path.join(ARTIFACTS_DIR, 'ner_tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, 'dataset-metadata.json'), 'r') as f:
    metadata = json.load(f)

VOCAB_SIZE = metadata['ner_vocab_size']
NUM_CLASSES = metadata['ner_num_classes']

def data_generator():
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            tags = data.get("ner_tags", [])

            # convert tokens to sequences
            seq = tokenizer.texts_to_sequences([tokens])[0]

            seq = seq[:MAX_LEN]
            tags = tags[:MAX_LEN]

            seq_padded = pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')[0]
            tags_padded = pad_sequences([tags], maxlen=MAX_LEN, padding='post', truncating='post', value=0)[0]

            yield seq_padded, tags_padded

# build tf.data.dataset pipeline
print("tf.data.Dataset")
train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32), # input: X (tokens)
        tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32)  # output: y (BIO tags)
    )
)

# optimize dataset pipeline
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# initialize model and start training
print("Inisialisasi Arsitektur NER")
from architectures import NERModel

model = NERModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=128,
    rnn_units=64,
    num_classes=NUM_CLASSES
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

if __name__ == "__main__":
    
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    model_save_path = os.path.join(BASE_DIR, 'models', 'ner_model.keras')
    
    init_wandb(
        project_name="career-diagnostic-system",
        run_name="ner-model-training",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "vocab_size": VOCAB_SIZE,
            "num_classes": NUM_CLASSES,
            "embedding_dim": 128,
            "rnn_units": 64
        }
    )

    # early stopping callback
    early_stopping = EarlyStopping(
        monitor='loss', 
        patience=3, 
        restore_best_weights=True, 
        verbose=1
    )

    # checkpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    
    wandb_logger = WandbMetricsLogger()
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, model_checkpoint, wandb_logger]
    )

    wandb.finish()
    print(f"\nTraining selesai.")
