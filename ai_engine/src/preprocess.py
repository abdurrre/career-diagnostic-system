import pandas as pd
import json
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


def build_gap_artifacts():
    print("Membangun kamus data untuk Gap Model")
    csv_path = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/clean data/final_ready_it_jobs (2).csv'

    output_dir = 'ai_engine/data'
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    job_encoder = LabelEncoder()
    df['profession_id'] = job_encoder.fit_transform(df['job_title'])
    
    df['skill_list'] = df['cleaned_skills'].apply(lambda x: [s.strip() for s in str(x).split(',')])
    skill_binarizer = MultiLabelBinarizer()
    skill_binarizer.fit(df['skill_list'])
    
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_skills'].astype(str))
    
    # Simpan Metadata Gap Model
    metadata = {
        "vocab_size": len(tokenizer.word_index) + 1,
        "num_professions": len(job_encoder.classes_),
        "num_skills": len(skill_binarizer.classes_),
        "max_length": 128
    }
    
    # Amankan file lama jika ada, biar bisa di-update
    metadata_path = os.path.join(output_dir, 'dataset-metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            existing_data = json.load(f)
            existing_data.update(metadata)
            metadata = existing_data

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    with open(os.path.join(output_dir, 'job_encoder.pkl'), 'wb') as f:
        pickle.dump(job_encoder, f)
        
    with open(os.path.join(output_dir, 'skill_binarizer.pkl'), 'wb') as f:
        pickle.dump(skill_binarizer, f)
        
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

def build_ner_artifacts():
    print("\nMembangun kamus data untuk NER Model")
    jsonl_path = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/clean data/dataset_ner_skills_bersih.jsonl'
    output_dir = 'ai_engine/data'
    
    if not os.path.exists(jsonl_path):
        print(f"File {jsonl_path} tidak ditemukan")
        return

    all_cv_texts = []
    max_len = 0
    
    # Ekstrak data dari JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            all_cv_texts.append(tokens)
            
            if len(tokens) > max_len:
                max_len = len(tokens)
                
    # Bikin Tokenizer KHUSUS NER
    ner_tokenizer = Tokenizer(oov_token="<OOV>")
    ner_tokenizer.fit_on_texts(all_cv_texts)
    
    # Simpan NER Tokenizer biar nanti bisa dipake pas Inference
    with open(os.path.join(output_dir, 'ner_tokenizer.pkl'), 'wb') as f:
        pickle.dump(ner_tokenizer, f)
        
    # Update Metadata dengan info NER
    metadata_path = os.path.join(output_dir, 'dataset-metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
        
    metadata["ner_vocab_size"] = len(ner_tokenizer.word_index) + 1
    metadata["ner_max_length"] = max_len
    metadata["ner_num_classes"] = 3
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Berhasil memproses {len(all_cv_texts)} CV untuk NER.")
    print(f"NER Vocab Size: {metadata['ner_vocab_size']}")
    print(f"NER Max Length: {metadata['ner_max_length']}")

if __name__ == "__main__":
    build_gap_artifacts()
    build_ner_artifacts()
