import pandas as pd
import json
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def build_artifacts():
    print("Membangun kamus data untuk Model AI")
    
    # Path ke CSV anak DS (Sesuaikan kalau beda)
    csv_path = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/clean data/final_ready_it_jobs (1).csv'
    output_dir = 'ai_engine/data'
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # 1. Bikin Job Encoder
    job_encoder = LabelEncoder()
    df['profession_id'] = job_encoder.fit_transform(df['job_title'])
    
    # Bikin Skill Binarizer
    df['skill_list'] = df['cleaned_skills'].apply(lambda x: [s.strip() for s in str(x).split(',')])
    skill_binarizer = MultiLabelBinarizer()
    skill_binarizer.fit(df['skill_list'])
    
    # Bikin Tokenizer
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_skills'].astype(str))
    
    # Simpan Metadata
    metadata = {
        "vocab_size": len(tokenizer.word_index) + 1,
        "num_professions": len(job_encoder.classes_),
        "num_skills": len(skill_binarizer.classes_),
        "max_length": 128
    }
    
    with open(os.path.join(output_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    with open(os.path.join(output_dir, 'job_encoder.pkl'), 'wb') as f:
        pickle.dump(job_encoder, f)
        
    with open(os.path.join(output_dir, 'skill_binarizer.pkl'), 'wb') as f:
        pickle.dump(skill_binarizer, f)
        
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    build_artifacts()
