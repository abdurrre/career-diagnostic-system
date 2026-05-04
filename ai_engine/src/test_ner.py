import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

BASE_DIR = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/career-diagnostic-system/ai_engine'
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ner_model.keras')

MAX_LEN = 256 

with open(os.path.join(ARTIFACTS_DIR, 'ner_tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

from architectures import NERModel
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'NERModel': NERModel})

def extract_skills_from_text(text: str):
    print(f"\nMemproses Input: '{text}'")
    
    tokens = re.findall(r"[\w']+|[.,!?;]", text)
    seq = tokenizer.texts_to_sequences([tokens])[0]
    padded_seq = pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')
    
    pred = model.predict(padded_seq, verbose=0)
    
    pred_tags = np.argmax(pred, axis=-1)[0]
    
    extracted_skills = []
    current_skill = []
    
    for word, tag in zip(tokens, pred_tags[:len(tokens)]):
        if tag == 1: # B-SKILL 
            if current_skill:
                extracted_skills.append(" ".join(current_skill))
            current_skill = [word]
        elif tag == 2: # I-SKILL (Lanjutan skill)
            if current_skill:
                current_skill.append(word)
        else: # O (Bukan skill)
            if current_skill:
                extracted_skills.append(" ".join(current_skill))
                current_skill = []
            
    if current_skill:
        extracted_skills.append(" ".join(current_skill))
        
    return extracted_skills

if __name__ == "__main__":
    # Test Case 1: Bahasa Indonesia
    cv_text_1 = "Saya seorang mahasiswa yang mahir menggunakan Python, React, dan SQL untuk web development."
    hasil_1 = extract_skills_from_text(cv_text_1)
    print(f"-> Skill Terdeteksi: {hasil_1}\n")
    
    # Test Case 2: Bahasa Inggris Kompleks
    cv_text_2 = "Experience in building Machine Learning models with TensorFlow and deploying via Docker containers."
    hasil_2 = extract_skills_from_text(cv_text_2)
    print(f"Skill Terdeteksi: {hasil_2}\n")
    
    # Test Case 3: Ujian Ekstrem (Typo & Campur Aduk)
    cv_text_3 = "Pernah jadi asisten lab Linear Algebra, bisa ngoding pakai C++ dan pake Pandas buat data."
    hasil_3 = extract_skills_from_text(cv_text_3)
    print(f"Skill Terdeteksi: {hasil_3}\n")
