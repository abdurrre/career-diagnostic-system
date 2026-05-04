import os
import json
import pickle
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from custom_metrics import weighted_binary_crossentropy
from architectures import GapModel, ScoringModel, NERModel
from skill_normalizer import normalize_skills

# SETUP PATHS
BASE_DIR = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/career-diagnostic-system/ai_engine'
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# LOAD ARTIFACTS
try:
    # Artifacts Gap & Scoring
    with open(f'{ARTIFACTS_DIR}/skill_binarizer.pkl', 'rb') as f:
        skill_binarizer = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/job_encoder.pkl', 'rb') as f:
        job_encoder = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/knowledge_base.json') as f:
        knowledge_base = json.load(f)
    SKILL_VOCAB = np.array(skill_binarizer.classes_)

    # Artifacts NER
    with open(f'{ARTIFACTS_DIR}/ner_tokenizer.pkl', 'rb') as f:
        ner_tokenizer = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/dataset-metadata.json', 'r') as f:
        metadata = json.load(f)
    MAX_LEN = 256

except FileNotFoundError as e:
    print(f"Warning: Artifacts not found. {e}")
    skill_binarizer, job_encoder, ner_tokenizer = None, None, None
    knowledge_base = {}
    SKILL_VOCAB = np.array([])

# LOAD MODELS
try:
    NER_MODEL = tf.keras.models.load_model(
        f'{MODELS_DIR}/ner_model.keras', 
        custom_objects={'NERModel': NERModel}
    )
except Exception as e:
    print(f"Warning: NER_MODEL not loaded. ({e})")
    NER_MODEL = None

try:
    SCORING_MODEL = tf.keras.models.load_model(
        f'{MODELS_DIR}/scoring_model.keras', 
        custom_objects={'ScoringModel': ScoringModel}
    )
except Exception as e:
    print(f"Warning: ScoringModel not loaded. ({e})")
    SCORING_MODEL = None

try:
    GAP_MODEL = tf.keras.models.load_model(
        f'{MODELS_DIR}/gap_model.keras', 
        custom_objects={'loss': weighted_binary_crossentropy, 'GapModel': GapModel}
    )
except Exception as e:
    print(f"Warning: GapModel not loaded. ({e})")
    GAP_MODEL = None


# EKSTRAKSI SKILL DARI TEKS MENTAH
def extract_skills(cv_text: str) -> list:
    if not NER_MODEL or not ner_tokenizer:
        return ["Error: NER Model or Tokenizer not loaded"]

    # Pecah teks jadi token
    tokens = re.findall(r"[\w']+|[.,!?;]", cv_text)
    if not tokens:
        return []

    # Sequencing & Padding
    seq = ner_tokenizer.texts_to_sequences([tokens])[0]
    padded_seq = pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')

    # Prediksi BIO Tags
    pred = NER_MODEL.predict(padded_seq, verbose=0)
    pred_tags = np.argmax(pred, axis=-1)[0]

    # Decoding Tags ke Teks
    extracted_skills = []
    current_skill = []

    for word, tag in zip(tokens, pred_tags[:len(tokens)]):
        if tag == 1: # B-SKILL
            if current_skill:
                extracted_skills.append(" ".join(current_skill))
            current_skill = [word]
        elif tag == 2: # I-SKILL
            if current_skill:
                current_skill.append(word)
        else: # O
            if current_skill:
                extracted_skills.append(" ".join(current_skill))
                current_skill = []
                
    if current_skill:
        extracted_skills.append(" ".join(current_skill))

    return extracted_skills


# ANALISIS GAP & SCORING
def analyze_cv(skills: list, profession: str) -> dict:
    if not knowledge_base or job_encoder is None:
        return {"error": "Artifacts not loaded."}
        
    # Validasi profesi
    if profession not in job_encoder.classes_:
        return {"error": f"Profesi '{profession}' tidak ditemukan dalam sistem."}

    # Normalize user skills
    user_skills_canon = set(normalize_skills(skills, list(SKILL_VOCAB)))
    required_canon = set(knowledge_base.get(profession, []))

    known_user = {s for s in user_skills_canon if s in SKILL_VOCAB}
    known_req = {s for s in required_canon if s in SKILL_VOCAB}

    matched = sorted(user_skills_canon & required_canon)

    # SCORING MODEL
    score = 0.0
    if SCORING_MODEL and skill_binarizer:
        user_vec = skill_binarizer.transform([list(known_user)])[0].astype('float32')
        req_vec = skill_binarizer.transform([list(known_req)])[0].astype('float32')
        feature = np.concatenate([user_vec, req_vec])[np.newaxis, :]
        score = float(SCORING_MODEL.predict(feature, verbose=0)[0][0])
    else:
        score = float(len(known_user & known_req) / len(known_req)) if known_req else 0.0

    # GAP MODEL
    critical, important, supplementary = [], [], []
    
    if GAP_MODEL:
        prof_id = job_encoder.transform([profession])[0]
        pred_probs = GAP_MODEL.predict(np.array([prof_id]), verbose=0)[0]
        
        for i, prob in enumerate(pred_probs):
            skill_name = SKILL_VOCAB[i]
            
            # Jika user belum punya skill ini, masukkan ke gap
            if skill_name not in known_user:
                if prob >= 0.8:
                    critical.append(skill_name)
                elif prob >= 0.4:
                    important.append(skill_name)
                elif prob >= 0.2:
                    supplementary.append(skill_name)
    else:
        missing = sorted(required_canon - user_skills_canon)
        n = len(missing)
        critical = missing[:max(1, n//3)]
        important = missing[max(1, n//3):max(2, 2*n//3)]
        supplementary = missing[max(2, 2*n//3):]

    return {
        "score": score,
        "matched_skills": matched,
        "gap": {
            "critical": critical,
            "important": important,
            "supplementary": supplementary
        }
    }

if __name__ == "__main__":
    cv_mentah = "Saya seorang mahasiswa yang mahir menggunakan Python, React, dan SQL untuk web development. Saya juga pernah menggunakan Git dan Docker."
    target_pekerjaan = "Data Analyst"
    
    print(f"TEST RUN END-TO-END PIPELINE")
    print(f"Profesi Target : {target_pekerjaan}")
    print(f"Teks CV Asli   : '{cv_mentah}'\n")
    
    # Ekstraksi (Deep Learning NER)
    extracted = extract_skills(cv_mentah)
    print(f"Ekstraksi NER Model: {extracted}\n")
    
    # Normalisasi, Gap, dan Scoring
    print(f"Hasil Analisis Akhir (JSON):")
    hasil_analisis = analyze_cv(extracted, target_pekerjaan)
    print(json.dumps(hasil_analisis, indent=4))
