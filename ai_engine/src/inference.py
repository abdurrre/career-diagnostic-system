import os
import json
import pickle
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from custom_metrics import weighted_binary_crossentropy
from architectures import GapModel, NERModel
from skill_normalizer import normalize_skills

# SETUP PATHS
#BASE_DIR = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/career-diagnostic-system/ai_engine'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
except FileNotFoundError as e:
    print(f"Warning: Artifacts Gap/Scoring not found. {e}")
    skill_binarizer, job_encoder = None, None
    knowledge_base = {}
    SKILL_VOCAB = np.array([])

try:
    # Artifacts NER
    with open(f'{ARTIFACTS_DIR}/ner_tokenizer.pkl', 'rb') as f:
        ner_tokenizer = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/dataset-metadata.json', 'r') as f:
        metadata = json.load(f)
    MAX_LEN = 256
except FileNotFoundError as e:
    print(f"Warning: Artifacts NER not found. {e}")
    ner_tokenizer = None
    metadata = {}
    MAX_LEN = 256

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
        print("Warning: Menggunakan Fallback Keyword Matching karena NER Model tidak ditemukan.")
        # Fallback: Naive keyword matching
        fallback_skills = []
        if len(SKILL_VOCAB) > 0:
            # Cari skill dari SKILL_VOCAB yang text-nya muncul di CV
            cv_lower = cv_text.lower()
            for skill in SKILL_VOCAB:
                if skill.lower() in cv_lower:
                    fallback_skills.append(skill)
        else:
            # Hardcode fallback jika SKILL_VOCAB juga kosong
            fallback_skills = ["python", "sql", "react", "git", "docker"]
        return fallback_skills

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

    # Scoring
    critical, important, supplementary = [], [], []
    total_weight_required = 0.0
    user_acquired_weight = 0.0

    WEIGHT_CRITICAL = 3.0
    WEIGHT_IMPORTANT = 2.0
    WEIGHT_SUPPLEMENTARY = 1.0

    if GAP_MODEL:
        prof_id = job_encoder.transform([profession])[0]

        pred_probs = GAP_MODEL.predict(np.array([prof_id]), verbose=0)[0]

        for i, prob in enumerate(pred_probs):
            skill_name = SKILL_VOCAB[i]

            if prob >= 0.8:
                weight = WEIGHT_CRITICAL
                target_gap_list = critical
            elif prob >= 0.4:
                weight = WEIGHT_IMPORTANT
                target_gap_list = important
            elif prob >= 0.2:
                weight = WEIGHT_SUPPLEMENTARY
                target_gap_list = supplementary
            else:
                continue # Skip

            total_weight_required += weight

            if skill_name in known_user:
                user_acquired_weight += weight
            else:
                target_gap_list.append(skill_name)
        
        score_ratio = (user_acquired_weight / total_weight_required) if total_weight_required > 0 else 0.0
    else:
        score_ratio = 0.0

    final_score_percentage = round(score_ratio * 100, 2)

    return {
        "score_percentage": final_score_percentage,
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
