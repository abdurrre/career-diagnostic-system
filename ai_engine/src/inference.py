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

# Standarisasi Penamaan
SKILL_ALIASES = {
    "ai": "artificial intelligence",
    "c": "c/c++",
    "c++": "c/c++",
    "aws": "amazon web services",
    "js": "javascript",
    "react js": "react",
    "react.js": "react",
    "node js": "node.js",
    "nodejs": "node.js",
    "vue js": "vue.js",
    "vuejs": "vue.js",
    "ts": "typescript",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "gcp": "google cloud platform",
    "k8s": "kubernetes",
    "html5": "html",
    "css3": "css",
    "postgresql": "postgres",
    "db": "database",
    "ui": "ui/ux",
    "ux": "ui/ux",
    "ui/ux design": "ui/ux",
    "rn": "react native",
    "tf": "tensorflow",
    "pytorch": "torch"
}

def get_standard_skill(skill_name: str) -> str:
    clean_name = skill_name.lower().strip()
    return SKILL_ALIASES.get(clean_name, clean_name)

# SETUP PATHS
BASE_DIR = r'C:\Users\rohman\OneDrive\Documents\KULIAH\SEMESTER 6\MBKM\CAPSTONE PROJECT\career-diagnostic-system\ai_engine'
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# LOAD ARTIFACTS
try:
    # Artifacts Gap & Scoring
    with open(os.path.join(ARTIFACTS_DIR, 'skill_binarizer.pkl'), 'rb') as f:
        skill_binarizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'job_encoder.pkl'), 'rb') as f:
        job_encoder = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'knowledge_base.json')) as f:
        knowledge_base = json.load(f)
    SKILL_VOCAB = np.array(skill_binarizer.classes_)

    # Artifacts NER
    with open(os.path.join(ARTIFACTS_DIR, 'ner_tokenizer.pkl'), 'rb') as f:
        ner_tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'dataset-metadata.json'), 'r') as f:
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
        os.path.join(MODELS_DIR, 'ner_model.keras'), 
        custom_objects={'NERModel': NERModel}
    )
except Exception as e:
    print(f"Warning: NER_MODEL not loaded. ({e})")
    NER_MODEL = None

try:
    SCORING_MODEL = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, 'scoring_model.keras'), 
        custom_objects={'ScoringModel': ScoringModel}
    )
except Exception as e:
    print(f"Warning: ScoringModel not loaded. ({e})")
    SCORING_MODEL = None

try:
    GAP_MODEL = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, 'gap_model.keras'), 
        custom_objects={'loss': weighted_binary_crossentropy, 'GapModel': GapModel}
    )
except Exception as e:
    print(f"Warning: GapModel not loaded. ({e})")
    GAP_MODEL = None


# EKSTRAKSI SKILL DARI TEKS MENTAH
def extract_skills(cv_text: str) -> list:
    # 1. RULE-BASED EXTRACTION
    cv_text_lower = cv_text.lower()
    rule_based_skills = []
    for skill in SKILL_VOCAB:
        skill_str = str(skill).lower()
        if re.search(r'\b' + re.escape(skill_str) + r'\b', cv_text_lower):
            rule_based_skills.append(skill_str)

    # 2. NER MODEL EXTRACTION
    ner_skills = []
    if NER_MODEL and ner_tokenizer:
        # Pecah teks jadi token
        tokens = re.findall(r"[\w']+|[.,!?;]", cv_text)
        if tokens:
            # Sequencing & Padding
            seq = ner_tokenizer.texts_to_sequences([tokens])[0]
            padded_seq = pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')

            # Prediksi BIO Tags
            pred = NER_MODEL.predict(padded_seq, verbose=0)
            pred_tags = np.argmax(pred, axis=-1)[0]

            # Decoding Tags ke Teks
            current_skill = []
            for word, tag in zip(tokens, pred_tags[:len(tokens)]):
                if tag == 1: # B-SKILL
                    if current_skill:
                        ner_skills.append(" ".join(current_skill).lower())
                    current_skill = [word]
                elif tag == 2: # I-SKILL
                    if current_skill:
                        current_skill.append(word)
                else: # O
                    if current_skill:
                        ner_skills.append(" ".join(current_skill).lower())
                        current_skill = []
                        
            if current_skill:
                ner_skills.append(" ".join(current_skill).lower())

    # 3. GABUNGKAN & HAPUS DUPLIKAT
    extracted_skills = list(set(rule_based_skills + ner_skills))
    return extracted_skills


# ANALISIS GAP & SCORING
def analyze_cv(skills: list, profession: str) -> dict:
    if not skills or len(skills) == 0:
        return {
            "score": 0.0,
            "matched_skills": [],
            "gap": {"critical": [], "important": [], "supplementary": []},
            "error": "Tidak ada skill relevan yang terdeteksi di CV."
        }

    if not knowledge_base or job_encoder is None:
        return {"error": "Artifacts not loaded."}
        
    # Validasi profesi
    if profession not in job_encoder.classes_:
        return {"error": f"Profesi '{profession}' tidak ditemukan dalam sistem."}

    # Normalize user skills
    user_skills_canon = set(normalize_skills(skills, list(SKILL_VOCAB)))
    required_canon = set(knowledge_base.get(profession, []))

    # Terapkan SKILL_ALIASES ke set skill
    user_skills_aliased = {get_standard_skill(s) for s in user_skills_canon}
    required_aliased = {get_standard_skill(s) for s in required_canon}

    # Untuk input model Keras
    known_user = {s for s in user_skills_canon if s in SKILL_VOCAB}
    known_req = {s for s in required_canon if s in SKILL_VOCAB}

    # Matched skills
    matched = sorted(user_skills_aliased & required_aliased)

    # SCORING MODEL
    base_score = float(len(matched) / len(required_aliased)) if required_aliased else 0.0
    score = 0.0
    
    if SCORING_MODEL and skill_binarizer:
        user_vec = skill_binarizer.transform([list(known_user)])[0].astype('float32')
        req_vec = skill_binarizer.transform([list(known_req)])[0].astype('float32')
        feature = np.concatenate([user_vec, req_vec])[np.newaxis, :]
        keras_pred_score = float(SCORING_MODEL.predict(feature, verbose=0)[0][0])
        
        # Hybrid Scoring Calculation
        score = (keras_pred_score * 0.5) + (base_score * 0.5)
        
        if len(matched) > 0 and score < 0.05:
            score = max(0.05, base_score)
    else:
        score = base_score

    # GAP MODEL
    critical, important, supplementary = set(), set(), set()
    
    if GAP_MODEL:
        prof_id = job_encoder.transform([profession])[0]
        pred_probs = GAP_MODEL.predict(np.array([prof_id]), verbose=0)[0]
        
        for i, prob in enumerate(pred_probs):
            raw_skill_name = SKILL_VOCAB[i]
            std_skill_name = get_standard_skill(raw_skill_name)
            
            if std_skill_name not in user_skills_aliased:
                if prob >= 0.8:
                    critical.add(std_skill_name)
                elif prob >= 0.4:
                    important.add(std_skill_name)
                elif prob >= 0.2:
                    supplementary.add(std_skill_name)
                    
        important = important - critical
        supplementary = supplementary - critical - important
        
        critical = sorted(list(critical))
        important = sorted(list(important))
        supplementary = sorted(list(supplementary))
    else:
        missing = sorted(required_aliased - user_skills_aliased)
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
    
    print(f"TEST RUN")
    print(f"Profesi Target : {target_pekerjaan}")
    print(f"Teks CV Asli   : '{cv_mentah}'\n")
    
    extracted = extract_skills(cv_mentah)
    print(f"Ekstraksi NER Model: {extracted}\n")
    
    print(f"Hasil Analisis Akhir (JSON):")
    hasil_analisis = analyze_cv(extracted, target_pekerjaan)
    print(json.dumps(hasil_analisis, indent=4))
