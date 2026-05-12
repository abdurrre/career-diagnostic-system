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
#BASE_DIR = '/content/drive/MyDrive/semester 6/MBKM/Project Capstone/career-diagnostic-system/ai_engine'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
except FileNotFoundError as e:
    print(f"Warning: Artifacts Gap/Scoring not found. {e}")
    skill_binarizer, job_encoder = None, None
    knowledge_base = {}
    SKILL_VOCAB = np.array([])

try:
    # Artifacts NER
    with open(os.path.join(ARTIFACTS_DIR, 'ner_tokenizer.pkl'), 'rb') as f:
        ner_tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'dataset-metadata.json'), 'r') as f:
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
        os.path.join(MODELS_DIR, 'ner_model.keras'), 
        custom_objects={'NERModel': NERModel}
    )
except Exception as e:
    print(f"Warning: NER_MODEL not loaded. ({e})")
    NER_MODEL = None

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
    
    print(f"TEST RUN")
    print(f"Profesi Target : {target_pekerjaan}")
    print(f"Teks CV Asli   : '{cv_mentah}'\n")
    
    extracted = extract_skills(cv_mentah)
    print(f"Ekstraksi NER Model: {extracted}\n")
    
    print(f"Hasil Analisis Akhir (JSON):")
    hasil_analisis = analyze_cv(extracted, target_pekerjaan)
    print(json.dumps(hasil_analisis, indent=4))
