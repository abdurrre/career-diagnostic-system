import os
import json
import pickle
import numpy as np
import tensorflow as tf
from src.custom_metrics import weighted_binary_crossentropy
from src.architectures import GapModel, ScoringModel
from src.skill_normalizer import normalize_skills

# -- LOAD ARTIFACTS --
ARTIFACTS_DIR = 'ai_engine/data'
try:
    with open(f'{ARTIFACTS_DIR}/skill_binarizer.pkl', 'rb') as f:
        skill_binarizer = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/job_encoder.pkl', 'rb') as f:
        job_encoder = pickle.load(f)
    with open(f'{ARTIFACTS_DIR}/knowledge_base.json') as f:
        knowledge_base = json.load(f)
    SKILL_VOCAB = np.array(skill_binarizer.classes_)
except FileNotFoundError:
    print("Warning: Artifacts not found. Please train models first.")
    skill_binarizer = None
    job_encoder = None
    knowledge_base = {}
    SKILL_VOCAB = np.array([])

# -- LOAD MODELS --
# NER_MODEL = tf.keras.models.load_model('ai_engine/models/ner_model.keras')

try:
    SCORING_MODEL = tf.keras.models.load_model('ai_engine/models/scoring_model.keras', custom_objects={'ScoringModel': ScoringModel})
except Exception as e:
    print(f"Warning: ScoringModel not loaded. ({e})")
    SCORING_MODEL = None

try:
    GAP_MODEL = tf.keras.models.load_model('ai_engine/models/gap_model.keras', custom_objects={'loss': weighted_binary_crossentropy, 'GapModel': GapModel})
except Exception as e:
    print(f"Warning: GapModel not loaded. ({e})")
    GAP_MODEL = None

def extract_skills(cv_text: str) -> list:
    # Dummy return sesuai dokumen, untuk NER nanti aja
    return ["Python", "TensorFlow", "SQL", "Docker", "Node JS"]

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

    # 1. SCORING MODEL
    score = 0.0
    if SCORING_MODEL and skill_binarizer:
        user_vec = skill_binarizer.transform([list(known_user)])[0].astype('float32')
        req_vec = skill_binarizer.transform([list(known_req)])[0].astype('float32')
        feature = np.concatenate([user_vec, req_vec])[np.newaxis, :]
        score = float(SCORING_MODEL.predict(feature, verbose=0)[0][0])
    else:
        score = float(len(known_user & known_req) / len(known_req)) if known_req else 0.0

    # 2. GAP MODEL
    critical, important, supplementary = [], [], []
    
    if GAP_MODEL:
        # Get profession ID
        prof_id = job_encoder.transform([profession])[0]
        
        # Predict probability for all skills
        pred_probs = GAP_MODEL.predict(np.array([prof_id]), verbose=0)[0]
        
        # Filter skills that are NOT in user's CV
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
        # Fallback jika GapModel gagal load
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
    cv = "Saya seorang data analyst dengan pengalaman Python dan SQL."
    raw = extract_skills(cv)
    print(f"=== TEST RUN ===")
    print(f"Profesi Target: Data Analyst")
    print(f"Extracted Skills: {raw}")
    print(f"Analysis Result:")
    print(json.dumps(analyze_cv(raw, "Data Analyst"), indent=4))