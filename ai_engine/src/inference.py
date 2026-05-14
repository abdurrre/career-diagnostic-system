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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
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
        custom_objects={'GapModel': GapModel},
        compile=False  # inference only, skip optimizer/loss deserialization
    )
    print("GapModel loaded successfully from gap_model.keras")
except Exception as e:
    print(f"Warning: GapModel not loaded. ({e})")
    GAP_MODEL = None


def extract_skills(cv_text: str) -> list:
    if not NER_MODEL or not ner_tokenizer:
        print("Warning: fallback to keyword matching (NER unavailable)")
        fallback_skills = []
        cv_lower = cv_text.lower()
        if len(SKILL_VOCAB) > 0:
            for skill in SKILL_VOCAB:
                pattern = r"\b" + re.escape(skill.lower()) + r"\b"
                if re.search(pattern, cv_lower):
                    fallback_skills.append(skill.lower().strip())
        else:
            fallback_skills = ["python", "sql", "react", "git", "docker"]
        return fallback_skills

    tokens = re.findall(r"[\w']+|[.,!?;]", cv_text)
    if not tokens:
        return []

    seq = ner_tokenizer.texts_to_sequences([tokens])[0]
    padded_seq = pad_sequences([seq], maxlen=MAX_LEN, padding='post', truncating='post')

    # Rule-based extraction runs alongside NER as a complement
    rule_based_skills = []
    cv_lower = cv_text.lower()
    if len(SKILL_VOCAB) > 0:
        for skill in SKILL_VOCAB:
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, cv_lower):
                rule_based_skills.append(skill.lower().strip())

    ner_skills = []

    pred = NER_MODEL.predict(padded_seq, verbose=0)
    pred_tags = np.argmax(pred, axis=-1)[0]

    # Decode BIO tags back to skill spans
    current_skill = []
    for word, tag in zip(tokens, pred_tags[:len(tokens)]):
        if tag == 1:  # B-SKILL
            if current_skill:
                ner_skills.append(" ".join(current_skill).lower())
            current_skill = [word]
        elif tag == 2:  # I-SKILL
            if current_skill:
                current_skill.append(word)
        else:  # O
            if current_skill:
                ner_skills.append(" ".join(current_skill).lower())
                current_skill = []
                
    if current_skill:
        ner_skills.append(" ".join(current_skill).lower())

    extracted_skills = list(set(rule_based_skills + ner_skills))
    return extracted_skills


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
        
    if profession not in job_encoder.classes_:
        return {"error": f"Profesi '{profession}' tidak ditemukan dalam sistem."}

    skills_cleaned = {s.lower().strip() for s in skills}
    required_raw = knowledge_base.get(profession, [])
    required_cleaned = {s.lower().strip() for s in required_raw}

    # Normalize aliases before matching
    user_skills_aliased = {get_standard_skill(s) for s in skills_cleaned}
    required_aliased = {get_standard_skill(s) for s in required_cleaned}
    
    print("Extracted Skills:", list(skills_cleaned))
    print("Required Skills for Profession:", list(required_cleaned))

    matched = sorted(user_skills_aliased & required_aliased)
    print("Matched Skills:", matched)

    known_user = {s for s in user_skills_aliased if s in SKILL_VOCAB}
    known_req = {s for s in required_aliased if s in SKILL_VOCAB}

    critical, important, supplementary = [], [], []
    total_weight_required = 0.0
    user_acquired_weight = 0.0

    WEIGHT_CRITICAL = 3.0
    WEIGHT_IMPORTANT = 2.0
    WEIGHT_SUPPLEMENTARY = 1.0

    if GAP_MODEL:
        prof_id = job_encoder.transform([profession])[0]
        pred_probs = GAP_MODEL.predict(np.array([prof_id]), verbose=0)[0]

        # Deduplicate aliased skills by keeping the max probability
        aliased_probs = {}
        for i, prob in enumerate(pred_probs):
            raw_skill_name = SKILL_VOCAB[i]
            aliased_skill = get_standard_skill(raw_skill_name)
            if aliased_skill in aliased_probs:
                aliased_probs[aliased_skill] = max(aliased_probs[aliased_skill], prob)
            else:
                aliased_probs[aliased_skill] = prob

        for aliased_skill, prob in aliased_probs.items():
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
                continue

            total_weight_required += weight

            if aliased_skill in user_skills_aliased:
                user_acquired_weight += weight
            else:
                target_gap_list.append(aliased_skill)
        
        score_ratio = (user_acquired_weight / total_weight_required) if total_weight_required > 0 else 0.0
    else:
        # Fallback: pure math scoring without AI model
        missing_skills = sorted(required_aliased - user_skills_aliased)
        important = missing_skills
        
        if len(required_aliased) > 0:
            score_ratio = len(matched) / len(required_aliased)
        else:
            score_ratio = 0.0
        print(f"Fallback scoring: {len(matched)}/{len(required_aliased)} = {score_ratio:.2%}")

    final_score_percentage = round(score_ratio * 100, 2)

    return {
        "score": score_ratio,
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
    
    print(f"Profesi Target : {target_pekerjaan}")
    print(f"Teks CV        : '{cv_mentah}'\n")
    
    extracted = extract_skills(cv_mentah)
    print(f"Extracted skills: {extracted}\n")
    
    hasil_analisis = analyze_cv(extracted, target_pekerjaan)
    print(json.dumps(hasil_analisis, indent=4))
