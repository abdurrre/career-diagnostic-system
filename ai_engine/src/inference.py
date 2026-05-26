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
    # AI / ML
    "artificial intelligence": "ai",
    "artificial intelligence (ai)": "ai",
    "ai/ml": "ai",
    
    # Cloud Services
    "amazon web services": "aws",
    "amazon web services (aws)": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    
    # NLP / CV
    "natural language processing": "nlp",
    "cv": "computer vision",
    
    # Languages & Frameworks
    "js": "javascript",
    "ts": "typescript",
    "ml": "machine learning",
    "dl": "deep learning",
    "k8s": "kubernetes",
    "html5": "html",
    "css3": "css",
    "postgres": "postgresql",
    "torch": "pytorch",
    "tf": "tensorflow",
    
    # C/C++
    "c": "c/c++",
    "c++": "c/c++",
    
    # Frontend / UI / UX
    "react js": "react",
    "react.js": "react",
    "reactjs": "react",
    "node js": "node.js",
    "nodejs": "node.js",
    "vue js": "vue.js",
    "vuejs": "vue.js",
    "ui": "ui/ux",
    "ux": "ui/ux",
    "ux/ui": "ui/ux",
    "ui/ux design": "ui/ux",
    "rn": "react native",
    
    # Databases & others
    "database": "db",
}

def get_standard_skill(skill_name: str) -> str:
    clean_name = skill_name.lower().strip()
    return SKILL_ALIASES.get(clean_name, clean_name)

def resolve_profession(profession: str) -> str:
    """Case-insensitive matching terhadap daftar profesi yang dikenal sistem."""
    if job_encoder is None:
        return profession
    profession_stripped = profession.strip()
    for cls in job_encoder.classes_:
        if cls.lower() == profession_stripped.lower():
            return cls
    return profession_stripped

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
    print(f"Artifacts Gap/Scoring not found. {e}")
    skill_binarizer, job_encoder = None, None
    knowledge_base = {}
    SKILL_VOCAB = np.array([])

try:
    with open(os.path.join(ARTIFACTS_DIR, 'role_skill_mapping.json')) as f:
        role_skill_mapping = json.load(f)
except FileNotFoundError:
    # print("Warning: role_skill_mapping.json not found, using knowledge_base instead.")
    role_skill_mapping = knowledge_base

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
    print(f"NER_MODEL not loaded. ({e})")
    NER_MODEL = None

try:
    GAP_MODEL = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, 'gap_model.keras'),
        custom_objects={'GapModel': GapModel},
        compile=False 
    )
    print("GapModel loaded successfully from gap_model.keras")
except Exception as e:
    print(f"GapModel not loaded. ({e})")
    GAP_MODEL = None


def extract_skills(cv_text: str) -> list:
    if not NER_MODEL or not ner_tokenizer:
        print("fallback to keyword matching (NER unavailable)")
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

    raw_extracted = sorted(list(set(rule_based_skills + ner_skills)))
    
    # Filter out obvious NER noise (like "seorang mahasiswa yang")
    extracted_skills = []
    for skill in raw_extracted:
        if len(SKILL_VOCAB) > 0 and (skill in SKILL_VOCAB or get_standard_skill(skill) in SKILL_VOCAB):
            extracted_skills.append(skill)
        elif len(skill.split()) <= 2: # Keep unknown skills if they are short (1-2 words max)
            extracted_skills.append(skill)
            
    return extracted_skills


def analyze_cv(skills: list, profession: str) -> dict:
    if not skills or len(skills) == 0:
        return {
            "score": 0.0,
            "matched_skills": [],
            "matched_categories": {},
            "gap": {"critical": [], "important": [], "supplementary": []},
            "error": "Tidak ada skill relevan yang terdeteksi di CV."
        }

    if not knowledge_base or job_encoder is None:
        return {"error": "Artifacts not loaded."}

    # Case-insensitive profession resolution
    profession = resolve_profession(profession)

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
    matched_categories = {}  # track category for each matched skill
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

        role_kb_raw = role_skill_mapping.get(profession, [])
        role_kb_aliased = {get_standard_skill(s) for s in role_kb_raw}

        cand_critical = []
        cand_important = []
        cand_supp = []

        for aliased_skill, prob in aliased_probs.items():
            # filter predicted skills based on the top-n knowledge base mapping
            if role_kb_aliased and aliased_skill not in role_kb_aliased:
                continue

            if prob >= 0.8:
                cand_critical.append((aliased_skill, prob))
            elif prob >= 0.4:
                cand_important.append((aliased_skill, prob))
            else:
                cand_supp.append((aliased_skill, prob))

        # Ensure any skills in the profession's knowledge base that are not in the model vocabulary are captured
        processed_skills = set(aliased_probs.keys())
        for kb_skill in role_kb_aliased:
            if kb_skill not in processed_skills:
                cand_supp.append((kb_skill, 0.0))

        # sort predicted skill recommendations by confidence (no capping limits to show all skills)
        cand_critical.sort(key=lambda x: x[1], reverse=True)
        cand_important.sort(key=lambda x: x[1], reverse=True)
        cand_supp.sort(key=lambda x: x[1], reverse=True)

        # separate acquired skills from gap categories
        for skill, prob in cand_critical:
            if skill in user_skills_aliased:
                matched.append(skill)
                matched_categories[skill] = "critical"
            else:
                critical.append(skill)
                
        for skill, prob in cand_important:
            if skill in user_skills_aliased:
                matched.append(skill)
                if skill not in matched_categories:
                    matched_categories[skill] = "important"
            else:
                important.append(skill)
                
        for skill, prob in cand_supp:
            if skill in user_skills_aliased:
                matched.append(skill)
                if skill not in matched_categories:
                    matched_categories[skill] = "supplementary"
            else:
                supplementary.append(skill)

        # calculate dynamic weighted scoring based on importance levels
        user_matched_critical = [s for s, p in cand_critical if s in user_skills_aliased]
        user_matched_important = [s for s, p in cand_important if s in user_skills_aliased]
        user_matched_supp = [s for s, p in cand_supp if s in user_skills_aliased]

        user_points = (len(user_matched_critical) * WEIGHT_CRITICAL) + (len(user_matched_important) * WEIGHT_IMPORTANT) + (len(user_matched_supp) * WEIGHT_SUPPLEMENTARY)
        
        # normalize score against total weighted points of ALL required skills for this profession
        # (matched + gap), so score truly reflects how much of the requirement is fulfilled
        total_weight_required = (
            len(cand_critical) * WEIGHT_CRITICAL +
            len(cand_important) * WEIGHT_IMPORTANT +
            len(cand_supp) * WEIGHT_SUPPLEMENTARY
        )

        if total_weight_required > 0:
            score_ratio = user_points / total_weight_required
        else:
            score_ratio = 0.0

        # FALLBACK: Jika model menghasilkan prediksi kosong (semua prob < 0.2),
        # gunakan Knowledge Base (role_skill_mapping) untuk mengisi gap berdasarkan rank.
        total_model_candidates = len(cand_critical) + len(cand_important) + len(cand_supp)
        if total_model_candidates == 0:
            print("WARNING: Model predictions all below threshold. Falling back to KB-based gap.")
            role_kb_raw = role_skill_mapping.get(profession, [])
            role_kb_list = [get_standard_skill(s) for s in role_kb_raw]
            # Deduplicate sambil menjaga urutan
            seen = set()
            role_kb_unique = []
            for s in role_kb_list:
                if s not in seen:
                    seen.add(s)
                    role_kb_unique.append(s)

            total_kb = len(role_kb_unique)
            cutoff_critical = int(total_kb * 0.20)     # Top 20%
            cutoff_important = int(total_kb * 0.50)    # Next 30%

            for idx, skill in enumerate(role_kb_unique):
                if skill in user_skills_aliased:
                    matched.append(skill)
                    if idx < cutoff_critical:
                        matched_categories[skill] = "critical"
                    elif idx < cutoff_important:
                        matched_categories.setdefault(skill, "important")
                    else:
                        matched_categories.setdefault(skill, "supplementary")
                else:
                    if idx < cutoff_critical:
                        critical.append(skill)
                    elif idx < cutoff_important:
                        important.append(skill)
                    else:
                        supplementary.append(skill)

            # Hitung skor berdasarkan jumlah skill yang cocok terhadap total KB
            if total_kb > 0:
                score_ratio = len([s for s in role_kb_unique if s in user_skills_aliased]) / total_kb
            else:
                score_ratio = 0.0

    else:
        # fallback rule-based math scoring when the gap model is not available
        missing_skills = sorted(required_aliased - user_skills_aliased)
        important = missing_skills
        
        if len(required_aliased) > 0:
            score_ratio = len(matched) / len(required_aliased)
        else:
            score_ratio = 0.0
        print(f"Fallback scoring: {len(matched)}/{len(required_aliased)} = {score_ratio:.2%}")

    final_score_percentage = round(score_ratio * 100, 2)

    # remove duplicates and sort alphabetically for consistent display
    matched = sorted(list(set(matched)))
    critical = sorted(list(set(critical)))
    important = sorted(list(set(important)))
    supplementary = sorted(list(set(supplementary)))

    return {
        "score": score_ratio,
        "score_percentage": final_score_percentage,
        "matched_skills": matched,
        "matched_categories": matched_categories,
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
