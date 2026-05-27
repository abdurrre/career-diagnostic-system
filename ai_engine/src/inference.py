import os
import json
import pickle
import numpy as np
import tensorflow as tf
import re
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences
from custom_metrics import weighted_binary_crossentropy
from architectures import GapModel, NERModel
from skill_aliases import SKILL_ALIASES, SKILL_COVERAGE

def get_standard_skill(skill_name: str) -> str:
    clean_name = skill_name.lower().strip()
    return SKILL_ALIASES.get(clean_name, clean_name)


def get_matchable_skills(skill_name: str) -> set:
    standard_skill = get_standard_skill(skill_name)
    covered_skills = {
        get_standard_skill(covered_skill)
        for covered_skill in SKILL_COVERAGE.get(standard_skill, [])
    }
    return {standard_skill, *covered_skills}


def expand_matchable_skills(skills: set) -> set:
    expanded = set()
    for skill in skills:
        expanded.update(get_matchable_skills(skill))
    return expanded

def resolve_profession(profession: str) -> str:
    """Case-insensitive matching terhadap daftar profesi yang dikenal sistem."""
    if job_encoder is None:
        return profession
    profession_stripped = profession.strip()
    for cls in job_encoder.classes_:
        if cls.lower() == profession_stripped.lower():
            return cls
    return profession_stripped

SCORING_SKILL_LIMIT = 20
WEIGHT_CRITICAL = 3.0
WEIGHT_IMPORTANT = 2.0
WEIGHT_SUPPLEMENTARY = 1.0


def dedupe_aliased_skills(skills: list) -> list:
    """Normalize aliases and deduplicate while preserving rank/order."""
    seen = set()
    deduped = []
    for skill in skills:
        canonical = get_standard_skill(str(skill))
        if canonical and canonical not in seen:
            seen.add(canonical)
            deduped.append(canonical)
    return deduped


def get_ranked_role_skills(profession: str) -> list:
    ranked_skills = role_skill_mapping.get(profession) or knowledge_base.get(profession, [])
    return dedupe_aliased_skills(ranked_skills)


def get_gap_model_probabilities(profession: str) -> dict:
    if not GAP_MODEL:
        return {}

    prof_id = job_encoder.transform([profession])[0]
    pred_probs = GAP_MODEL.predict(np.array([prof_id]), verbose=0)[0]

    aliased_probs = {}
    for i, prob in enumerate(pred_probs):
        aliased_skill = get_standard_skill(SKILL_VOCAB[i])
        aliased_probs[aliased_skill] = max(float(prob), aliased_probs.get(aliased_skill, 0.0))

    return aliased_probs


def get_skill_category(index: int, total: int) -> str:
    critical_cutoff = max(1, math.ceil(total * 0.20))
    important_cutoff = max(critical_cutoff, math.ceil(total * 0.50))

    if index < critical_cutoff:
        return "critical"
    if index < important_cutoff:
        return "important"
    return "supplementary"


def get_category_weight(category: str) -> float:
    if category == "critical":
        return WEIGHT_CRITICAL
    if category == "important":
        return WEIGHT_IMPORTANT
    return WEIGHT_SUPPLEMENTARY


def clamp_score(score: float) -> float:
    return min(max(score, 0.0), 1.0)


def build_gap_requirements(profession: str) -> list:
    ranked_role_skills = get_ranked_role_skills(profession)
    gap_probs = get_gap_model_probabilities(profession)

    if gap_probs:
        role_rank = {skill: idx for idx, skill in enumerate(ranked_role_skills)}
        candidate_skills = ranked_role_skills or sorted(gap_probs.keys())
        candidate_skills = sorted(
            candidate_skills,
            key=lambda skill: (-gap_probs.get(skill, 0.0), role_rank.get(skill, len(role_rank)))
        )
    else:
        candidate_skills = ranked_role_skills

    candidate_skills = candidate_skills[:SCORING_SKILL_LIMIT]
    total = len(candidate_skills)

    return [
        {
            "skill": skill,
            "category": get_skill_category(index, total),
            "probability": gap_probs.get(skill, 0.0),
        }
        for index, skill in enumerate(candidate_skills)
    ]

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
    user_skills_matchable = expand_matchable_skills(user_skills_aliased)
    required_aliased = {get_standard_skill(s) for s in required_cleaned}
    
    print("Extracted Skills:", list(skills_cleaned))
    print("Required Skills for Profession:", list(required_cleaned))

    matched = sorted(user_skills_matchable & required_aliased)
    print("Matched Skills:", matched)

    critical, important, supplementary = [], [], []
    matched_categories = {}  # track category for each matched skill
    gap_requirements = build_gap_requirements(profession)

    if not gap_requirements:
        fallback_requirements = sorted(required_aliased)[:SCORING_SKILL_LIMIT]
        gap_requirements = [
            {
                "skill": skill,
                "category": get_skill_category(index, len(fallback_requirements)),
                "probability": 0.0,
            }
            for index, skill in enumerate(fallback_requirements)
        ]

    gap_requirement_set = {item["skill"] for item in gap_requirements}
    user_points = 0.0
    total_points = 0.0

    for requirement in gap_requirements:
        required_skill = requirement["skill"]
        category = requirement["category"]
        weight = get_category_weight(category)
        total_points += weight

        if required_skill in user_skills_matchable:
            matched.append(required_skill)
            matched_categories[required_skill] = category
            user_points += weight
        elif category == "critical":
            critical.append(required_skill)
        elif category == "important":
            important.append(required_skill)
        else:
            supplementary.append(required_skill)

    # Scoring is based only on GapModel-derived requirements. Extra valid
    # matches stay visible but cannot inflate the score beyond the gap basis.
    for skill in matched:
        if skill not in gap_requirement_set:
            matched_categories.setdefault(skill, "supplementary")

    if total_points > 0:
        score_ratio = clamp_score(user_points / total_points)
    else:
        score_ratio = 0.0

    print(f"Gap-based scoring: {user_points:.1f}/{total_points:.1f} = {score_ratio:.2%}")

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
