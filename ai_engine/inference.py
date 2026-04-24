import tensorflow as tf
import numpy as np
# from ai_engine.src.custom_metrics import weighted_binary_crossentropy

# NER_MODEL = tf.keras.models.load_model('ai_engine/models/ner_model.keras')
# SCORING_MODEL = tf.keras.models.load_model('ai_engine/models/scoring_model.keras')
# GAP_MODEL = tf.keras.models.load_model('ai_engine/models/gap_model.keras', custom_objects={'loss': weighted_binary_crossentropy})
# KNOWLEDGE_BASE = load_knowledge_base('data_science/clean_data/kb.json')

def extract_skills(cv_text: str) -> list:
    # Dummy return sesuai dokumen
    return ["Python", "TensorFlow", "SQL"]

def analyze_cv(skills: list, profession: str) -> dict: 
    # Return dictionary sesuai spesifikasi output
    return {
        "score": 72, 
        "gap": { 
            "critical": ["Docker", "Kubernetes"],
            "important": ["Terraform", "CI/CD"],
            "supplementary": ["Helm", "Istio"] 
        }
    }
