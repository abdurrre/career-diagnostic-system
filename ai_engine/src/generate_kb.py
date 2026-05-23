import pandas as pd
import json
import os
from collections import Counter

# Kata-kata yang di-exclude (HR benefits/noise)
EXCLUDE_WORDS = {
    '401(k)', '401k', 'discount', 'insurance', 'admission discounts',
    'benefits', 'health insurance', 'dental insurance', 'vision insurance',
    'paid time off', 'pto', 'retirement plan', 'bonus', 'equity', 'stock options',
    'flexible schedule', 'remote', 'hybrid', 'relocation assistance', 'discounts',
    'admission', 'employee', 'employee discount'
}

def generate_kb():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CSV_PATH = os.path.join(DATA_DIR, 'final_ready_it_jobs (2).csv')
    
    print(f"Membaca dataset dari {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    role_skill_mapping = {}
    roles = df['job_category'].dropna().unique()
    
    for role in roles:
        # Ignore DROP or other invalid roles just in case
        if role == 'DROP' or not isinstance(role, str):
            continue
            
        print(f"Memproses role: {role}")
        role_df = df[df['job_category'] == role]
        
        all_skills = []
        for skills_str in role_df['cleaned_skills'].dropna():
            skills = [s.strip().lower() for s in str(skills_str).split(',')]
            
            # Filter out exclude words and empty strings
            filtered_skills = [
                s for s in skills 
                if s and not any(ex in s for ex in EXCLUDE_WORDS)
            ]
            all_skills.extend(filtered_skills)
            
        # Hitung frekuensi
        skill_counts = Counter(all_skills)
        
        # Ambil Top 150
        top_skills = [skill for skill, count in skill_counts.most_common(150)]
        role_skill_mapping[role] = top_skills
        
    output_path = os.path.join(DATA_DIR, 'role_skill_mapping.json')
    with open(output_path, 'w') as f:
        json.dump(role_skill_mapping, f, indent=4)
        
    print(f"\nBerhasil menyimpan Knowledge Base ke {output_path}")

if __name__ == "__main__":
    generate_kb()
