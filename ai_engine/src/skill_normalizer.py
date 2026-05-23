import re
import difflib

# beberapa saja, nanti niatnya mau di banyakin lagi
ALIASES = {
    'node js': 'node.js',
    'nodejs': 'node.js',
    'react js': 'react',
    'reactjs': 'react',
    'react.js': 'react',
    'vue js': 'vue.js',
    'vuejs': 'vue.js',
    'c++': 'c/c++',
    'c': 'c/c++',
    'ml': 'machine learning',
    'js': 'javascript',
    'ts': 'typescript',
    'ai': 'artificial intelligence',
    'golang': 'go',
    'k8s': 'kubernetes',
    'aws': 'amazon web services',
    'gcp': 'google cloud',
    'postgres': 'postgresql',
    'sql server': 'microsoft sql server',
    'ms sql': 'microsoft sql server'
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def fuzzy_match(skill: str, vocabulary: list[str], threshold: float = 0.8) -> str:
    matches = difflib.get_close_matches(skill, vocabulary, n=1, cutoff=threshold)
    if matches:
        return matches[0]
    return skill

def normalize_skills(raw_skills: list[str], vocabulary: list[str] = None) -> list[str]:
    normalized = set()
    
    for raw in raw_skills:
        clean_skill = clean_text(raw)
        if not clean_skill:
            continue
            
        if clean_skill in ALIASES:
            canonical = ALIASES[clean_skill]
        else:
            canonical = clean_skill
            
        if vocabulary and canonical not in vocabulary:
            canonical = fuzzy_match(canonical, vocabulary)
            
        normalized.add(canonical)
        
    return sorted(list(normalized))

if __name__ == "__main__":
    test_raw = ["Node JS", "ML", "Python", "ReactJS", "postgres"]
    test_vocab = ["node.js", "machine learning", "python", "react", "postgresql"]
    print(f"Raw : {test_raw}")
    print(f"Result : {normalize_skills(test_raw, test_vocab)}")
