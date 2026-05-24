import re

# Input Sanitization
def sanitize_text(text: str) -> str:
    """Strip whitespace and remove HTML tags."""
    text = text.strip()
    text = re.sub(r"<[^>]*>", "", text)
    return text

# Prompt Injection Detection (Regex Fuzzy Matching)
def _leet_pattern(phrase: str) -> re.Pattern:
    """
    Convert a plain phrase into a regex pattern that matches
    leet-speak variants and optional separator characters.

    Example: "ignore" -> matches "ign0re", "i.g.n.o.r.e", "ignor3"
    """
    leet_map = {
        'a': '[a@4àáâã]',
        'e': '[e3èéêë]',
        'i': '[i1!ìíîï]',
        'o': '[o0òóôõ]',
        's': '[s$5]',
        'u': '[uùúûü]',
        'l': '[l1|]',
        't': '[t7+]',
    }
    # Allow optional separators between characters (space, dot, dash, underscore)
    separator = r'[\s._\-]*'
    regex_parts = []
    for char in phrase:
        if char == ' ':
            # Word boundary: require at least one separator or whitespace
            regex_parts.append(r'[\s._\-]+')
        elif char in leet_map:
            regex_parts.append(leet_map[char])
        else:
            regex_parts.append(re.escape(char))
        regex_parts.append(separator)

    pattern_str = ''.join(regex_parts)
    return re.compile(pattern_str, re.IGNORECASE)


# Pre-compiled injection patterns with leet-speak resistance
UNSAFE_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "ignore the rules",
    "forget your instructions",
    "forget previous instructions",
    "reveal system prompt",
    "show system prompt",
    "print system prompt",
    "display system prompt",
    "what is your system prompt",
    "print hidden instructions",
    "show hidden instructions",
    "jailbreak",
    "developer mode",
    "dan mode",
    "override instructions",
    "bypass instructions",
    "you are now",
    "pretend to be",
    "act as",
    "pretend you are",
    "roleplay as",
    "new instructions",
    "disregard all",
    "disregard previous",
    "do anything now",
    "no restrictions",
    "unlimited mode",
    "admin mode",
    "sudo mode",
    "god mode",
]

UNSAFE_PATTERNS = [_leet_pattern(phrase) for phrase in UNSAFE_PHRASES]

def detect_prompt_injection(text: str) -> bool:
    """
    Check if user input contains prompt injection patterns.
    Uses regex fuzzy matching to resist leet-speak bypass attempts.
    """
    for pattern in UNSAFE_PATTERNS:
        if pattern.search(text):
            return True
    return False


# Code Request Detection (Pre-filter Keywords)
CODE_KEYWORDS = [
    "contoh kode",
    "contoh program",
    "contoh query",
    "contoh sintaks",
    "contoh syntax",
    "contoh kodingan",
    "contoh implementasi",
    "tuliskan kode",
    "tulis kode",
    "tulis query",
    "bikin kode",
    "bikin program",
    "buatkan kode",
    "buat kode",
    "buat query",
    "code",
    "script",
    "query",
    "sql",
    "endpoint",
    "function",
    "class",
    "linked list",
    "linkedlist",
    "contoh kecil",
    "contoh coding",
    "source code",
]

def detect_code_request(text: str) -> bool:
    text_lower = text.lower()
    for keyword in CODE_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


# Output Safety Filter (Regex-based Secret & Code Block Detection)
SECRET_PATTERNS = [
    # OpenAI / Groq style API keys
    re.compile(r'sk-[A-Za-z0-9]{20,}', re.IGNORECASE),
    re.compile(r'gsk_[A-Za-z0-9]{20,}', re.IGNORECASE),
    # GitHub tokens
    re.compile(r'ghp_[A-Za-z0-9]{20,}', re.IGNORECASE),
    re.compile(r'gho_[A-Za-z0-9]{20,}', re.IGNORECASE),
    # Bearer tokens (long alphanumeric strings)
    re.compile(r'Bearer\s+[A-Za-z0-9\-_.]{20,}', re.IGNORECASE),
    # JWT-like tokens (xxx.xxx.xxx)
    re.compile(r'eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+'),
    # PEM private key blocks
    re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'),
    # Generic long hex/base64 secrets (40+ chars)
    re.compile(r'(?:key|secret|token|password)\s*[=:]\s*["\']?[A-Za-z0-9+/\-_]{40,}', re.IGNORECASE),
]

# Patterns representing programming code leaking in LLM responses
CODE_LEAK_PATTERNS = [
    re.compile(r'```'),  # Any markdown code block
    re.compile(r'\bdef\s+\w+\s*\(.*?\):', re.IGNORECASE),  # Python function definition
    re.compile(r'\bclass\s+\w+[\s{:]', re.IGNORECASE),  # Class definition
    re.compile(r'\bimport\s+\w+(\s*,\s*\w+)*', re.IGNORECASE),  # Python imports
    re.compile(r'#include\s*[<"]\w+\.?h?[>"]', re.IGNORECASE),  # C/C++ includes
    re.compile(r'\bSELECT\b.*\bFROM\b', re.IGNORECASE | re.DOTALL),  # SQL queries
    re.compile(r'<\s*html\b', re.IGNORECASE),  # HTML documents
    re.compile(r'const\s+\w+\s*=\s*\([^)]*\)\s*=>', re.IGNORECASE),  # JS ES6 arrow function
    re.compile(r'function\s+\w+\s*\(', re.IGNORECASE),  # JS standard function
]


def check_output_safety(text: str) -> str:
    # 1. First scan for secret leaks
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return "Maaf, tidak bisa merespond itu."
            
    # 2. Scan for program code leaks
    for pattern in CODE_LEAK_PATTERNS:
        if pattern.search(text):
            return (
                "Maaf, saya tidak dapat memberikan contoh kode program atau query teknis, "
                "namun saya dapat menjelaskan konsepnya secara teori atau memberikan jalur pembelajaran yang sesuai."
            )
            
    return text

