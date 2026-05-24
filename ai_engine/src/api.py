import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

AI_ENGINE_SRC = os.path.dirname(os.path.abspath(__file__))
if AI_ENGINE_SRC not in sys.path:
    sys.path.insert(0, AI_ENGINE_SRC)

from inference import extract_skills, analyze_cv

app = FastAPI(title="AI Engine - Career Diagnostic API", version="1.0.0")

PROFESSION_ID_MAP = {
    "AI / Machine Learning Engineer": 4,
    "Backend Developer": 1,
    "Data Analyst": 6,
    "Data Engineer": 5,
    "Data Scientist": 7,
    "Frontend Developer": 2,
    "Fullstack Developer": 3,
}

class DiagnosticRequest(BaseModel):
    raw_text: str           # Gabungan teks CV (PDF) + input teks tambahan dari Frontend
    target_profession: str  # Nama profesi target (harus sesuai dengan kelas di job_encoder)

def build_skill_analysis(analysis: dict) -> list:
    """
    Menggabungkan matched_skills dan gap_skills menjadi satu list
    skill_analysis sesuai API Contract Backend.
    """
    skill_analysis = []

    matched_skills = analysis.get("matched_skills", [])
    matched_categories = analysis.get("matched_categories", {})

    for skill in matched_skills:
        category = matched_categories.get(skill, "supplementary")
        skill_analysis.append({
            "name": skill,
            "status": "match",
            "category": category,
            "description": ""
        })

    gap = analysis.get("gap", {})

    for skill in gap.get("critical", []):
        skill_analysis.append({
            "name": skill,
            "status": "gap",
            "category": "critical",
            "description": ""
        })

    for skill in gap.get("important", []):
        skill_analysis.append({
            "name": skill,
            "status": "gap",
            "category": "important",
            "description": ""
        })

    for skill in gap.get("supplementary", []):
        skill_analysis.append({
            "name": skill,
            "status": "gap",
            "category": "supplementary",
            "description": ""
        })

    return skill_analysis

@app.post("/api/diagnose")
async def diagnose_career(req: DiagnosticRequest):
    if not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="Teks CV tidak boleh kosong.")

    if not req.target_profession.strip():
        raise HTTPException(status_code=400, detail="Target profesi tidak boleh kosong.")

    try:
        extracted_skills = extract_skills(req.raw_text)

        analysis = analyze_cv(extracted_skills, req.target_profession)

        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])

        skill_analysis = build_skill_analysis(analysis)
        id_profession = PROFESSION_ID_MAP.get(req.target_profession, 0)

        return {
            "message": "Analisis CV berhasil diproses",
            "profession_name": req.target_profession,
            "score": analysis.get("score_percentage", 0),
            "skill_analysis": skill_analysis,
            "id_profession": id_profession
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal pada AI Engine: {str(e)}"
        )
