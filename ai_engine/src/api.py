import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_ENGINE_SRC = os.path.join(BASE_DIR, "ai_engine", "src")
sys.path.append(AI_ENGINE_SRC)

from inference import extract_skills, analyze_cv

app = FastAPI(title="ML Inference API")

class PredictionRequest(BaseModel):
    cv_text: str
    target_profession: str

@app.post("/api/predict")
def predict(req: PredictionRequest):
    if not req.cv_text.strip():
        raise HTTPException(status_code=400, detail="CV text is required")
    
    if not req.target_profession.strip():
        raise HTTPException(status_code=400, detail="Target profession is required")

    try:
        ner_skills = extract_skills(req.cv_text)

        analysis = analyze_cv(ner_skills, req.target_profession)

        return {
            "status": "success",
            "ner_extracted_skills": ner_skills,
            "gap_analysis": analysis.get("gap", {}),
            "matched_skills": analysis.get("matched_skills", []),
            "score_percentage": analysis.get("score_percentage", 0)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    