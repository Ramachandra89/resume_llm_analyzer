from fastapi import FastAPI, Request
from pydantic import BaseModel
from inference import ResumeFitModel

app = FastAPI()
model = ResumeFitModel()

class ResumeJobRequest(BaseModel):
    resume: str
    job_description: str

@app.post("/predict-fit")
def predict_fit(request: ResumeJobRequest):
    score = model.predict_fit_score(request.resume, request.job_description)
    return {"fit_score": score}
