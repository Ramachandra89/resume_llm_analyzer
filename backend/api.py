from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_service import LLaMAService
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analysis API")

# Initialize LLaMA service
try:
    llama_service = LLaMAService()
except Exception as e:
    logger.error(f"Failed to initialize LLaMA service: {str(e)}")
    raise

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str

@app.post("/analyze")
async def analyze_resume(request: AnalysisRequest):
    try:
        # Load prompt template
        prompt_template_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "resume_analysis_prompt.txt")
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()

        # Format the prompt
        prompt = prompt_template.format(
            resume_text=request.resume_text,
            job_description=request.job_description
        )

        # Generate response
        result = llama_service.generate_response(prompt)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
            
        return {"analysis": result["response"]}
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 