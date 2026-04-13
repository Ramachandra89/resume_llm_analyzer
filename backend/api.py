from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.skill_matcher import SkillMatcher
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analysis API")

# ── Model service: SageMaker if endpoint name is set, otherwise local ─────────
if os.getenv("SAGEMAKER_ENDPOINT_NAME"):
    from backend.sagemaker_service import SageMakerService
    try:
        local_model_service = SageMakerService()
        logger.info("Using SageMaker endpoint: %s", os.getenv("SAGEMAKER_ENDPOINT_NAME"))
    except Exception as e:
        logger.error("Failed to initialize SageMakerService: %s", e)
        raise
else:
    from backend.local_model_service import LocalModelService
    try:
        local_model_service = LocalModelService()
        logger.info("Using local model service (no SAGEMAKER_ENDPOINT_NAME set)")
    except Exception as e:
        logger.error("Failed to initialize LocalModelService: %s", e)
        raise

# Initialize Skill Matcher
try:
    skill_matcher = SkillMatcher(local_model_service)
except Exception as e:
    logger.error(f"Failed to initialize Skill Matcher: {str(e)}")
    raise

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str

class ChatRequest(BaseModel):
    conversation: str

class SkillAssessmentRequest(BaseModel):
    resume_text: str
    job_description: str

class SkillEvaluationRequest(BaseModel):
    user_responses: dict  # {question_num: answer_text}
    missing_skills: list  # List of skills to assess

class JobScoutRequest(BaseModel):
    company_name: str
    resume_text: str

class TailorRequest(BaseModel):
    resume_text: str
    job_title: str
    job_url: str
    job_description: str
    company_name: str


def load_prompt(filename: str) -> str:
    prompt_template_path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(prompt_template_path, 'r') as f:
        return f.read()


@app.post("/analyze")
async def analyze_resume(request: AnalysisRequest):
    try:
        prompt_template = load_prompt("resume_analysis_prompt.txt")
        prompt = prompt_template.format(
            resume_text=request.resume_text,
            job_description=request.job_description
        )

        result = local_model_service.generate_response(prompt)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {"analysis": result["response"]}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        prompt = f"You are a professional career coach. Continue the conversation with the user using the context below.\n\n{request.conversation}\nCoach:"
        result = local_model_service.generate_response(prompt, max_length=512)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {"response": result["response"]}
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/skill-assessment")
async def skill_assessment(request: SkillAssessmentRequest):
    """
    Performs comprehensive skill matching between resume and job description.
    Returns matched skills, missing skills, and probing questions.
    """
    try:
        logger.info("Starting skill assessment...")
        
        # Extract skills from both texts
        resume_skills = skill_matcher.extract_skills(request.resume_text, "resume")
        job_skills = skill_matcher.extract_skills(request.job_description, "job description")
        
        logger.info(f"Extracted resume skills: {resume_skills}")
        logger.info(f"Extracted job skills: {job_skills}")
        
        # Match skills
        match_result = skill_matcher.match_skills(resume_skills, job_skills)
        logger.info(f"Skill match result: {match_result}")
        
        # Collect missing skills for probing questions
        missing_skills = []
        for category, skills in match_result.get("missing", {}).items():
            missing_skills.extend(skills)
        
        # Generate probing questions if there are missing skills
        probing_questions = {}
        if missing_skills:
            probing_questions = skill_matcher.generate_probing_questions(
                request.resume_text,
                request.job_description,
                missing_skills[:5],  # Limit to top 5 missing skills
            )
        
        return {
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "skill_match": match_result,
            "missing_skills": missing_skills,
            "probing_questions": probing_questions,
        }
    except Exception as e:
        logger.error(f"Error in skill assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scout-jobs")
async def scout_jobs(request: JobScoutRequest):
    """
    Find and rank job listings from a company's public careers page.
    Returns a list of jobs sorted by relevance to the candidate's resume.
    """
    from backend.job_scout import find_careers_url, scrape_job_listings, rank_jobs

    careers_url = find_careers_url(request.company_name)
    if not careers_url:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find a careers page for '{request.company_name}'. "
                   "Try a more specific company name.",
        )

    jobs = scrape_job_listings(careers_url)
    if not jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Found the careers page ({careers_url}) but could not extract job listings. "
                   "The page may be JavaScript-rendered.",
        )

    ranked = rank_jobs(jobs, request.resume_text, local_model_service)
    return {
        "careers_url": careers_url,
        "total_found": len(ranked),
        "jobs": ranked[:20],
    }


@app.post("/tailor-for-job")
async def tailor_for_job(request: TailorRequest):
    """
    Generate an ATS-optimized resume and cover letter tailored to a specific job.
    Also returns a grammar/formatting audit and before/after ATS scores.
    """
    try:
        prompt_template = load_prompt("tailor_for_job_prompt.txt")
        prompt = prompt_template.format(
            resume_text=request.resume_text,
            job_title=request.job_title,
            company_name=request.company_name,
            job_description=request.job_description or f"Position: {request.job_title} at {request.company_name}",
        )
        result = local_model_service.generate_response(prompt, max_length=4096)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return {"result": result["response"]}
    except Exception as exc:
        logger.error("Error in tailor-for-job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate-experience")
async def evaluate_experience(request: SkillEvaluationRequest):
    """
    Evaluates user's responses to probing questions.
    Returns newly identified skills and resume improvement suggestions.
    """
    try:
        logger.info("Evaluating user experience responses...")
        
        evaluation = skill_matcher.evaluate_response_to_questions(
            request.user_responses,
            request.missing_skills,
        )
        
        return {
            "newly_identified_skills": evaluation["newly_identified_skills"],
            "gaps_remaining": evaluation["gaps_remaining"],
            "resume_improvements": evaluation["resume_improvements"],
            "updated_score": evaluation["updated_score"],
            "explanation": evaluation["explanation"],
        }
    except Exception as e:
        logger.error(f"Error in experience evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
