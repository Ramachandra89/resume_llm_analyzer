# Resume Coach - LLM-Powered Resume Analysis & Coaching

## Overview

Resume Coach is an AI-powered application that analyzes your resume against job descriptions to provide personalized coaching advice. It leverages a self-hosted LLM via **AWS SageMaker or local EC2 model hosting** to generate detailed insights, gap analysis, and recommendations for improving your candidacy.

### Key Features
- **Resume Analysis**: Upload your resume (PDF) and provide a job description for AI-powered analysis
- **Coaching Reports**: Get detailed coaching including fit assessment, gap analysis, and unique strengths
- **Skill Metrics & Assessment** (NEW): Comprehensive skill matching with:
  - Matched skills by category (Technical, Soft, Tools, Certifications, Languages)
  - Missing required skills with opportunities for growth
  - Bonus skills in your resume that you can highlight
  - Overall fit score (0-100)
- **Interactive Gap Detection** (NEW): Answer personalized coaching questions to:
  - Uncover hidden experience related to missing skills
  - Get updated match scores based on your explanations
  - Receive resume articulation suggestions (without fabricating bulletpoints)
- **Interactive Chat**: Ask follow-up questions to the career coach for personalized guidance
- **LLM Hosting**: Self-hosted model on SageMaker or a local EC2 GPU instance ensures privacy and customization

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      User Interface (Streamlit)                      │
│                  frontend/app.py (Port 8501)                         │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend Server                          │
│                   backend/api.py (Port 8000)                        │
│  - /analyze (POST): Resume + JD → Coaching Report                  │
│  - /chat (POST): Conversation → Coach Response                     │
│  - /health (GET): Health Check                                     │
└──────────┬──────────────────────────────────┬──────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  Resume Parser               │  │  Local Model Service        │
│  (PyPDF2)                    │  │  (transformers / pipeline)  │
│  - Extract text from PDFs    │  │  - Load local model artifacts│
│  - Clean & format text       │  │  - Generate text locally    │
└──────────────────────────────┘  └──────────────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────┐
                  │  Local or Cloud LLM Service         │
                  │  (S3 / Hugging Face / SageMaker)    │
                  │  GPU-backed inference on EC2        │
                  └────────────────────────────────────┘
```

## Project Structure

```
resume_llm_analyzer/
├── backend/
│   ├── api.py                     # FastAPI server
│   ├── local_model_service.py     # Local model inference service
│   ├── sagemaker_service.py       # SageMaker client wrapper (legacy)
│   ├── resume_parser.py           # PDF extraction utilities
│   ├── FT_llm/                    # Fine-tuning code (optional)
│   ├── llama_service.py           # [DEPRECATED] Local LLM loader
│   └── __pycache__/
├── frontend/
│   ├── app.py                     # Streamlit UI (SageMaker version)
│   ├── app_v1.py                  # [DEPRECATED] Legacy version
│   └── app_v2.py                  # [DEPRECATED] Legacy version
├── prompts/
│   ├── resume_analysis_prompt.txt # Main coaching prompt
│   ├── prompt_2.txt               # [LEGACY] Alternative prompt
│   └── prompt_3.txt               # [LEGACY] Alternative prompt
├── docs/
│   ├── SAGEMAKER_DEPLOYMENT.md    # SageMaker setup guide
│   └── deployment_guide.md        # [LEGACY] General deployment
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .env.example                   # Environment template
└── __init__.py
```

## Prerequisites

- Python 3.9 or higher
- AWS Account with SageMaker permissions
- Valid AWS credentials configured locally
- 2GB+ available disk space

## Installation & Setup

### 1. Clone or Download Repository
```bash
cd /path/to/resume_llm_analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your deployment details:
```bash
cp .env.example .env
# Edit .env with your local model or SageMaker settings
```

**Environment options:**
- For local EC2 model hosting, set `MODEL_LOCAL_PATH`, `MODEL_DEVICE`, and optionally `S3_MODEL_BUCKET` / `S3_MODEL_PREFIX` or `HUGGINGFACE_MODEL_ID`
- For SageMaker deployment, set `SAGEMAKER_ENDPOINT_NAME`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`

### 4. Set Up Model Hosting

Follow the guide in [docs/SAGEMAKER_DEPLOYMENT.md](docs/SAGEMAKER_DEPLOYMENT.md) to set up either:
- a SageMaker endpoint, or
- a local EC2 model deployment using S3 or Hugging Face model artifacts

## Running the Application

### Start Backend API
```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or with reload for development:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Start Streamlit Frontend (in another terminal)
```bash
cd frontend
streamlit run app.py
```

The app will be available at: `http://localhost:8501`

## Usage

### Basic Workflow

1. **Resume Analyzer Tab**
   - Upload your resume (PDF)
   - Paste the job description
   - Click "Analyze Resume & Generate Coaching Report"
   - Read the AI coaching advice

2. **Skill Assessment Tab** (NEW)
   - Click "Start Skill Assessment"
   - View your skill metrics:
     - **Match Score**: Overall fit percentage
     - **Matched Skills**: Skills you have that are required
     - **Missing Skills**: Gaps you should address
     - **Bonus Skills**: Extra strengths to highlight in interviews
   - Answer personalized coaching questions about your experience
   - Click "Evaluate Responses & Get Updated Score"
   - Review:
     - Newly identified skills (based on your answers)
     - Remaining gaps to address
     - Resume articulation suggestions

3. **Career Coach Chat Tab**
   - Ask the AI coach questions:
     - "How can I best position my AWS experience for a GCP role?"
     - "What interview questions should I prepare for?"
     - "Are my soft skills competitive for this level?"
   - Get personalized guidance

### Smart Skill Matching Features

**Probing Questions**: Instead of requiring perfect skill matches, the app asks targeted questions to discover:
- Related experience (e.g., "Have you used any distributed training frameworks?" to assess TensorFlow readiness)
- Transferable skills (e.g., AWS experience vs. required GCP)
- Adjacent expertise (e.g., DevOps background for Kubernetes understanding)

**Interactive Gap Detection**: 
- No bullet points are fabricated or added to your resume
- You explain your actual experience
- AI identifies how your experience translates to missing skills
- You get specific resume articulation suggestions (rephrasing, emphasis, context)

**Dynamic Scoring**:
- Initial score based on exact skill matches
- Updated score after you answer probing questions
- Shows progress as hidden connections are discovered

**Skill Categories**:
- Technical Skills (Python, Machine Learning, etc.)
- Soft Skills (Leadership, Communication, etc.)
- Tools & Platforms (AWS, Docker, etc.)
- Certifications (AWS Certified Solutions Architect, etc.)
- Languages (English, Spanish, Mandarin, etc.)

## API Endpoints

### POST /analyze
Generates a coaching report based on resume and job description.

**Request:**
```json
{
  "resume_text": "John Doe\nSoftware Engineer...",
  "job_description": "We are hiring a Senior ML Engineer..."
}
```

**Response:**
```json
{
  "analysis": "Thank you for sharing your resume...[full coaching report]"
}
```

### POST /skill-assessment (NEW)
Performs comprehensive skill matching and generates probing questions.

**Request:**
```json
{
  "resume_text": "John Doe\nSkills: Python, ML, AWS...",
  "job_description": "Required: Python, TensorFlow, GCP..."
}
```

**Response:**
```json
{
  "resume_skills": {
    "Technical Skills": ["Python", "Machine Learning"],
    "Tools & Platforms": ["AWS", "Docker"]
  },
  "job_skills": {
    "Technical Skills": ["Python", "TensorFlow"],
    "Tools & Platforms": ["GCP", "Kubernetes"]
  },
  "skill_match": {
    "matched": {
      "Technical": ["Python"],
      "Tools": []
    },
    "missing": {
      "Technical": ["TensorFlow"],
      "Tools": ["GCP", "Kubernetes"]
    },
    "bonus": {
      "Tools": ["Docker", "AWS"]
    },
    "overall_score": 45,
    "explanation": "You have core ML skills but lack cloud platform and orchestration experience..."
  },
  "missing_skills": ["TensorFlow", "GCP", "Kubernetes"],
  "probing_questions": {
    1: {
      "question": "Have you worked with any distributed training frameworks like TensorFlow or PyTorch?",
      "motivation": "TensorFlow is a required skill and understanding your experience with similar frameworks helps us assess your ability to transition."
    },
    2: {
      "question": "Describe any cloud platform experience you have (AWS, GCP, Azure). What services have you used?",
      "motivation": "GCP is required for this role and we want to understand if your AWS experience could transfer."
    }
  }
}
```

### POST /evaluate-experience (NEW)
Evaluates user responses to probing questions and updates match score.

**Request:**
```json
{
  "user_responses": {
    1: "I've used PyTorch for deep learning projects and understand distributed training concepts...",
    2: "Extensive AWS experience: EC2, S3, Lambda, SageMaker. I learned GCP basics during a hackathon..."
  },
  "missing_skills": ["TensorFlow", "GCP", "Kubernetes"]
}
```

**Response:**
```json
{
  "newly_identified_skills": [
    "Deep Learning Framework Experience - High confidence",
    "Cloud Platform Fundamentals - Medium confidence",
    "Distributed Systems Understanding - Medium confidence"
  ],
  "gaps_remaining": [
    "GCP production experience",
    "Kubernetes container orchestration",
    "TensorFlow specific framework"
  ],
  "resume_improvements": "You can strengthen your resume by: 1) Creating a dedicated 'Cloud & ML Platforms' section highlighting PyTorch and AWS experience with specific metrics... 2) Emphasizing your distributed training work and how it transfers to production ML pipelines...",
  "updated_score": 68,
  "explanation": "Your PyTorch and AWS experience significantly strengthens your candidacy. Focus your interview prep on: 1) Transitioning from PyTorch to TensorFlow knowledge, 2) GCP platform familiarity."
}
```

### POST /chat
Continues a conversation with the career coach.

**Request:**
```json
{
  "conversation": "User: What should I do to improve my candidacy?\nCoach: Focus on..."
}
```

**Response:**
```json
{
  "response": "Based on your assessment, I recommend: 1) Complete a GCP fundamentals course (2-3 weeks)..."
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Deployment

### Local Development
See the setup instructions above.

### AWS EC2 Production
See [docs/SAGEMAKER_DEPLOYMENT.md](docs/SAGEMAKER_DEPLOYMENT.md) for:
- EC2 instance setup
- Systemd service configuration
- Security group rules
- Monitoring and scaling

## Configuration

### Adjusting LLM Parameters
Edit `backend/sagemaker_service.py` to modify generation parameters:
- `max_new_tokens`: Maximum length of generated response (default: 1024)
- `temperature`: Creativity (0-1, default: 0.7)
- `top_p`: Diversity (0-1, default: 0.9)

### Customizing Prompts
Edit prompts in `prompts/resume_analysis_prompt.txt` or create new prompt files and reference them in `backend/api.py`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Import Error: `ModuleNotFoundError`** | Run `pip install -r requirements.txt` |
| **SageMaker endpoint not found** | Verify `SAGEMAKER_ENDPOINT_NAME` in `.env` |
| **AWS authentication error** | Check credentials in `.env` and IAM role permissions |
| **PDF extraction fails** | Ensure PDF is valid and not password-protected |
| **Timeout on analysis** | Check endpoint status in SageMaker console; may need to increase instance size |

## Optional: Fine-Tuning

If you have training data, you can fine-tune the model using code in `backend/FT_llm/`. See documentation there for details.

## Performance Notes

- **First API call** may be slow (3-5s) due to endpoint warmup
- **Subsequent calls** are typically 1-3 seconds
- **Large documents** (>20,000 tokens) may hit output limits

## Security Considerations

- Keep `.env` file secure and do not commit to version control
- Use IAM roles instead of hardcoded credentials in production
- Enable VPC endpoints for private SageMaker communication
- Implement rate limiting and authentication for public deployments

## Future Enhancements

- [ ] Resume parsing with structured field extraction
- [ ] Pre/post matching score (cosine similarity)
- [ ] Dataset scraping from job sites
- [ ] Cover letter generation
- [ ] Interview prep module
- [ ] Batch analysis for multiple resumes

## Support

For issues or questions, please refer to [docs/SAGEMAKER_DEPLOYMENT.md](docs/SAGEMAKER_DEPLOYMENT.md) or open an issue in the repository.
