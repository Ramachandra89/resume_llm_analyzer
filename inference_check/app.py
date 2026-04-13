"""
Inference Check — Streamlit frontend for resume analysis via Nebius AI.

Run:
    streamlit run inference_check/app.py

Requires:
    NEBIUS_API_KEY env var, or enter it in the sidebar.
"""

import os
import sys
import tempfile

from dotenv import load_dotenv
import streamlit as st

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from the project root before anything reads env vars
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from backend.resume_parser import extract_text_from_pdf
from inference_check.nebius_service import NebiusService

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Inference Check — Resume Coach", layout="wide")
st.title("Inference Check — Resume Coach")

# ---------------------------------------------------------------------------
# Sidebar — model selection & settings
# ---------------------------------------------------------------------------
nebius_key = os.getenv("NEBIUS_API_KEY", "")

with st.sidebar:
    st.header("Model Settings")
    nebius_model = st.selectbox(
        "LLM",
        NebiusService.available_models(),
        index=0,
        help="All models run on Nebius AI (eu-north1)",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.6, 0.05)
    st.caption(f"Provider: Nebius AI")

st.caption(f"Model: `{nebius_model}`")


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------
def _service() -> NebiusService:
    if not nebius_key:
        st.error("Enter your Nebius API key in the sidebar.")
        st.stop()
    return NebiusService(api_key=nebius_key, model=nebius_model, temperature=temperature)


def call_llm(prompt: str, max_tokens: int = 2048) -> str:
    result = _service().generate(prompt, max_tokens=max_tokens)
    if result["status"] == "error":
        st.error(f"Nebius error: {result.get('error')}")
        st.stop()
    if result.get("truncated"):
        st.warning("Response was cut off (hit token limit). Consider reducing prompt length.")
    return result["response"]


def call_llm_chat(messages: list[dict], max_tokens: int = 1024) -> str:
    result = _service().chat(messages, max_tokens=max_tokens)
    if result["status"] == "error":
        st.error(f"Nebius error: {result.get('error')}")
        st.stop()
    if result.get("truncated"):
        st.warning("Response was cut off (hit token limit).")
    return result["response"]


# ---------------------------------------------------------------------------
# Load the analysis prompt template
# ---------------------------------------------------------------------------
_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
    "resume_analysis_prompt.txt",
)
with open(_PROMPT_PATH) as fh:
    _ANALYSIS_PROMPT_TEMPLATE = fh.read()


def build_analysis_prompt(resume_text: str, job_description: str) -> str:
    return _ANALYSIS_PROMPT_TEMPLATE.replace("{resume_text}", resume_text).replace(
        "{job_description}", job_description
    )


# ---------------------------------------------------------------------------
# Skill helpers (no class dependency on LocalModelService)
# ---------------------------------------------------------------------------

def extract_skills(text: str, context: str = "resume") -> dict:
    prompt = f"""Extract all technical and professional skills from the following {context}.
Categorize them as: Technical Skills, Soft Skills, Tools & Platforms, Certifications, Languages.
Return a CONCISE list with no explanations.

{context.upper()}:
{text}

Format your response EXACTLY as:
Technical Skills: [skill1, skill2, ...]
Soft Skills: [skill1, skill2, ...]
Tools & Platforms: [skill1, skill2, ...]
Certifications: [skill1, skill2, ...]
Languages: [skill1, skill2, ...]
"""
    raw = call_llm(prompt, max_tokens=512)
    skills: dict[str, list[str]] = {
        "Technical Skills": [], "Soft Skills": [],
        "Tools & Platforms": [], "Certifications": [], "Languages": [],
    }
    for line in raw.split("\n"):
        line = line.strip()
        for cat in skills:
            if line.startswith(cat + ":"):
                skills[cat] = [s.strip() for s in line.replace(cat + ":", "").split(",") if s.strip()]
    return skills


def match_skills(resume_skills: dict, job_skills: dict) -> dict:
    def fmt(s: dict) -> str:
        return "\n".join(f"{k}: {', '.join(v)}" for k, v in s.items() if v)

    prompt = f"""Analyze the skill match between resume and job description.

RESUME SKILLS:
{fmt(resume_skills)}

REQUIRED JOB SKILLS:
{fmt(job_skills)}

For each category list MATCHED, MISSING, and BONUS skills.

Format your response EXACTLY as:
TECHNICAL SKILLS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

SOFT SKILLS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

TOOLS & PLATFORMS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

CERTIFICATIONS:
Matched: [skills]
Missing: [skills]
Bonus: [skills]

Overall Match Score (0-100): [score]
Explanation: [brief explanation]
"""
    raw = call_llm(prompt, max_tokens=1024)
    result: dict = {"matched": {}, "missing": {}, "bonus": {}, "overall_score": 0, "explanation": ""}
    cat = None
    for line in raw.split("\n"):
        line = line.strip()
        if "Overall Match Score" in line:
            try:
                result["overall_score"] = int(line.split(":")[-1].strip().split("/")[0])
            except ValueError:
                pass
        elif "Explanation:" in line:
            result["explanation"] = line.split(":", 1)[-1].strip()
        elif "TECHNICAL SKILLS:" in line:
            cat = "Technical"
        elif "SOFT SKILLS:" in line:
            cat = "Soft"
        elif "TOOLS & PLATFORMS:" in line:
            cat = "Tools"
        elif "CERTIFICATIONS:" in line:
            cat = "Certifications"
        elif cat and line.startswith("Matched:"):
            result["matched"][cat] = [s.strip() for s in line.replace("Matched:", "").split(",") if s.strip()]
        elif cat and line.startswith("Missing:"):
            result["missing"][cat] = [s.strip() for s in line.replace("Missing:", "").split(",") if s.strip()]
        elif cat and line.startswith("Bonus:"):
            result["bonus"][cat] = [s.strip() for s in line.replace("Bonus:", "").split(",") if s.strip()]
    return result


def generate_probing_questions(resume_text: str, job_desc: str, missing: list[str]) -> dict:
    prompt = f"""Based on the user's resume and job requirements, generate 3-5 intelligent probing questions.
These should explore if the user has experience with missing required skills.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_desc}

MISSING SKILLS:
{', '.join(missing)}

Format:
Question 1: [question]
Motivation: [why this matters]

Question 2: [question]
Motivation: [why this matters]

Question 3: [question]
Motivation: [why this matters]
"""
    raw = call_llm(prompt, max_tokens=768)
    questions: dict[int, dict] = {}
    cur = None
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("Question "):
            try:
                num = int(line.split(":")[0].replace("Question", "").strip())
                questions[num] = {"question": line.split(":", 1)[-1].strip(), "motivation": ""}
                cur = num
            except ValueError:
                pass
        elif line.startswith("Motivation:") and cur:
            questions[cur]["motivation"] = line.split(":", 1)[-1].strip()
    return questions


def evaluate_responses(user_responses: dict, missing_skills: list[str]) -> dict:
    responses_text = "\n".join(f"Q{i}: {r}" for i, r in user_responses.items())
    prompt = f"""Based on the user's responses, assess which missing skills are now covered.

MISSING SKILLS:
{', '.join(missing_skills)}

USER RESPONSES:
{responses_text}

Format:
NEWLY IDENTIFIED SKILLS:
[list]

GAPS STILL REMAINING:
[list]

RECOMMENDED RESUME IMPROVEMENTS:
[suggestions]

UPDATED MATCH SCORE: [0-100]
EXPLANATION: [why score changed]
"""
    raw = call_llm(prompt, max_tokens=1024)
    result: dict = {
        "newly_identified_skills": [], "gaps_remaining": [],
        "resume_improvements": "", "updated_score": 0, "explanation": "",
    }
    section = None
    for line in raw.split("\n"):
        line = line.strip()
        if "NEWLY IDENTIFIED SKILLS:" in line:
            section = "new"
        elif "GAPS STILL REMAINING:" in line:
            section = "gaps"
        elif "RECOMMENDED RESUME IMPROVEMENTS:" in line:
            section = "improve"
        elif "UPDATED MATCH SCORE:" in line:
            try:
                result["updated_score"] = int(line.split(":")[-1].strip().split("/")[0])
            except ValueError:
                pass
            section = None
        elif "EXPLANATION:" in line:
            result["explanation"] = line.split(":", 1)[-1].strip()
        elif section == "new" and line:
            result["newly_identified_skills"].append(line)
        elif section == "gaps" and line:
            result["gaps_remaining"].append(line)
        elif section == "improve" and line:
            result["resume_improvements"] += line + "\n"
    return result


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Resume Analyzer", "Skill Assessment", "Career Coach Chat"])

# ── Tab 1: Resume Analyzer ──────────────────────────────────────────────────
with tab1:
    st.header("Resume & Job Description Analyzer")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        if resume_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name
            resume_text = extract_text_from_pdf(tmp_path)
            os.remove(tmp_path)
            st.success(f"Resume uploaded ({len(resume_text)} characters)")
            st.session_state.resume_text = resume_text

    with col2:
        st.subheader("Job Description")
        job_desc = st.text_area("Paste Job Description", height=250)
        if job_desc:
            st.success(f"Job description provided ({len(job_desc)} characters)")

    if st.button("Analyze Resume & Generate Coaching Report", use_container_width=True):
        if not resume_file or not job_desc:
            st.warning("Please upload a resume and provide a job description.")
        else:
            with st.spinner("Analyzing with Nebius AI..."):
                analysis = call_llm(
                    build_analysis_prompt(st.session_state.get("resume_text", ""), job_desc),
                    max_tokens=3000,
                )
                st.session_state.job_description = job_desc
                st.session_state.last_analysis = analysis
            st.markdown("---")
            st.caption(f"Generated by `{nebius_model}`")
            st.markdown("## Coaching Report")
            st.markdown(analysis)

# ── Tab 2: Skill Assessment ─────────────────────────────────────────────────
with tab2:
    st.header("Skill Assessment & Matching")

    if "resume_text" not in st.session_state or "job_description" not in st.session_state:
        st.info("Upload a resume and job description in the 'Resume Analyzer' tab first.")
    else:
        if st.button("Start Skill Assessment", use_container_width=True):
            with st.spinner("Extracting and matching skills with Nebius AI..."):
                resume_skills = extract_skills(st.session_state.resume_text, "resume")
                job_skills = extract_skills(st.session_state.job_description, "job description")
                match = match_skills(resume_skills, job_skills)
                all_missing = [s for skills in match.get("missing", {}).values() for s in skills]
                questions = generate_probing_questions(
                    st.session_state.resume_text,
                    st.session_state.job_description,
                    all_missing,
                )
                st.session_state.skill_assessment = {
                    "skill_match": match,
                    "probing_questions": questions,
                    "missing_skills": all_missing,
                }
                st.session_state.assessment_model = nebius_model
                st.rerun()

        if "skill_assessment" in st.session_state:
            assessment = st.session_state.skill_assessment
            match_result = assessment["skill_match"]

            st.caption(f"Generated by `{st.session_state.get('assessment_model', nebius_model)}`")
            col1, col2, col3 = st.columns(3)
            matched_count = sum(len(v) for v in match_result.get("matched", {}).values())
            missing_count = sum(len(v) for v in match_result.get("missing", {}).values())
            col1.metric("Overall Match Score", f"{match_result.get('overall_score', 0)}/100")
            col2.metric("Matched Skills", matched_count)
            col3.metric("Missing Skills", missing_count)

            st.markdown("---")
            with st.expander("Matched Skills (Your strengths)"):
                for cat, skills in match_result.get("matched", {}).items():
                    if skills:
                        st.write(f"**{cat}:** {', '.join(skills)}")
            with st.expander("Missing Skills (Growth areas)", expanded=True):
                for cat, skills in match_result.get("missing", {}).items():
                    if skills:
                        st.write(f"**{cat}:** {', '.join(skills)}")
            with st.expander("Bonus Skills (Extra strengths)"):
                for cat, skills in match_result.get("bonus", {}).items():
                    if skills:
                        st.write(f"**{cat}:** {', '.join(skills)}")

            st.markdown("---")
            questions = assessment.get("probing_questions", {})
            if questions:
                st.markdown("## Personalized Coaching Questions")
                user_responses = {}
                for q_num, q_data in sorted(questions.items()):
                    st.markdown(f"**Question {q_num}:** {q_data['question']}")
                    st.caption(f"Why: {q_data.get('motivation', '')}")
                    user_responses[q_num] = st.text_area(
                        f"Your answer to Question {q_num}:", key=f"q_{q_num}", height=100
                    )

                if st.button("Evaluate Responses & Get Updated Score", use_container_width=True):
                    if not any(user_responses.values()):
                        st.warning("Answer at least one question.")
                    else:
                        with st.spinner("Evaluating with Nebius AI..."):
                            evaluation = evaluate_responses(
                                {k: v for k, v in user_responses.items() if v},
                                assessment.get("missing_skills", []),
                            )
                            st.session_state.skill_evaluation = evaluation
                            st.session_state.evaluation_model = nebius_model
                            st.rerun()

            if "skill_evaluation" in st.session_state:
                ev = st.session_state.skill_evaluation
                st.markdown("---")
                st.caption(f"Generated by `{st.session_state.get('evaluation_model', nebius_model)}`")
                st.markdown("## Evaluation Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Updated Match Score: {ev['updated_score']}/100**\n\n{ev['explanation']}")
                with col2:
                    with st.expander("Newly Identified Skills"):
                        for s in ev["newly_identified_skills"]:
                            st.write(f"• {s}")
                    with st.expander("Remaining Gaps"):
                        if ev["gaps_remaining"]:
                            for g in ev["gaps_remaining"]:
                                st.write(f"• {g}")
                        else:
                            st.write("No significant gaps remaining!")
                with st.expander("Resume Improvement Suggestions"):
                    st.write(ev["resume_improvements"])

# ── Tab 3: Career Coach Chat ─────────────────────────────────────────────────
with tab3:
    st.header("Career Coach Chat")
    st.markdown("Ask the AI coach anything about your resume, the role, or job search strategy.")

    if "chat_messages" not in st.session_state:
        context = ""
        if "resume_text" in st.session_state:
            context += f"\n\nCandidate resume (excerpt):\n{st.session_state.resume_text[:800]}..."
        if "job_description" in st.session_state:
            context += f"\n\nJob description (excerpt):\n{st.session_state.job_description[:400]}..."
        st.session_state.chat_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert career coach helping a candidate improve their resume "
                    "and prepare for job applications. Be specific, practical, and encouraging."
                    + context
                ),
            }
        ]

    for msg in st.session_state.chat_messages:
        if msg["role"] == "system":
            continue
        label = "You" if msg["role"] == "user" else "Coach"
        st.markdown(f"**{label}:** {msg['content']}")
        if msg["role"] == "assistant":
            st.caption(f"`{msg.get('model', nebius_model)}`")

    user_input = st.text_input("Your message:", key="chat_input")
    col1, col2 = st.columns([4, 1])
    with col1:
        send = st.button("Send", key="send_btn", use_container_width=True)
    with col2:
        if st.button("Clear", key="clear_btn", use_container_width=True):
            del st.session_state["chat_messages"]
            st.rerun()

    if send:
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.spinner("Coach is thinking..."):
                reply = call_llm_chat(st.session_state.chat_messages, max_tokens=1024)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply, "model": nebius_model})
            st.rerun()
