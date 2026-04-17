"""
Resume Coach — single-file Streamlit app
Calls Llama 3.1 8B Instruct on SageMaker JumpStart directly (no FastAPI).

Usage:
    streamlit run streamlit_app.py
"""

import json
import os

import boto3
import PyPDF2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ENDPOINT = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "jumpstart-dft-llama-3-1-8b-instruct-20260417-030131",
)
REGION = os.getenv("AWS_REGION", "us-east-1")


# ── SageMaker helper ──────────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    """Call Llama 3.1 8B Instruct with the proper chat template."""
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False,
        },
    }
    resp = get_client().invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    data = json.loads(resp["Body"].read())
    if isinstance(data, list):
        return data[0].get("generated_text", "")
    return data.get("generated_text", str(data))


# ── PDF extractor ─────────────────────────────────────────────────────────────

def extract_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ── Prompts ───────────────────────────────────────────────────────────────────

COACH_SYSTEM = (
    "You are an expert resume coach and ATS specialist with 15+ years of "
    "experience in technical recruiting. You give detailed, actionable advice "
    "that helps candidates land their target roles."
)


def build_coaching_prompt(resume: str, jd: str) -> str:
    return f"""Analyze the resume below against the job description and produce a coaching report.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}

Reply in this exact format — do not skip any section:

ATS SCORE: [integer 0-100]

## Executive Summary
[2-3 sentences on overall fit and biggest opportunity]

## Key Strengths
[4-5 bullet points where the resume strongly matches the JD]

## Critical Gaps
[4-5 bullet points of missing skills or experience the JD requires]

## Rewritten Experience Bullets
[Rewrite 3-5 weak resume bullets with specific numbers and impact]

## Recommended Actions
[3-5 concrete steps the candidate should take to improve their chances]

## Cover Letter Opening
[A compelling 2-3 sentence opening paragraph tailored to this role]"""


CHAT_SYSTEM = (
    "You are an expert resume coach. The candidate has just received a "
    "coaching report (shown below). Answer their follow-up questions "
    "concisely and specifically. Always give actionable advice."
)


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume Coach",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume Coach")
st.caption(f"Powered by Llama 3.1 8B Instruct · SageMaker JumpStart · `{ENDPOINT}`")

# Session state
for key, default in [("report", None), ("messages", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Input section ─────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Your Resume")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    resume_text = ""
    if uploaded:
        resume_text = extract_pdf(uploaded)
        with st.expander("Preview extracted text"):
            st.text(resume_text[:1500] + ("…" if len(resume_text) > 1500 else ""))

with col_right:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the full job description",
        height=320,
        placeholder="Copy and paste the job posting here…",
    )

generate = st.button(
    "🚀 Generate Coaching Report",
    type="primary",
    disabled=not (resume_text and jd_text),
)

if generate:
    with st.spinner("Analyzing your resume against the job description… (~30–60 s)"):
        try:
            report = call_llm(
                COACH_SYSTEM,
                build_coaching_prompt(resume_text, jd_text),
                max_tokens=1200,
            )
            st.session_state.report = report
            st.session_state.messages = []   # reset chat for new report
        except Exception as e:
            st.error(f"SageMaker error: {e}")


# ── Report section ────────────────────────────────────────────────────────────

if st.session_state.report:
    st.divider()
    st.subheader("📊 Coaching Report")

    # Extract and display ATS score as a metric
    report_text = st.session_state.report
    ats_score = None
    for line in report_text.splitlines():
        if "ATS SCORE" in line.upper():
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                ats_score = int(digits[:3])   # cap at 3 digits
            break

    if ats_score is not None:
        color = (
            "normal" if ats_score >= 70
            else "off"
        )
        st.metric(
            label="ATS Match Score",
            value=f"{ats_score} / 100",
            delta="Strong match" if ats_score >= 70 else (
                "Moderate match" if ats_score >= 50 else "Needs work"
            ),
            delta_color=color,
        )

    st.markdown(report_text)

    # ── Chat section ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Chat with Your Coach")
    st.caption("Ask follow-up questions about your report — e.g. *How do I close the skill gaps?*")

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New message input
    if user_input := st.chat_input("Ask the coach anything about your report…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build context: report + conversation history
        history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.messages[:-1]
        )
        context = (
            f"Coaching report:\n{st.session_state.report}\n\n"
            + (f"Conversation so far:\n{history}\n\n" if history else "")
            + f"Candidate: {user_input}"
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = call_llm(CHAT_SYSTEM, context, max_tokens=512)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
