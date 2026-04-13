import os
import tempfile

import requests
import streamlit as st
from backend.resume_parser import extract_text_from_pdf

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume LLM Analyzer", layout="wide")
st.title("Resume & JD Analyzer + Career Coach")

TABS = ["Resume Analyzer", "Career Coach Chat", "Company Job Scout"]
tab1, tab2, tab3 = st.tabs(TABS)

# ── Tab 1: Resume Analyzer ────────────────────────────────────────────────────
with tab1:
    st.header("Resume & Job Description Analyzer")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Paste Job Description", height=200)
    submit = st.button("Analyze Resume")
    if submit:
        if not resume_file or not job_desc:
            st.warning("Please upload a resume and provide a job description.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name
            resume_text = extract_text_from_pdf(tmp_path)
            os.remove(tmp_path)
            payload = {
                "resume_text": resume_text,
                "job_description": job_desc
            }
            with st.spinner("Analyzing with SageMaker..."):
                try:
                    response = requests.post(f"{BACKEND_API_URL}/analyze", json=payload, timeout=120)
                    response.raise_for_status()
                    result = response.json()
                    st.markdown(result.get("analysis", "No analysis returned."))
                except requests.RequestException as exc:
                    st.error(f"Analysis request failed: {exc}")

# ── Tab 2: Career Coach Chat ──────────────────────────────────────────────────
with tab2:
    st.header("Career Coach Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send", key="send_btn") and user_input:
        st.session_state.chat_history.append(("user", user_input))
        conversation = "\n".join([
            ("User: " + msg if role == "user" else "Coach: " + msg)
            for role, msg in st.session_state.chat_history
        ])
        payload = {"conversation": conversation}
        with st.spinner("Getting coach response..."):
            try:
                response = requests.post(f"{BACKEND_API_URL}/chat", json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                bot_reply = data.get("response", "")
            except requests.RequestException as exc:
                bot_reply = f"[Error: {exc}]"
        st.session_state.chat_history.append(("coach", bot_reply))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Coach:** {msg}")

# ── Tab 3: Company Job Scout ──────────────────────────────────────────────────
with tab3:
    st.header("Company Job Scout")
    st.markdown(
        "Enter a company name and upload your resume. "
        "The agent will scrape their careers page, find relevant roles, "
        "rank them by fit, and tailor your resume + cover letter for any role you select."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        scout_resume_file = st.file_uploader(
            "Upload Your Resume (PDF)", type=["pdf"], key="scout_resume_file"
        )

    with col_right:
        company_input = st.text_input(
            "Company Name",
            placeholder="e.g. Google, Stripe, OpenAI",
            key="company_input",
        )

    find_btn = st.button("Find Matching Jobs", key="scout_find_btn")

    if find_btn:
        if not scout_resume_file:
            st.warning("Please upload your resume.")
        elif not company_input.strip():
            st.warning("Please enter a company name.")
        else:
            # Parse resume
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(scout_resume_file.read())
                tmp_path = tmp.name
            resume_text = extract_text_from_pdf(tmp_path)
            os.remove(tmp_path)
            st.session_state["scout_resume_text"] = resume_text
            st.session_state["scout_company"] = company_input.strip()

            # Call backend
            with st.spinner(f"Searching {company_input.strip()} careers page and ranking roles..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_API_URL}/scout-jobs",
                        json={
                            "company_name": company_input.strip(),
                            "resume_text": resume_text,
                        },
                        timeout=90,
                    )
                    resp.raise_for_status()
                    st.session_state["scout_results"] = resp.json()
                    st.session_state["selected_job"] = None
                    st.session_state["tailor_result"] = None
                except requests.RequestException as exc:
                    st.error(f"Scout failed: {exc}")

    # ── Job results ────────────────────────────────────────────────────────────
    if st.session_state.get("scout_results"):
        data = st.session_state["scout_results"]
        company = st.session_state.get("scout_company", "")
        careers_url = data.get("careers_url", "")
        jobs = data.get("jobs", [])

        st.success(
            f"Found **{data.get('total_found', len(jobs))} relevant roles** at {company}. "
            f"[View careers page]({careers_url})"
        )

        st.subheader("Top Matches (ranked by resume fit)")

        for job in jobs:
            score = job.get("score", "?")
            title = job.get("title", "Unknown Role")
            url = job.get("url", "#")

            # Color-code the score badge
            if isinstance(score, (int, float)):
                badge_color = "green" if score >= 70 else ("orange" if score >= 45 else "red")
            else:
                badge_color = "gray"

            col_title, col_score, col_btn = st.columns([5, 1, 2])
            with col_title:
                st.markdown(f"[{title}]({url})")
            with col_score:
                st.markdown(
                    f"<span style='color:{badge_color}; font-weight:bold'>{score}%</span>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("Tailor Resume", key=f"tailor_btn_{url[-40:]}"):
                    st.session_state["selected_job"] = job
                    st.session_state["tailor_result"] = None

        # ── Tailoring panel ────────────────────────────────────────────────────
        if st.session_state.get("selected_job"):
            job = st.session_state["selected_job"]
            st.divider()
            st.subheader(f"Tailoring for: {job.get('title', '')}")
            st.markdown(f"**Job URL:** [{job.get('url', '')}]({job.get('url', '')})")

            job_desc_tailor = st.text_area(
                "Paste the full job description here (strongly recommended for best results)",
                value=job.get("snippet", ""),
                height=180,
                key="job_desc_tailor",
            )

            gen_btn = st.button(
                "Generate ATS Resume + Cover Letter", key="gen_tailor_btn"
            )

            if gen_btn:
                resume_text = st.session_state.get("scout_resume_text", "")
                if not resume_text:
                    st.warning("Resume text is missing — please re-upload your resume above.")
                else:
                    with st.spinner(
                        "Crafting your ATS-optimized resume, checking grammar, and writing cover letter..."
                    ):
                        payload = {
                            "resume_text": resume_text,
                            "job_title": job.get("title", ""),
                            "job_url": job.get("url", ""),
                            "job_description": job_desc_tailor or job.get("snippet", ""),
                            "company_name": st.session_state.get("scout_company", ""),
                        }
                        try:
                            resp = requests.post(
                                f"{BACKEND_API_URL}/tailor-for-job",
                                json=payload,
                                timeout=180,
                            )
                            resp.raise_for_status()
                            st.session_state["tailor_result"] = resp.json().get("result", "")
                        except requests.RequestException as exc:
                            st.error(f"Tailoring failed: {exc}")

            if st.session_state.get("tailor_result"):
                result_text = st.session_state["tailor_result"]
                st.divider()

                # Split and display each section with nice headers
                sections = {
                    "## 1. Grammar & Formatting Audit": "Grammar & Formatting Audit",
                    "## 2. ATS-Optimized Tailored Resume": "ATS-Optimized Tailored Resume",
                    "## 3. Cover Letter": "Cover Letter",
                    "## 4. ATS Alignment Score": "ATS Alignment Score",
                }

                displayed = False
                for marker, label in sections.items():
                    if marker in result_text:
                        displayed = True

                if displayed:
                    st.markdown(result_text)
                else:
                    st.markdown(result_text)

                st.download_button(
                    label="Download Tailored Resume & Cover Letter (.txt)",
                    data=result_text,
                    file_name=f"tailored_{job.get('title', 'resume').replace(' ', '_')}.txt",
                    mime="text/plain",
                )
