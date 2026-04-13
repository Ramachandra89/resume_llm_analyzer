import os
import tempfile
from json import JSONDecodeError

import requests
import streamlit as st
from backend.resume_parser import extract_text_from_pdf

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume LLM Analyzer", layout="wide")
st.title("🎯 Resume Coach - AI Career Coach")

TABS = ["Resume Analyzer", "Skill Assessment", "Career Coach Chat"]
tab1, tab2, tab3 = st.tabs(TABS)

with tab1:
    st.header("📄 Resume & Job Description Analyzer")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        if resume_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name
            resume_text = extract_text_from_pdf(tmp_path)
            os.remove(tmp_path)
            st.success(f"✓ Resume uploaded ({len(resume_text)} characters)")
            st.session_state.resume_text = resume_text
    
    with col2:
        st.subheader("Job Description")
        job_desc = st.text_area("Paste Job Description", height=250)
        if job_desc:
            st.success(f"✓ Job description provided ({len(job_desc)} characters)")
    
    submit = st.button("🔍 Analyze Resume & Generate Coaching Report", use_container_width=True)
    
    if submit:
        if not resume_file or not job_desc:
            st.warning("Please upload a resume and provide a job description.")
        else:
            with st.spinner("🤖 Analyzing with the local AI model..."):
                try:
                    payload = {
                        "resume_text": resume_text,
                        "job_description": job_desc
                    }
                    response = requests.post(f"{BACKEND_API_URL}/analyze", json=payload, timeout=120)
                    response.raise_for_status()
                    result = response.json()
                    
                    st.session_state.job_description = job_desc
                    st.session_state.last_analysis = result.get("analysis", "")
                    
                    st.markdown("---")
                    st.markdown("## 📊 Coaching Report")
                    st.markdown(result.get("analysis", "No analysis returned."))
                except requests.RequestException as exc:
                    st.error(f"❌ Analysis request failed: {exc}")

with tab2:
    st.header("🎓 Skill Assessment & Matching")
    
    if "resume_text" not in st.session_state or "job_description" not in st.session_state:
        st.info("ℹ️ Please first upload a resume and job description in the 'Resume Analyzer' tab")
    else:
        st.markdown(
            "Get detailed skill matching analysis and personalized coaching questions "
            "to bridge any gaps in your qualifications."
        )
        
        assess_btn = st.button("📈 Start Skill Assessment", use_container_width=True)
        
        if assess_btn:
            with st.spinner("🔍 Analyzing skills and generating personalized questions..."):
                try:
                    payload = {
                        "resume_text": st.session_state.resume_text,
                        "job_description": st.session_state.job_description,
                    }
                    response = requests.post(
                        f"{BACKEND_API_URL}/skill-assessment",
                        json=payload,
                        timeout=120,
                    )
                    response.raise_for_status()
                    assessment = response.json()
                    st.session_state.skill_assessment = assessment
                    
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"❌ Skill assessment failed: {exc}")
        
        if "skill_assessment" in st.session_state:
            assessment = st.session_state.skill_assessment
            
            # Display skill matching metrics
            match_result = assessment.get("skill_match", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "📊 Overall Match Score",
                    f"{match_result.get('overall_score', 0)}/100",
                )
            
            with col2:
                matched_count = sum(
                    len(v) for v in match_result.get("matched", {}).values()
                )
                st.metric("✅ Matched Skills", matched_count)
            
            with col3:
                missing_count = sum(
                    len(v) for v in match_result.get("missing", {}).values()
                )
                st.metric("⚠️ Missing Skills", missing_count)
            
            st.markdown("---")
            
            # Display matched skills
            with st.expander("✅ **Matched Skills** (Your strengths for this role)"):
                for category, skills in match_result.get("matched", {}).items():
                    if skills:
                        st.write(f"**{category}:**")
                        for skill in skills:
                            st.write(f"  • {skill}")
            
            # Display missing skills
            with st.expander("⚠️ **Missing Skills** (Opportunities for growth)", expanded=True):
                for category, skills in match_result.get("missing", {}).items():
                    if skills:
                        st.write(f"**{category}:**")
                        for skill in skills:
                            st.write(f"  • {skill}")
            
            # Display bonus skills
            with st.expander("🎁 **Bonus Skills** (Additional strengths you can highlight)"):
                for category, skills in match_result.get("bonus", {}).items():
                    if skills:
                        st.write(f"**{category}:**")
                        for skill in skills:
                            st.write(f"  • {skill}")
            
            st.markdown("---")
            
            # Display probing questions
            probing_questions = assessment.get("probing_questions", {})
            if probing_questions:
                st.markdown("## 🤔 Personalized Coaching Questions")
                st.markdown(
                    "Answer these questions to help us uncover any hidden experience "
                    "that could improve your match score:"
                )
                
                user_responses = {}
                for q_num, q_data in sorted(probing_questions.items()):
                    with st.container():
                        st.markdown(f"### Question {q_num}")
                        st.write(f"**{q_data['question']}**")
                        st.caption(f"💡 Why: {q_data.get('motivation', '')}")
                        
                        response = st.text_area(
                            f"Your answer to Question {q_num}:",
                            key=f"q_{q_num}",
                            height=100,
                        )
                        user_responses[q_num] = response
                
                evaluate_btn = st.button(
                    "📊 Evaluate Responses & Get Updated Score",
                    use_container_width=True,
                )
                
                if evaluate_btn:
                    # Check if user provided responses
                    if not any(user_responses.values()):
                        st.warning("Please answer at least one question.")
                    else:
                        with st.spinner("📈 Evaluating your experience..."):
                            try:
                                payload = {
                                    "user_responses": {
                                        int(k): v
                                        for k, v in user_responses.items()
                                        if v
                                    },
                                    "missing_skills": assessment.get("missing_skills", []),
                                }
                                response = requests.post(
                                    f"{BACKEND_API_URL}/evaluate-experience",
                                    json=payload,
                                    timeout=120,
                                )
                                response.raise_for_status()
                                evaluation = response.json()
                                st.session_state.skill_evaluation = evaluation
                                
                                st.rerun()
                            except (requests.RequestException, JSONDecodeError) as exc:
                                st.error(f"❌ Evaluation failed: {exc}")
            
            if "skill_evaluation" in st.session_state:
                evaluation = st.session_state.skill_evaluation
                
                st.markdown("---")
                st.markdown("## 📊 Evaluation Results")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.info(
                        f"📈 **Updated Match Score: {evaluation['updated_score']}/100**\n\n"
                        f"{evaluation['explanation']}"
                    )
                
                with col2:
                    with st.expander("✨ **Newly Identified Skills**"):
                        for skill in evaluation["newly_identified_skills"]:
                            st.write(f"• {skill}")
                    
                    with st.expander("⚠️ **Remaining Gaps**"):
                        if evaluation["gaps_remaining"]:
                            for gap in evaluation["gaps_remaining"]:
                                st.write(f"• {gap}")
                        else:
                            st.write("🎉 No significant gaps remaining!")
                
                with st.expander("✏️ **Resume Improvement Suggestions**"):
                    st.write(evaluation["resume_improvements"])

with tab3:
    st.header("💬 Career Coach Chat")
    st.markdown("Ask the AI coach any questions about your application, resume, or the role.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_input = st.text_input("You:", key="chat_input")
    
    if st.button("Send", key="send_btn"):
        if not user_input:
            st.warning("Please enter a message.")
        else:
            st.session_state.chat_history.append(("user", user_input))
            conversation = "\n".join(
                [
                    ("User: " + msg if role == "user" else "Coach: " + msg)
                    for role, msg in st.session_state.chat_history
                ]
            )
            payload = {"conversation": conversation}
            with st.spinner("🤔 Coach is thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_API_URL}/chat", json=payload, timeout=120
                    )
                    response.raise_for_status()
                    data = response.json()
                    bot_reply = data.get("response", "")
                except requests.RequestException as exc:
                    bot_reply = f"❌ Error: {exc}"
            
            st.session_state.chat_history.append(("coach", bot_reply))
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**👤 You:** {msg}")
        else:
            st.markdown(f"**🎓 Coach:** {msg}")

