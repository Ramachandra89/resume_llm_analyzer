import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st

import tempfile
from resume_llm_analyzer.backend.resume_parser import extract_text_from_pdf
from resume_llm_analyzer.backend.llama_service import LLaMAService

# Load prompt template
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'prompt_3.txt')
with open(PROMPT_PATH, 'r') as f:
    PROMPT_TEMPLATE = f.read()

llama_service = LLaMAService(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

st.set_page_config(page_title="Resume LLM Analyzer", layout="wide")
st.title("Resume & JD Analyzer + Llama Chatbot")

TABS = ["Resume Analyzer", "Llama Chatbot"]
tab1, tab2 = st.tabs(TABS)

with tab1:
    st.header("Resume & Job Description Analyzer")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Paste Job Description", height=200)
    submit = st.button("Analyze Resume")
    if submit:
        if not resume_file or not job_desc:
            st.warning("Please upload a resume and provide a job description.")
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(resume_file.read())
                    tmp_path = tmp.name
                resume_text = extract_text_from_pdf(tmp_path)
                os.remove(tmp_path)
                prompt = PROMPT_TEMPLATE.format(resume_text=resume_text, job_description=job_desc)
                with st.spinner("Analyzing with Llama..."):
                    result = llama_service.generate_response(prompt, max_length=4096)
                if result["status"] == "success":
                    st.markdown(result["response"])
                else:
                    st.error(f"Error: {result.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"Exception occurred: {e}")

with tab2:
    st.header("Llama 3 Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send", key="send_btn") and user_input:
        st.session_state.chat_history.append(("user", user_input))
        # Build conversation context
        conversation = "\n".join([
            ("User: " + msg if role == "user" else "Llama: " + msg)
            for role, msg in st.session_state.chat_history
        ])
        prompt = conversation + "\nLlama:"
        with st.spinner("Llama is thinking..."):
            result = llama_service.generate_response(prompt, max_length=512)
        if result["status"] == "success":
            llama_reply = result["response"].split("Llama:")[-1].strip()
            st.session_state.chat_history.append(("llama", llama_reply))
        else:
            st.session_state.chat_history.append(("llama", "[Error: {}]".format(result.get('message', 'Unknown error'))))
    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Llama:** {msg}") 