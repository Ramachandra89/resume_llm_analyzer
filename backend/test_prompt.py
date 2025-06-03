# test_prompt.py: Script to test LLaMAService model responses from the terminal with different prompts.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from resume_llm_analyzer.backend.llama_service import LLaMAService
import torch

# Load your prompt template from the correct prompts directory
PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'prompt_3.txt')
with open(PROMPT_PATH, 'r') as f:
    PROMPT_TEMPLATE = f.read()

# Example resume and job description
resume_text = """John Doe  
Email: john.doe@email.com  
Phone: (123) 456-7890  
Location: San Francisco, CA  

Professional Summary:  
Data analyst with 3 years of experience in generating business insights using SQL, Excel, and Tableau. Passionate about using data to solve real-world problems.

Experience:

Data Analyst  
ABC Retail Inc. | San Francisco, CA | Jan 2021 – Present  
- Designed and maintained Tableau dashboards to track key sales metrics.  
- Wrote complex SQL queries to analyze customer behavior and product trends.  
- Collaborated with cross-functional teams to improve reporting accuracy.  
- Reduced weekly reporting time by 40% through automation.

Business Intelligence Intern  
XYZ Tech | San Jose, CA | Jun 2020 – Dec 2020  
- Assisted in developing Power BI reports for internal stakeholders.  
- Cleaned and transformed raw data using Excel and SQL.  

Education:  
B.S. in Statistics, University of California, Berkeley | 2020  

Skills:  
- SQL, Excel, Tableau, Power BI  
- Data cleaning, exploratory analysis, visualization  
- Python (basic)
"""
job_description = """Position: Data Scientist – E-commerce Analytics  
Company: GlobalMart  

We are looking for a Data Scientist to join our e-commerce analytics team. This individual will work closely with product, marketing, and engineering teams to deliver insights that drive strategic decisions.

Responsibilities:
- Build predictive models using Python and machine learning libraries.  
- Design A/B experiments and analyze test results.  
- Automate data pipelines using SQL and Python.  
- Create intuitive visualizations using tools like Tableau or Looker.

Requirements:
- 3+ years of experience in data analysis or data science.  
- Proficient in Python and SQL.  
- Experience with machine learning (e.g., scikit-learn, XGBoost).  
- Strong communication and storytelling skills.  
- Bachelor's or Master's degree in a quantitative field (e.g., Statistics, Computer Science).

Nice to Have:
- Experience with cloud platforms (AWS, GCP).  
- Prior work in e-commerce or retail analytics.
"""

# Format the prompt
prompt = PROMPT_TEMPLATE.format(resume_text=resume_text, job_description=job_description)

# Initialize the LLaMA service
llama_service = LLaMAService(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

# Tokenize input
inputs = llama_service.tokenizer(prompt, return_tensors="pt").to(llama_service.device)

# Generate output
eos_token_id = llama_service.tokenizer.eos_token_id
with torch.no_grad():
    generation_output = llama_service.model.generate(
        **inputs,
        max_length=2048,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=eos_token_id
    )

# Extract only the new tokens
input_ids = inputs["input_ids"]
new_tokens = generation_output[0][input_ids.shape[-1]:]
new_text = llama_service.tokenizer.decode(new_tokens, skip_special_tokens=True)

# Truncate at first stop sequence
for stop_seq in ["</s>", "[/INST]", "<<SYS>>"]:
    if stop_seq in new_text:
        new_text = new_text.split(stop_seq)[0]
print(new_text.strip())
