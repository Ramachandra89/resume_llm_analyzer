import os
import replicate 
from dotenv import load_dotenv
from resume_parser import extract_text_from_pdf

load_dotenv(dotenv_path="/Users/rahultaduri/Interview_Kickstart/Capstone_ResumeCoach/resume_llm_analyzer/.env")

#Set replicate API Key
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

#Get resume path and extract text
resume_path = input("Enter the path of the resume in pdf format: ").strip()
resume_text = extract_text_from_pdf(resume_path)
job_description = input("Enter the job description: ").strip()

# Load prompt template from file
prompt_template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "resume_analysis_prompt.txt")
with open(prompt_template_path, 'r') as f:
    prompt_template = f.read()

# Format the prompt with the resume and job description
prompt = prompt_template.format(resume_text=resume_text, job_description=job_description)

# Call the model
output = replicate.run(
    "openai/gpt-4o-mini",
    input={
        "prompt": prompt,
        "temperature": 0.2,
    }
)

print("\nLLM Response:\n")
print("".join(output))