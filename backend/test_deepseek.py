import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from resume_parser import extract_text_from_pdf

# Model name
model_name = "deepseek-ai/deepseek-llm-7b-base"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)

# Verify GPU usage
print(f"Model loaded on: {model.device}")

# Get resume path and extract text
resume_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resume_List", "Resume_Taduri.pdf")
resume_text = extract_text_from_pdf(resume_path)
job_description = input("Enter the job description: ").strip()

# Load prompt template from file
prompt_template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "resume_analysis_prompt.txt")
with open(prompt_template_path, 'r') as f:
    prompt_template = f.read()

# Format the prompt with the resume and job description
prompt = prompt_template.format(resume_text=resume_text, job_description=job_description)

# DeepSeek-R1-0528 supports up to 64K tokens context window and very long outputs
MAX_INPUT_TOKENS = 64000
MAX_OUTPUT_TOKENS = 4096

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate output
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKENS, temperature=0.2)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\nLLM Response:\n")
print(output_text) 

## Deepseek is useless for this task

## Output venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ 
"""(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ What You Will Do:
Command 'What' not found, did you mean:
  command 'jhat' from deb openjdk-8-jdk-headless (8u442-b06~us1-0ubuntu1~22.04)
  command 'phat' from deb phat-utils (1.6-2build2)
  command 'chat' from deb ppp (2.4.9-1+1ubuntu3)
Try: sudo apt install <deb name>
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Work on ML Solution for RxQuality for MFC Project
Command 'Work' not found, did you mean:
  command 'zork' from snap zork (1.0.2)
See 'snap info <snapname>' for additional versions.
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ 
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Required Skills:
Required: command not found
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Required:
Required:: command not found
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ 7+ years of hands-on experience in applied machine learning, deep learning, and AI system deployment
7+: command not found
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Strong Python engineering background with ML/DL frameworks: TensorFlow, PyTorch, Keras, OpenCV
Strong: command not found
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Proven experience in Computer Vision tasks, including object detection, segmentation, and OCR
Proven: command not found
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Experience training and fine-tuning models such as: YOLOv5/v8, EfficientNet, Faster-RCNN, TrOCR, Vision Transformers (ViT)
bash: syntax error near unexpected token `('
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Practical experience building and serving REST APIs for inference (TF Serving, TorchServe, FastAPI)
bash: syntax error near unexpected token `('
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Hands-on with MLOps tools: DVC, MLflow, Git, CI/CD, containerization (Docker/Kubernetes)
bash: syntax error near unexpected token `('
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Cloud deployment experience (Azure preferred; AWS or GCP acceptable)
bash: syntax error near unexpected token `('
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ LLM/GenAI experience: building, fine-tuning, or prompting models such as GPT-4, LLaMA, Claude, etc.
bash: LLM/GenAI: No such file or directory
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Familiarity with RAG (Retrieval-Augmented Generation) pipelines and integration into enterprise systems
bash: syntax error near unexpected token `('
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Understanding of Agentic AI architectures (e.g., LangChain, CrewAI, AutoGPT) for orchestrated task agents or workflow automation
bash: syntax error near unexpected token `('
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ Strong foundations in statistics, optimization, and deep learning principles
Strong: command not found
(venv) ubuntu@ip-172-31-71-121:~/resume_llm_analyzer/resume_llm_analyzer/backend$ """