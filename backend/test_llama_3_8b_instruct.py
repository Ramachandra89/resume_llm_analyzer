import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
torch.cuda.empty_cache()
from resume_parser import extract_text_from_pdf

# Model name
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
prompt_template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "prompt_2.txt")
with open(prompt_template_path, 'r') as f:
    prompt_template = f.read()

# Format the prompt with the resume and job description
prompt = prompt_template.format(resume_text=resume_text, job_description=job_description)

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Use eos_token_id for proper stopping
eos_token_id = tokenizer.eos_token_id

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=eos_token_id
    )

# Extract only the new tokens after the prompt
input_ids = inputs["input_ids"]
new_tokens = output_ids[0][input_ids.shape[-1]:]
new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

# Truncate at first stop sequence
for stop_seq in ["</s>", "[/INST]", "<<SYS>>"]:
    if stop_seq in new_text:
        new_text = new_text.split(stop_seq)[0]

print("\nLLM Response (new tokens only):\n")
print(new_text.strip()) 