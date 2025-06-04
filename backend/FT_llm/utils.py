from datasets import load_dataset

def load_and_prepare_data():
    dataset = load_dataset("cnamuangtoun/resume-job-description-fit")
    data = dataset['train']

    # Label encoding: "no fit"=0, others=1
    encode_label = lambda label: 0 if label == "No Fit" else 1

    samples = []
    for item in data:
        resume = item['resume_text']
        job_desc = item['job_description_text']
        label = item['label']  # assuming the field is named 'label'

        fit_score = encode_label(label)
        input_text = f"Resume: {resume}\nJob Description: {job_desc}\n"
        target_text = str(fit_score)
        samples.append({'input_text': input_text, 'target_text': target_text})

    return samples

# %%
print(load_and_prepare_data()[0])



