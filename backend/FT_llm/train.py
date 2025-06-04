from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import torch
from utils import load_and_prepare_data

class ResumeFitDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_enc = self.tokenizer(
            item['input_text'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        target_enc = self.tokenizer(
            item['target_text'], truncation=True, padding='max_length', max_length=32, return_tensors='pt'
        )
        return {
            'input_ids': input_enc.input_ids.squeeze(),
            'attention_mask': input_enc.attention_mask.squeeze(),
            'labels': target_enc.input_ids.squeeze()
        }

def main():
    # Load and prepare data
    samples = load_and_prepare_data()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = ResumeFitDataset(samples, tokenizer)

    # Model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/t5-resume-fit",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        #evaluation_strategy="no",
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./models/t5-resume-fit")
    tokenizer.save_pretrained("./models/t5-resume-fit")

if __name__ == "__main__":
    main()
