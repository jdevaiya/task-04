from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Step 1: Load the Dataset
# Use a public dataset or replace "wikitext" with your custom dataset path
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Step 2: Load the Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Step 3: Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Prepare Data for Training
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 5: Load the Pre-trained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,  # Set to True if you want to upload to Hugging Face Hub
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 8: Train the Model
trainer.train()

# Step 9: Generate Text
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test Text Generation
prompt = "The future of artificial intelligence is"
print(generate_text(prompt))

