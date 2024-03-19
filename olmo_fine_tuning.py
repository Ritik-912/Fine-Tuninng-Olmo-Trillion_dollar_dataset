import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset
from trl import SFTTrainer

print("Making the bits and bytes Configuration as:BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_double_quant=True, bnb_4bit_compute_dtype=torch.float16,)")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("configuring Model,,,,")
model_id = "allenai/OLMo-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
    cache_dir='/NS/ssdecl/work/'
)
print("model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-7B', quantization_config=bnb_config, device_map='auto', trust_remote_code=True, cache_dir='/NS/ssdecl/work/')")

print("Configuring Tokenizer,,,,")
print("tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/NS/ssdecl/work/')")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/NS/ssdecl/work/')

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(tokenizer.all_special_tokens)

print(model)

train = pd.read_excel("data/training_data/test-and-training/training_data/lab-manual-combine-train-5768.xlsx", engine='openpyxl')
test = pd.read_excel('data/training_data/test-and-training/test_data/lab-manual-combine-test-5768.xlsx', engine='openpyxl')

print(train.head(9))
print(test.head(9))

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      input_data = self.data.iloc[idx]['sentence']
      target_data = self.data.iloc[idx]['label']
      return {'input_ids': input_data, 'labels': target_data}

train_dataset = MyDataset(train)
eval_dataset = MyDataset(test)

def create_prompt(sample):
    instruction = "Below is an sentence. Write a label for that sentence from the set of labels (0:Dovish,1:Hawkish,and2:Neutral).\
    Where  Dovish sentences are any sentence that indicates future monetary policy easing.\
    Hawkish sentences are any sentence that would indicate a future monetary policy tightening.\
    Meanwhile, neutral sentences are those with mixed sentiment, indicating no change in the monetary policy, or those that were not directly related to monetary policystance."
    eos_token = "<|endoftext|>"

    full_prompt = ''
    full_prompt+= "### Instruction:" + instruction
    full_prompt += "\n" + "### Input:"
    full_prompt += "\n" + sample['input_ids']
    full_prompt += "\n" + "### Labels:"
    full_prompt += "\n" + str(sample['labels'])
    return full_prompt

print(create_prompt(train_dataset[1]))

def generate_response(prompt, model, tokenizer):
  inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
  inputs = {k: v.to('cuda') for k,v in inputs.items()}
  response = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
  return tokenizer.batch_decode(response, skip_special_tokens=True)[0]

print(generate_response("### Instruction:\nBelow is an sentence. Write a label for that sentence from the set of labels (0:Dovish,1:Hawkish,and2:Neutral).    Where  Dovish sentences are any sentence that indicates future monetary policy easing.    Hawkish sentences are any sentence that would indicate a future monetary policy tightening.    Meanwhile, neutral sentences are those with mixed sentiment, indicating no change in the monetary policy, or those that were not directly related to monetary policystance.\n\n### Input:\nInflation continued to run below the Committee's longer-run objective, held down in part by the effects of declines in energy and non-energy import prices.\n\n### Response:",
                  model,
                  tokenizer))

model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
print("prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)")

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["att_proj"],
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, features):
        tokenized_inputs = []
        attention_masks = []
        max_length = 512  # Set a smaller max_length value

        # Find the maximum length in dimension 1
        max_length_dim1 = max(len(feature['input_ids']) for feature in features)

        for feature in features:
            # Tokenize the input text data in chunks
            inputs = self.tokenizer(
                feature['input_ids'],
                padding='max_length',
                truncation=True,
                max_length=max_length_dim1,
                return_tensors="pt",
            )

            # Extract tokenized inputs and attention masks for each chunk
            tokenized_inputs.append(inputs['input_ids'].squeeze(0))
            attention_masks.append(inputs['attention_mask'].squeeze(0))

        # Pad tokenized inputs and attention masks to match the maximum length
        input_ids = torch.nn.utils.rnn.pad_sequence(tokenized_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
        }

# Create an instance of the custom data collator with mlm=False
custom_data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
  output_dir = "OLMo_finetuned_output",
  #num_train_epochs=5,
  max_steps = 500, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 4,
  warmup_steps = 0.03,
  logging_steps=10,
  save_strategy="epoch",
  #evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=3e-4,
  bf16=False,
  lr_scheduler_type='constant',
)

max_seq_length = 512
trainer = SFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=max_seq_length,
  tokenizer=tokenizer,
  packing=True,
  formatting_func=create_prompt,
  args=args,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  data_collator=custom_data_collator
)

print("Training")
trainer.train()
print("Training completed")

model.save_pretrained()
