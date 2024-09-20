import argparse
import torch
from torch.nn.functional import sigmoid
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from helper import clean_full_light


# Create the parser
parser = argparse.ArgumentParser(description="hate speech comment of human or AI")

# Add an argument
parser.add_argument('input', type=str, help='insert hate speech comment')

# Parse the arguments
args = parser.parse_args()

input_text = args.input
preprocessed_text = clean_full_light(input_text)
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/media/data/hf_models/BERT/BertLarge/model_final" # change
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True, device=device)
tokenized_dict = tokenizer.encode_plus(preprocessed_text,add_special_tokens=True,
            max_length=256,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
input_ids, attention_masks = tokenized_dict["input_ids"].to(device), tokenized_dict["attention_mask"].to(device)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
output = model(input_ids, token_type_ids=None, attention_mask=attention_masks, return_dict=True)
logits = output.logits
logits = logits.squeeze()
probability = sigmoid(logits)
if torch.round(probability[0], decimals=1) == 0.5:
    pred = "I'm very uncertain. The chances are 50:50"
elif probability[0] > probability[1]:
    pred = f"Most likely a human ({int(torch.round(probability[0]*100, decimals=0))} %)"
else:
    pred = f"Most likely AI ({int(torch.round(probability[1]*100, decimals=0))} %)"
print(pred)