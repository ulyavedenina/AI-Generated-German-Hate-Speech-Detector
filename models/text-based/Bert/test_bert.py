import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
import numpy as np
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Bert Model")

# Add an argument
parser.add_argument('model', type=str, help='bert-base resp. bert-Large')

# Parse the arguments
args = parser.parse_args()

if args.model == "bert-base":
    BERT_MODEL = 'dbmdz/bert-base-german-uncased'
elif args.model =="bert-large":
    BERT_MODEL = 'deepset/gbert-large'
else:
    raise Exception("Argument required: bert-base, bert-large")

def tokenize_sentences(sentences, tokenizer, max_length=256):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def evaluate_model(model, dataloader, device):
    model.eval()

    all_predictions, all_true_labels, all_tokenized_texts = [], [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=True)

        logits = result.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        all_predictions.extend(logits)
        all_true_labels.extend(label_ids)
        all_tokenized_texts.extend(tokenizer.batch_decode(b_input_ids, skip_special_tokens=True))

    return np.array(all_predictions), np.array(all_true_labels), all_tokenized_texts

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./dataset/text-based/test.tsv", delimiter='\t')
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
sentences = df.text_light_clean.values
labels = df.label.values

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

# Tokenize all of the sentences
input_ids, attention_masks = tokenize_sentences(sentences, tokenizer)

# Convert the lists into tensors.
labels = torch.tensor(labels)

# Set the batch size.
BATCH_SIZE = 64

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

# Load a trained model and vocabulary that you have fine-tuned
bert_dir = './models/text-based/Bert/'
MODEL_PATH = f'{bert_dir}{args.model}'
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model.to(device)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def f_score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat)

predictions, true_labels, tokenized_texts = evaluate_model(model, prediction_dataloader, device)
accuracy = flat_accuracy(predictions, true_labels)
f1score = f_score(predictions, true_labels)

# Write results to a file
output_file_path = f'{bert_dir}test_results_{args.model}.csv'
results_df = pd.DataFrame({
    'Tokenized_Text': tokenized_texts,
    'True_Label': true_labels,
    'Predicted_Label': np.argmax(predictions, axis=1),
})

results_df.to_csv(output_file_path, index=False)

print(f'Results written to: {output_file_path}')
print('Accuracy:', accuracy)
print('F1 Score:', f1score)
print('DONE.')
