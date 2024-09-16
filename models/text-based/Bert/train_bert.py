import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
import os 
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import random
import time
import datetime
from torch import nn
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Bert Model")

# Add an argument
parser.add_argument('model', type=str, help='bert-base resp. bert-Large')

# Parse the arguments
args = parser.parse_args()

if args.model == "bert-base":
    model_id = 'dbmdz/bert-base-german-uncased'
elif args.model =="bert-large":
    model_id = 'deepset/gbert-large'
else:
    raise Exception("Argument required: bert-base, bert-large")

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print(output_dir)
# Set up output directory
output_dir = os.path.dirname(os.path.abspath(__file__)) + f'/{args.model}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Model will be saved to %s" % output_dir)

# Load data
df = pd.read_csv('./dataset/training_set.tsv', sep='\t', encoding='utf-8', engine='python')
df = df.drop(['index', 'text'], axis=1)

# Split data into training and test sets
y = df['author']
X = df.drop(['author'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenize using BERT tokenizer
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)

X_train_list = X_train['text_light_clean'].values
X_test_list = X_test['text_light_clean'].values

y_train = y_train.values
y_test = y_test.values

# Encode data
def encode(comment, labels):
    input_ids = []
    attention_masks = []

    for word in comment:
        encoded_dict = tokenizer.encode_plus(
            word,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset

X_train = encode(X_train_list, y_train)
X_test = encode(X_test_list, y_test)

# Create DataLoaders
BATCH_SIZE = 64
train_dataloader = DataLoader(X_train, sampler=RandomSampler(X_train), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(X_test, sampler=SequentialSampler(X_test), batch_size=BATCH_SIZE)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
params = list(model.named_parameters())

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 2)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to calculate the accuracy of our predictions vs labels
def f_score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model.to(device)


training_stats = []
all_gold_labels = []
all_predictions = []
total_t0 = time.time()
#best_val_loss = float('inf')
#best_train_loss = float('inf')

# Train for each epoch
for epoch_i in range(0, 2):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, 2))
    print('Training...')
    
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        loss = result.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)            
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.3f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    # Validation
    print("")
    print("Running Validation...")
    
    t0 = time.time()
    model.eval()
    all_predictions = []
    all_gold_labels = []
    all_texts = []

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    total_f1_score = 0

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        all_gold_labels.extend(label_ids)
        all_predictions.extend(np.argmax(logits, axis=1))
        all_texts.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in b_input_ids])

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        total_f1_score += f_score(logits, label_ids)

    all_predictions = np.array(all_predictions)
    all_gold_labels = np.array(all_gold_labels)

    np.savetxt('validation_predictions.txt', np.column_stack((all_gold_labels, all_predictions, all_texts)), fmt='%s', delimiter='\t', header='gold\tpred\ttext', comments='')

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("  Accuracy: {0:.3f}".format(avg_val_accuracy))

    avg_val_fscore = total_f1_score / len(test_dataloader)
    print("  F-Score: {0:.3f}".format(avg_val_fscore))

    avg_val_loss = total_eval_loss / len(test_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.3f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Valid. F-score.': avg_val_fscore,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    #if avg_val_loss < best_val_loss and avg_train_loss < best_train_loss:
        #best_val_loss = avg_val_loss
        #best_train_loss = avg_train_loss

print("")
print("Training complete!")

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

with open(f'{output_dir}/training_stats.json', 'w') as json_file:
    json.dump(training_stats, json_file)

print("Training stats saved to 'training_stats.json'")
print("Predictions saved to 'validation_predictions.txt'")
