import numpy as np
import pandas as pd
import datetime
import time
import csv
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

SEED_VAL = 42
torch.manual_seed(SEED_VAL)
np.random.seed = SEED_VAL
random.seed(SEED_VAL)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

torch.backends.cudnn.deterministic = True
torch.cuda.amp.autocast(enabled=True)

df = pd.read_csv('../../../dataset/training_set.tsv', sep='\t', encoding='utf-8', engine='python')
df = df.drop(['index', 'text'], axis=1)

y = df['author']
X = df.drop(['author'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_list = X_train['text_light_clean']
X_val_list = X_val['text_light_clean']

y_train = y_train.values
y_val = y_val.values

df1 = pd.read_csv("../../../dataset/test_set.tsv", delimiter='\t')
X_test_list = df1.text_light_clean.values
y_test = df1.label.values

# Encode data
def encode(comment, labels):
    input_ids = []
    attention_masks = []

    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')

    for word in comment:
        encoded_dict = tokenizer.encode_plus(
            word,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset

X_train = encode(X_train_list, y_train)
X_val = encode(X_val_list, y_val)
X_test = encode(X_test_list, y_test)

BATCH_SIZE = 32
train_dataloader = DataLoader(X_train, sampler=RandomSampler(X_train), batch_size=BATCH_SIZE)
val_dataloader = DataLoader(X_val, sampler=SequentialSampler(X_val), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(X_test, sampler=SequentialSampler(X_test), batch_size=BATCH_SIZE)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


import torch.nn as nn

class Bert_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        output_channel = config.output_channel  # number of kernels
        num_classes = config.num_classes  # number of targets to predict
        dropout = config.dropout  # dropout value
        embedding_dim = config.embedding_dim  # length of embedding dim
        
        #input_channel = 1  # for single embedding, input_channel = 1
        #self.conv1 = nn.Conv2d(input_channel, output_channel, (3, embedding_dim), padding=(2, 0), groups=1)
        self.conv1 = nn.Conv1d(embedding_dim, output_channel, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        # Expected input dimensions: [batch_size, embedding_dim, sequence_length]

        # Permute the dimensions to [batch_size, sequence_length, embedding_dim]
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        
        # Apply 1D convolution
        x = nn.functional.relu(self.conv1(x))

        # Perform max pooling over the sequence length
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)

        # Apply dropout
        x = self.dropout(x)

        # Apply fully connected layer
        x = self.fc1(x)

        return x

def train(model, dataloader, optimizer):


    total_t = time.time()

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')

    train_total_loss = 0
    train_total_f1 = 0

    model.train()
    bert_cnn.train()

    
    for step, batch in enumerate(dataloader):

        if step % 40 == 0 and not step == 0:

            print('   Batch  {:>5,}    of    {:>5,}.'.format(step, len(dataloader)))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).long()

        optimizer.zero_grad()

        with autocast():

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=True)

            hidden_layers = outputs[2]
            hidden_layers = torch.stack(hidden_layers, dim=1)
            hidden_layers = hidden_layers[:, -1:]
            
        #print("Shape of hidden_layers before passing to Bert_CNN:", hidden_layers.shape)
        logits = bert_cnn(hidden_layers)
        loss = criterion(logits.view(-1, 2), b_labels)

        train_total_loss += loss.item()

        loss.backward()
        optimizer.step()

        scheduler.step()

        _, predicted = torch.max(logits, 1)

        predicted = predicted.detach().cpu().numpy()
        y_true = b_labels.detach().cpu().numpy()

        train_total_f1 += f1_score(predicted, y_true, average = 'weighted', labels = np.unique(predicted))

        avg_train_loss = train_total_loss / len(dataloader)
        avg_train_f1 = train_total_f1 / len(dataloader)

        train_time = format_time(time.time()- total_t)

        training_stats.append(
        {
            'Training Loss': avg_train_loss,
            'Training F1-score': avg_train_f1,
            'Training Time': train_time,
        }
    )

    print()
    print("epoch | trn loss | trn f1 | trn time ")
    print(f"{epoch+1:5d} | {avg_train_loss:.5f} | {avg_train_f1:.5f} | {train_time:}")

    return None

def validation(model, dataloader):

    total_t = time.time()

    print("")
    print('Validation...')

    model.eval()
    bert_cnn.eval()

    total_val_acc = 0
    total_val_loss = 0
    total_val_f1 = 0

    for batch in dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).long()

        with torch.no_grad():

            outputs = model(input_ids = b_input_ids, attention_mask = b_input_mask)

            hidden_layers = outputs[2]

            hidden_layers = torch.stack(hidden_layers, dim=1)
            
            hidden_layers = hidden_layers[:, -1:]
            logits = bert_cnn(hidden_layers)
            loss = criterion(logits.view(-1,2), b_labels.view(-1))

            total_val_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            
            predicted = predicted.detach().cpu().numpy()
            y_true = b_labels.detach().cpu().numpy()

            total_val_f1 += f1_score(y_pred=predicted, y_true=y_true, average='weighted', labels=np.unique(predicted))

            total_val_acc += accuracy_score(y_pred=predicted, y_true=y_true)
                
        avg_accuracy = total_val_acc / len(dataloader)

        global avg_val_f1
        avg_val_f1 = total_val_f1 / len(dataloader)

        val_time = format_time(time.time() - total_t)

        global avg_val_loss
        avg_val_loss = total_val_loss / len(dataloader)

        val_stats.append(
        {
            'Validation Loss': avg_val_loss,
            'Validation F1-score': avg_val_f1,
            'Validation Accuracy': avg_accuracy,
            'Validation Time': val_time,
        }
        )

    print()
    print("epoch | val loss | val f1 | trn time ")
    print(f"{epoch+1:5d} | {avg_val_loss:.5f} | {avg_val_f1:.5f} | {val_time:}")

    return None

def test(model, dataloader, output_file):

    print('')
    print(' Running test')

    all_predictions = []
    all_true_labels = []
    total_t = time.time()

    model.eval()
    bert_cnn.eval()

    total_test_acc = 0
    total_test_loss = 0
    total_test_f1 = 0

    for batch in dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).long()

        with torch.no_grad():

            outputs = model(input_ids = b_input_ids, attention_mask = b_input_mask)

            hidden_layers = outputs[2]

            hidden_layers = torch.stack(hidden_layers, dim=1)
            
            hidden_layers = hidden_layers[:, -1:]
            logits = bert_cnn(hidden_layers)
            loss = criterion(logits.view(-1,2), b_labels.view(-1))

            total_test_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            
            predicted = predicted.detach().cpu().numpy()
            y_true = b_labels.detach().cpu().numpy()

            total_test_f1 += f1_score(y_pred=predicted, y_true=y_true, average='weighted', labels=np.unique(predicted))

            total_test_acc += accuracy_score(y_pred=predicted, y_true=y_true)

            all_predictions.extend(predicted)
            all_true_labels.extend(y_true)


        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['true label', 'prediction'])
            writer.writerows(zip(all_true_labels, all_predictions))
            
    avg_accuracy = total_test_acc / len(dataloader)

    avg_test_f1 = total_test_f1 / len(dataloader)

    avg_test_loss = total_test_loss / len(dataloader)

    training_time = format_time(time.time() - total_t)

    test_stats.append(
        {
            'Test Loss': avg_test_loss,
            'Test Accur.': avg_accuracy,
            'Test F1': avg_test_f1,
            'Test Time': training_time
        }
    )
    # print result summaries
    print("")
    print("summary results")
    print("epoch | test loss | test f1 | test time")
    print(f"{epoch+1:5d} | {avg_test_loss:.5f} | {avg_test_f1:.5f} | {training_time:}")

    return None

model = BertModel.from_pretrained(
    "dbmdz/bert-base-german-uncased",
    output_hidden_states=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model.to(device)

class config:

    def __init__(self):
        config.num_classes = 2
        config.output_channel = 16
        config.embedding_dim = 768
        config.dropout = 0.2
        return None
    
config1 = config()

bert_cnn = Bert_CNN(config1).to(device)

criterion = nn.CrossEntropyLoss()

epochs = 5

Bert_params = model.parameters()
optimizer = optim.SGD([{'params': Bert_params, 'lr': 0.002}], weight_decay=0.001)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

scaler = GradScaler()

training_stats = []
val_stats = []
best_val_loss = float('inf')

for epoch in range(epochs):

    train(model, train_dataloader, optimizer)

    validation(model, val_dataloader)

    torch.save(model.state_dict(), 'bert_cnn.pt')
    model_pt = model.module if hasattr(model, 'module') else model
    model_pt.save_pretrained('./model/')

test_stats = []
model.load_state_dict(torch.load('bert_cnn.pt'))

if all(torch.all(torch.eq(p1, p2)) for p1, p2 in zip(model.parameters(), torch.load('bert_cnn.pt').values())):
    print("Model parameters loaded successfully.")
else:
    print("Error: Model parameters not loaded correctly.")

test(model, dataloader=test_dataloader, output_file='test_predictions.tsv')
