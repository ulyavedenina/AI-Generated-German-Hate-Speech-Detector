
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score
import time

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

df = pd.read_csv('dataset_metadata_classifier/bot_human_metadata.tsv', sep='\t', encoding='utf-8', engine='python')

y = df['human/bot']
X = df.drop(['human/bot'], axis=1)


# num humans
len([element for element in y if element == 0])
# num bots
len([element for element in y if element == 1])

from scipy.stats import zscore

column_types = X.dtypes

columns_to_convert = ['is_gold', 'is_mod']
X[columns_to_convert] = X[columns_to_convert].astype(int)
X['has_verified_email'] = X['has_verified_email'].map(lambda x: 1 if x else 0)

column_types = X.dtypes

X_scaled = X.drop(columns=['User_ID'], axis=1)
print(X_scaled.columns)

X_scaled = X_scaled.apply(zscore)

X_scaled = pd.DataFrame(X_scaled, columns=X.drop(columns=['User_ID']).columns)

#X_scaled['User_ID'] = X['User_ID']
#X_scaled['is_gold'] = X['is_gold']
#X_scaled['is_mod'] = X['is_mod']
#X_scaled['has_verified_email'] = X['has_verified_email']


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.head())

y_train = y_train.values
y_test= y_test.values

X_train = X_train.values
X_test = X_test.values


# Define NuSVC hyperparameters
params = {'criterion': 'entropy', 
     'max_features': 'log2', 
     'min_samples_split': 8, 
     'n_estimators': 10
     }

# Create Classifier
clf = RandomForestClassifier(**params)
start_time = time.time()

# Train the model
clf.fit(X_train, y_train)

# Evaluate on the test set
train_predictions = clf.predict(X_test)
end_time = time.time()

acc = accuracy_score(y_test, train_predictions)
fscore = f1_score(y_test, train_predictions)
train_predictions_prob = clf.predict_proba(X_test)
ll = log_loss(y_test, train_predictions_prob)

elapsed_time = end_time - start_time


print("=" * 30)
print("NuCVC")
print("Best Hyperparameters:", params)
print('****Results****')
print("Time", elapsed_time, "seconds")
print("Accuracy: {:.4%}".format(acc))
print("F1-score: {:.4%}".format(fscore))
print("Log Loss: {}".format(ll))
print("=" * 30)
