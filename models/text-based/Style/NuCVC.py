import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.svm import NuSVC

np.random.seed(42)
min_max_scaler = MinMaxScaler()

# Load and preprocess data
df = pd.read_csv('./dataset/text-based/train.tsv', sep='\t', encoding='utf-8', engine='python')
df = df.drop(['index', 'text', 'text_light_clean'], axis=1)
df = df.reindex(sorted(df.columns), axis=1)
df_test = pd.read_csv('./dataset/text-based/test.tsv', sep='\t', encoding='utf-8', engine='python')
df_test = df_test.drop(['index', 'text_light_clean', 'text'], axis=1)
df_test = df_test.reindex(sorted(df_test.columns), axis=1)

y_train = df['label']
X_train = df.drop(['label'], axis=1)

y_test = df_test['label']
X_test = df_test.drop(['label'], axis=1)

# Scale features in the range of 0 to 1
X_train_scaled = min_max_scaler.fit_transform(X_train)
y_train = y_train.values

X_test_scaled = min_max_scaler.transform(X_test)
y_test = y_test.values

# Define NuSVC hyperparameters
params = {
    'gamma': 'auto', 
    'kernel': 'linear'
}

# Create Classifier
clf = NuSVC(**params, probability=True)

# Train the model
clf.fit(X_train_scaled, y_train)

# Evaluate on the test set
train_predictions = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, train_predictions)
fscore = f1_score(y_test, train_predictions)
train_predictions_prob = clf.predict_proba(X_test_scaled)
ll = log_loss(y_test, train_predictions_prob)

print("=" * 30)
print("NuCVC")
print("Best Hyperparameters:", params)
print('****Results****')
print("Accuracy: {:.4%}".format(acc))
print("F1-score: {:.4%}".format(fscore))
print("Log Loss: {}".format(ll))
print("=" * 30)
