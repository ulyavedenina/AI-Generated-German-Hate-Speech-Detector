import pandas as pd
import spacy
nlp = spacy.load('de_core_news_lg')
from helper import posgrams_corpus, feature_1, feature_2

# ==================== Cleaning for preprocessing training/test sets ====================

# Preprocessed training set is saved under 'cleaned_features_selected' (features with 10% frequency are kept)
#df = pd.read_csv('training_set.tsv', sep='\t', encoding='utf-8', usecols=[0, 1, 2], skiprows=1, names=['index', 'text', 'label'])
#df = posgrams_corpus(df)
#df = feature_1(df)
#df = feature_2(df)
#df = df.loc[:, (df != 0).any(axis=0)]
#threshold = 0.1 * len(df)
#df = df.loc[:, (df != 0).sum(axis=0) >= threshold]


df1 = pd.read_csv('test_set.tsv', sep='\t', encoding='utf-8', usecols=[0,1], names=['text','author'],  skiprows=1)
df1['author'] = df1['author'].astype(int)
df1['index'] = df1.index
df1 = df1[ ['index'] + [ col for col in df1.columns if col != 'index' ]]

df1 = posgrams_corpus(df1)
df1 = feature_1(df1)
df1 = feature_2(df1)

# for the test set, extract only those features which are present in the training set
df_training = pd.read_csv('training_set.tsv', sep='\t', encoding='utf-8')
df1 = df1[df1.columns.intersection(df_training.columns)]

# Save to file
df1.to_csv('test_set.tsv', sep='\t', index=None)
