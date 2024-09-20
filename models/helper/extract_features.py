import pandas as pd
import spacy
nlp = spacy.load('de_core_news_lg')
from helper import posgrams_corpus, feature_1, feature_2
import argparse

# ==================== Cleaning for stylometric preprocessing training/test set ====================

# Preprocessed training set is saved under 'train.tsv' (features with 10% frequency are kept)

# Create the parser
parser = argparse.ArgumentParser(description="Feature Extraction")

# Add an argument
parser.add_argument('training_file', type=str, help='path to train set, expected columns: "text", "label"')
parser.add_argument('test_file', type=str, help='path to test set, expected columns: "text", "label"')

# Parse the arguments
args = parser.parse_args()

df = pd.read_csv(args.training_file, sep='\t', encoding='utf-8', usecols=[0, 1], names=['text', 'label'],  skiprows=1)
df['label'] = df['label'].astype(int)
df['index'] = df.index
df = posgrams_corpus(df)
df = feature_1(df)
df = feature_2(df)

columns_to_keep = [ 'emoji_count', 'username_count', 'link_count']

df = df.loc[:, (df != 0).any(axis=0) | df.columns.isin(columns_to_keep)]
threshold = 0.1 * len(df)
df = df.loc[:, ((df != 0).sum(axis=0) >= threshold) | df.columns.isin(columns_to_keep)]

df1 = pd.read_csv(args.test_file, sep='\t', encoding='utf-8', usecols=[0,1], names=['text','label'],  skiprows=1)
df1['label'] = df1['label'].astype(int)
df1['index'] = df1.index

df1 = posgrams_corpus(df1)
df1 = feature_1(df1)
df1 = feature_2(df1)

# for the test set, extract only those features which are present in the training set
df1 = df1[df1.columns.intersection(df.columns)]

# Save to file
df.to_csv('../../dataset/text-based/train.tsv', sep='\t', index=None)
df1.to_csv('../../dataset/text-based/test.tsv', sep='\t', index=None)
