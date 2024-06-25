
#!pip install -U pip setuptools wheel
#!pip install -U 'spacy[apple]'
#!python -m spacy download de_core_news_lg

import pandas as pd
import spacy
nlp = spacy.load('de_core_news_lg')
import emoji
import re

# Remove unwanted symbols
def remove_symbols(text):
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s<>]', '', text)
    return text

# Replace hashtags
def replace_hashtags(text):
    hashtag_pattern = r'#\s*([-\w]+)'
    #return re.sub(hashtag_pattern, r'<hashtag>', text)
    return re.sub(hashtag_pattern, r'<hashtag\1>', text)

# Remove hashtag signs
def remove_hashtags(text):
    return re.sub('#', '', text)

# Replace emojis to 'emoji'
def replace_emojis(text):
    return emoji.replace_emoji(text, ' emoji ')

# Remove emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, '')

# Replace links to 'link'
def replace_links(text):
    link_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.sub(link_pattern, 'link', text)

# Remove links
def remove_links(text):
    link_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.sub(link_pattern, '', text)

# Replace user mentions to 'user'
def replace_user_mentions(text):
    user_pattern = r'@\w+'
    return re.sub(user_pattern, 'user', text)

# Remove user mentions
def remove_user_mentions(text):
    user_pattern = r'@\w+'
    return re.sub(user_pattern, '', text)

# Lemmatize text
def lemmatize(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# Remove stopwords
def remove_stopwords(text):
    doc = nlp(text)
    text_without_stopwords = ' '.join([token.text for token in doc if not token.is_stop])
    return text_without_stopwords

# Remove short words (less than 2 symbols)
def remove_short(text):
    short_word = re.compile(r'\W*\b\w{1,2}\b')
    return short_word.sub('', text)

# ==================== Heavy cleaning for the feature extraction ====================
def clean_full(text):

    replaced_text = replace_emojis(text)
    replaced_text = replace_links(replaced_text)
    replaced_text = replace_user_mentions(replaced_text)
    replaced_text = remove_stopwords(replaced_text)
    replaced_text = lemmatize(replaced_text)
    replaced_text = remove_symbols(replaced_text)
    replaced_text = replaced_text.lower()
    replaced_text = re.sub(' {2,}', ' ', replaced_text)
    replaced_text = remove_short(replaced_text)
    
    return replaced_text

# ==================== Auxiliary cleaning for the feature extraction ====================
def clean(text):

    replaced_text = remove_hashtags(text)
    replaced_text = remove_emojis(replaced_text)
    replaced_text = remove_links(replaced_text)
    replaced_text = remove_user_mentions(replaced_text)

    return replaced_text

# Emoji counter
def count_emojis(text):
    return text.count('emoji')/len(text)*100


# Username counter
def count_usernames(text):
    return text.count('user')/len(text)*100

# Link counter
def count_links(text):
    return text.count('link')/len(text)*100

# Download ner tagger
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("flair/ner-german-large")

# Ner extractor
def ner(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    
    entity_dict = {'PER': 0, 'ORG': 0, 'LOC': 0, 'MISC': 0}

    for token in sentence.get_spans('ner'):
        if 'PER' in token.tag:
            entity_dict['PER'] += 1/len(sentence)*100
        elif 'ORG' in token.tag:
            entity_dict['ORG'] += 1/len(sentence)*100
        elif 'LOC' in token.tag:
            entity_dict['LOC'] += 1/len(sentence)*100
        elif 'MISC' in token.tag:
            entity_dict['MISC'] += 1/len(sentence)*100

    return entity_dict

# Read offensive words and lemmatise
with open('offensive_words.txt', 'r') as file:
    offensive_words = file.read().replace(',\n', ' ')
    offensive_words = lemmatize(offensive_words).lower()
    offensive_words_list = list(offensive_words.split(" "))
    offensive_words_list = list(dict.fromkeys(offensive_words_list))


# Count offensive words in the comment and normalise for the comment length
def offensive_words(text):
    offensive_words=0
    offensive_words = sum(1 for word in text.lower().split() if word in offensive_words_list)/len(text)*100
    return offensive_words

# Type-token ratio
def ttr(text):
    words = text.split()
    types = set(words)
    ttr = len(types) / len(words)
    return ttr

# Calculate avg length of words in a comment
def word_len_avg(text):
    text = clean(text)
    
    doc = nlp(text)
    res = [len(token.text) for token in doc if not (token.text == ' ' or token.is_punct)]
    if len(res) > 0:
        return sum(res) / len(res)
    else:
        return 0

# Lix text complexity
def lix(text):
    text = clean(text)

    doc = nlp(text)
    num_sent = len(list(doc.sents))
    res = [len(token.text) for token in doc if not (token.text == ' ' or token.is_punct)]
    long_word_num = len([word for word in res if word >= 6])
    if len(res) > 0:
        return len(res) + (long_word_num * 100 / len(res)) / (num_sent)
    else:
        return 0

# Function to extract POS bi- and trigrams from the comment and normalise for the comment length
from collections import Counter
from itertools import combinations
import pandas as pd

# POS-grams extraction for the corpus
def posgrams_corpus(df1):
    pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']

    combinations_bigram = list(combinations(pos, 2))
    combinations_trigram = list(combinations(pos, 3))

    def pos_ngram(text, ngram=2):
        doc = nlp(''.join(text))
        pos_tags = [token.pos_ for token in doc]
        ngrams_list = [tuple(pos_tags[i:i+ngram]) for i in range(len(pos_tags) - ngram + 1)]
        ngram_counter = Counter(ngrams_list)
        normalized_counter = {key: value / len(pos_tags) * 100 for key, value in ngram_counter.items()}
        return normalized_counter

    df1['pos_bigram'] = df1['text'].apply(lambda x: pos_ngram(x, ngram=2))
    df1['pos_trigram'] = df1['text'].apply(lambda x: pos_ngram(x, ngram=3))

    # Distribute the obtained values into corresponding columns
    for combination in combinations_bigram:
        column_name = f'{combination[0]}_{combination[1]}'
        df1[column_name] = df1['pos_bigram'].apply(lambda x: x[combination] if combination in x else 0)

    for combination in combinations_trigram:
        column_name = f'{combination[0]}_{combination[1]}_{combination[2]}'
        df1[column_name] = df1['pos_trigram'].apply(lambda x: x[combination] if combination in x else 0)

    df1 = df1.drop(['pos_bigram','pos_trigram'], axis=1)

    return df1


# Apply functions that require auxiliary cleaning
def feature_1(df1):
    df1['ner'] = df1['text'].apply(lambda x: ner(x))
    df2 = (df1['ner'].apply(pd.Series)).reset_index()
    df1 = pd.merge(df1, df2, on='index')
    df1 = df1.drop(columns='ner')
    df1['word_len_avg'] = df1['text'].apply(lambda x: word_len_avg(x))
    df1['ttr'] = df1['text'].apply(lambda x: ttr(x))
    df1['lix'] = df1['text'].apply(lambda x: lix(x))

    return df1

# Apply functions that require heavy cleaning
def feature_2(df1):
    df1['cleaned_text'] = df1['text'].apply(lambda x: clean_full(x))
    df1['offensive_w'] = df1['cleaned_text'].apply(lambda x: offensive_words(x))
    df1['emoji_count'] = df1['cleaned_text'].apply(lambda x: count_emojis(x))
    df1['username_count'] = df1['cleaned_text'].apply(lambda x: count_usernames(x))
    df1['link_count'] = df1['cleaned_text'].apply(lambda x: count_links(x))
    df1 = df1.drop(columns='cleaned_text')
    return df1

# ==================== Light cleaning for the BERT fine-tuning ====================

#import pandas as pd
#df = pd.read_csv('../training_set.tsv', sep='\t', encoding='utf-8', engine='python')
#df = pd.read_csv('../dataset_linguistic_classifier/test_data/test_set.tsv', sep='\t', encoding='utf-8', engine='python')

# Remvove unwanted symbols
# def remove_symbols_light(text):
#     text = re.sub(r'\s[^\w\s]\s', '', text)
#     text = re.sub(r'\w*\d+\w*', '', text)
#     text = re.sub(r'[\'"„“/()]', '', text)
#     return text

# Light-clean the text 
# def clean_full_light(text):
    
#     replaced_text = replace_emojis(text)
#     replaced_text = replace_links(replaced_text)
#     replaced_text = replace_user_mentions(replaced_text)
#     replaced_text = remove_symbols_light(replaced_text)
#     replaced_text = replaced_text.lower()
#     replaced_text = re.sub(' {2,}', ' ', replaced_text)
#     return replaced_text


#df['text_light_clean'] = df['text'].apply(lambda x: clean_full_light(x))
#df.to_csv('training_set.tsv', sep='\t', index=None)


