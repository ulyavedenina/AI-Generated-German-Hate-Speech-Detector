# ==================== Light cleaning for the BERT fine-tuning ====================

import pandas as pd
import argparse
from helper import clean_full_light

# Create the parser
parser = argparse.ArgumentParser(description="Light Cleaning for Bert models")

# Add an argument
parser.add_argument('data_file', type=str, help='path to dataset with "text" column')

# Parse the arguments
args = parser.parse_args()

df = pd.read_csv(args.data_file, sep='\t', encoding='utf-8')
df['text_light_clean'] = df['text'].apply(lambda x: clean_full_light(x))
df.to_csv('dataset.tsv', sep='\t', index=None)
