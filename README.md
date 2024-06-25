# BotCatcher

BotCatcher is a comprehensive collection of models and data designed for the text-based detection of German bot-generated hate speech. It also includes a metadata-based, language-agnostic dataset for bot detection on Reddit and corresponding models for identifying bots. The datasets for this project were predominantly developed internally. BotCatcher is part of the project [BoTox](https://botox.h-da.de).

## Repository Structure

1. dataset *without text fields*
   - *training set.tsv* -- bot- and human-generated German hate speech comments (train set)
   - *test set.tsv* -- bot- and human-generated German hate speech comments (test set)
   - *bot_human_metadata.tsv* -- bot- and human-metadata dataset
2. models
   - *helper* -- preprocessing functions
   - *text-based* -- models designed for text-based hate speech bot detection
   - *metadata-based* -- models designed for metadata-based bot detection 

## Text-Based Bot-Generared Hate Speech Detection (for German) 

Below you find the sources for the data collection. Note that models were tested on the outputs of an unseen LLM to ensure robustness.

#### Part 1: Human-Written Comments

| Sources                     | Total Comments |
|-----------------------------|----------------|
| DeTox                       | 4,504          | 
| RP-MOD                      | 2,813          |
| HASOC 2019                  | 543            |
| GermEval-2018 (test set)    | 1598           |

#### Part 2: Bot-Generated Comments

| Sources                                          | Total Comments |
|--------------------------------------------------|----------------|
| GPT 3.5                                          | 1600           |
| GPT 4                                            | 1601           |
| TheBloke/em_german_13b_v01-GPTQ                  | 1600           |
| TheBloke/em_german_leo_mistral-GPTQ              | 1600           |
| TheBloke/leo-hessianai-13B-chat-GPTQ             | 1600           |
| mistralai/Mixtral-8x7B-Instruct-v0.1 (test set)  | 1599           |

### Models

We have implemented the following models for the text-based approach:

| Model                 | F-Score |
|-----------------------|-------- |
| BERT Base             | 0.974   | 
| BERT Large            | 0.986   |
| BERT Base-CNN         | 0.980   |
| BERT Base+Stylometric | 0.952   |
| Stylometric           | 0.881   |
| LLM (Llama2) 7B       | 0.943   |
| LLM (Llama2) 13B      | 0.947   |

## Metadata-Based Reddit Bot Detection

### Dataset

The Reddit metadata dataset comprises **818** Human and **816** Bot Accounts with corresponsing features.

### Feature Set for Metadata Classification

| Feature name            | Description                                            |
|-------------------------|--------------------------------------------------------|
| comment_karma           | Comment karma of a user                                |
| post_karma              | Post karma of a user                                   |
| is_gold                 | Premium status of a user                               |
| is_mod                  | User is a subreddit moderator                          |
| has_verified_email      | User’s email is verified                               |
| comment_activity_day    | Number of comments per day                             |
| posts_activity_day      | Number of posts per day                                |
| time_to_first_post      | Time from account creation to the first post (in sec)  | 
| AskReddit_activity      | Proportion of posts and comments in AskReddit          | 
| subreddit_unique        | Proportion of unique subreddits in the user’s history  |
| avg_frequency_posts     | Average time between the posts (in seconds)            | 
| avg_frequency_all       | Average time between any activity (in seconds)         |
| min_time_post           | Minimal time between the posts (in seconds)            |
| min_time_all            | Minimal time between any activity (in seconds)         | 
| num_url                 | Proportion of links in posts and comments              | 
| num_repeated_post       | Proportion of repeated posts                           | 
| num_repeated_comment    | Proportion of repeated comments                        |

### Models

We have implemented the following models for the metadata-based approach:

| Model             | F-Score |
|-------------------|-------- |
| Random Forest     | 0.934   | 
| Gradient Boosting | 0.880   |
| Ada Boost         | 0.857   |
| K-Neigbours       | 0.825   |
| Decision Tree     | 0.793   |
| NuCVC             | 0.712   |
| LDA               | 0.583   |
| CVC               | 0.560   |

## Request the complete dataset

Due to ethical considerations, the hate speech comments dataset has empty text fields, and the Reddit metadata dataset is lacking usernames. If you need the missing data for your research purposes, please write to melanie.siegel@g-da.de, shortly describing the future use of any of the datasets.

## Citation

TBD
