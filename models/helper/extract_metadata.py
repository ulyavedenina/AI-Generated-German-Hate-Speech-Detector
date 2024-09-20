import pandas as pd
import json
import os
from datetime import datetime

json_folder = 'dataset_metadata_classifier/combined_json/'
#json_folder = 'dataset_metadata_classifier/hwsbot_json/'

data = []

# Number of comments per day 
def comments_day(user_info):
    if not user_info['recent_comments']:
        return 0
    account_created = datetime.strptime(user_info['created_utc'], "%Y-%m-%d %H:%M:%S UTC")
    last_comment = datetime.strptime(user_info['recent_comments'][0]['created_utc'], "%Y-%m-%d %H:%M:%S UTC")
    total_time = (last_comment-account_created).days +1

    return len(user_info['recent_comments']) / total_time

# Number of posts per day 
def posts_day(user_info):
    if not user_info['recent_posts']:
        return 0
    account_created = datetime.strptime(user_info['created_utc'], "%Y-%m-%d %H:%M:%S UTC")
    last_post = datetime.strptime(user_info['recent_posts'][0]['created_utc'], "%Y-%m-%d %H:%M:%S UTC")
    total_time = (last_post-account_created).days +1

    return len(user_info['recent_posts']) / total_time

# average difference between the posts (in seconds)
def avg_frequency_posts(user_info):

    if not user_info['recent_posts'] or len(user_info['recent_posts']) == 1:
            return 0
    
    time = []
    for post in user_info['recent_posts']:
        time.append(datetime.strptime(post['created_utc'], "%Y-%m-%d %H:%M:%S UTC"))
    time = sorted(time)
    time_dif = []
    for i in range(1, len(time)):
        time_dif.append((time[i] - time[i-1]).total_seconds())
    
    return sum(time_dif) / len(time_dif)

# average difference between any activity (in seconds)
def avg_frequency_all(user_info):

    posts = user_info['recent_posts']
    comments = user_info['recent_comments']

    total_activity = len(posts) + len(comments)
    if total_activity == 0 or total_activity == 1:
        return 0
    
    all = posts + comments
    time = []
    for instance in all:
        time.append(datetime.strptime(instance['created_utc'], "%Y-%m-%d %H:%M:%S UTC"))
    time = sorted(time)
    time_dif = [(time[i] - time[i-1]).total_seconds() for i in range(1, len(time))]
    
    return sum(time_dif) / len(time_dif)

# minimal time between any activity (in seconds)
def min_time_all(user_info):

    posts = user_info['recent_posts']
    comments = user_info['recent_comments']

    total_activity = len(posts) + len(comments)
    if total_activity == 0 or total_activity == 1:
        return 0
    
    all = posts + comments
    #print(all, '\n')
    time = []
    for instance in all:
        time.append(datetime.strptime(instance['created_utc'], "%Y-%m-%d %H:%M:%S UTC"))
    time = sorted(time)
    time_dif = [(time[i] - time[i-1]).total_seconds() for i in range(1, len(time))]

    return  min(time_dif)

# proportion of links in posts + comments
def num_url(user_info):
    total_activity = len(user_info['recent_posts']) + len(user_info['recent_comments'])

    if total_activity == 0:
        return 0
    
    urls_post = sum(1 for post in user_info['recent_posts'] if 'http://' in post['title'] or 'https://' in post['title'])
    
    urls_comment = 0
    if len(user_info['recent_comments']) != 0:
        urls_comment = sum(1 for comment in user_info['recent_comments'] if ('body' in comment and ('http://' in comment['body'] or 'https://' in comment['body'])))
    return (urls_comment + urls_post) / total_activity

#proportion of repeated posts
def num_repeated_post(user_info):
    recent_posts = user_info['recent_posts']
    
    if not recent_posts:
        return 0
    
    post_content = set()
    repeated_posts = 0 
    
    for post in recent_posts:
        p = post['title']
        if p in post_content and p is not None:
            repeated_posts += 1 
        else:
            post_content.add(p)

    return repeated_posts / len(recent_posts)

# proportion of repeated comments
def num_repeated_comment(user_info):
    recent_comments = user_info['recent_comments']
    
    if not recent_comments:
        return 0
    
    comment_content = set()
    repeated_comments = 0 
    
    for comment in recent_comments:
        if 'body' in comment:
            c = comment['body']
            if c in comment_content and c is not None:
                repeated_comments += 1 
            else:
                comment_content.add(c)
    return float(repeated_comments / len(recent_comments))


for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        
        user_id = filename.split('.')[0]

        with open(os.path.join(json_folder, filename), 'r') as f:
            user_info = json.load(f)
        
        info = {
            'User_ID':user_info['username'],
            'comment_karma': user_info['comment_karma'],
            'post karma': user_info['link_karma'],
            'comment_activity_day': comments_day(user_info),
            'posts_activity_day': posts_day(user_info),
            'avg_frequency_posts': avg_frequency_posts(user_info),
            'avg_frequency_all': avg_frequency_all(user_info),
            'min_time_all': min_time_all(user_info),
            'num_url': num_url(user_info),
            'num_repeated_post': num_repeated_post(user_info),
            'num_repeated_comment': num_repeated_comment(user_info)
            }
        
    data.append(info)

      

new_df = pd.DataFrame(data)
new_df.to_csv('dataset_metadata_classifier/bot_metadata.tsv', index=False)
