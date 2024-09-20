import sys
import requests
import json
from datetime import datetime
import pandas as pd

def format_timestamp(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')

def fetch_user_metadata(username):
    # Reddit API endpoint for user information
    user_url = f"https://www.reddit.com/user/{username}/about.json"
    # Reddit API endpoint for user's recent posts
    posts_url = f"https://www.reddit.com/user/{username}/submitted.json"
    limit_nr = 100 # max
    posts_url += f"?limit={limit_nr}"
    # Reddit API endpoint for user's recent comments
    comments_url = f"https://www.reddit.com/user/{username}/comments.json"
    comments_url += f"?limit={limit_nr}"
    
    # Send GET request to fetch user information
    user_response = requests.get(user_url, headers={"User-Agent": "YOUR_APP_NAME"})

    # Check if the request was successful
    if user_response.status_code == 200:
        # Parse JSON response for user information
        user_data = user_response.json()['data']
        try:   
            # Extract relevant user metadata
            user_metadata = {
                'username': user_data['name'],
                'created_utc': format_timestamp(user_data['created_utc']),
                'comment_karma': user_data['comment_karma'],
                'link_karma': user_data['link_karma'],
                'is_gold': user_data['is_gold'],
                'is_mod': user_data['is_mod'],
                'has_verified_email': user_data['has_verified_email']
            }
            
            pages = 10
            recent_posts_metadata = []
            for page in range(pages):
                if page != 0:
                    after = posts_response.json()['data']['after']
                    if after == None:
                        break
                    posts_url += f"&after={after}&count={len(recent_posts_metadata)}"
                posts_response = requests.get(posts_url, headers={"User-Agent": "YOUR_APP_NAME"})
                # Parse JSON response for user's recent posts
                if posts_response.status_code == 200:
                    posts_data = posts_response.json()['data']['children']
                    # Extract metadata of recent posts
                    for post in posts_data:
                        post_metadata = {
                            'title': post['data']['title'],
                            'permalink': post['data']['permalink'],
                            'num_comments': post['data']['num_comments'],
                            'score': post['data']['score'],
                            'subreddit': post['data']['subreddit'],
                            'is_self': post['data']['is_self'],
                            'domain': post['data']['domain']
                        }
                        if post['data'].get('created_utc'):
                            post_metadata['created_utc'] = format_timestamp(post['data']['created_utc'])
                        recent_posts_metadata.append(post_metadata)
                    user_metadata['recent_posts'] = recent_posts_metadata
                else:
                    print(f"Failed to fetch recent posts. Status code: {posts_response.status_code}")
            
            for page in range(pages):
                if page != 0:
                    after = comments_response.json()['data']['after']
                    if after == None:
                        break
                    comments_url += f"&after={after}&count={len(recent_comments_metadata)}"
                # Send GET request to fetch user's recent comments
                comments_response = requests.get(comments_url, headers={"User-Agent": "YOUR_APP_NAME"})
                # Parse JSON response for user's recent comments
                if comments_response.status_code == 200:
                    comments_data = comments_response.json()['data']['children']
                    # Extract metadata of recent comments
                    recent_comments_metadata = []
                    for comment in comments_data:
                        comment_metadata = {
                            'body': comment['data']['body'],
                            'created_utc': format_timestamp(comment['data']['created_utc']),
                            'link_id': comment['data']['link_id'],
                            'score': comment['data']['score'],
                            'subreddit': comment['data']['subreddit'],
                            'permalink': comment['data']['permalink']
                        }
                        recent_comments_metadata.append(comment_metadata)
                    user_metadata['recent_comments'] = recent_comments_metadata
            else:
                print(f"Failed to fetch recent comments. Status code: {comments_response.status_code}")
            
            return user_metadata
        except KeyError as e:
            print(f"Failed to fetch user metadata. Key {e.args[0]} not existing")
            return None
    else:
        print(f"Failed to fetch user metadata. Status code: {user_response.status_code}")
        return None

def save_metadata_to_json(username, user_metadata):
    if user_metadata:
        filename = f"./{username}.json"
        with open(filename, 'w') as f:
            json.dump(user_metadata, f, indent=4)
        print(f"Metadata saved to {filename}")
    else:
        print("Failed to fetch metadata, cannot save to file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <username>")
        sys.exit(1)
    username = sys.argv[1]
    user_metadata = fetch_user_metadata(username)
    save_metadata_to_json(username, user_metadata)
