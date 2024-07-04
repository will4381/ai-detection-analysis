# From Google Colab notebook, with help from Claude Sonnet 3.5.
# pip install openai requests datasets tqdm pandas matplotlib numpy textblob

import json
import os
import re
import time
from collections import Counter, defaultdict
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from tqdm import tqdm
from google.colab import drive
from google.colab import userdata
from openai import OpenAI
import requests
from datasets import load_dataset

# Mount Google Drive
drive.mount('/content/drive')

# Define API keys (replace with actual keys)
undetectable_api_key = userdata.get('undetectable_secret')
gptzero_api_key = userdata.get('gptzero_api_key')
openai_api_key = userdata.get('openai_secret')

# Authenticate OpenAI
client = OpenAI(api_key=openai_api_key)

def remove_markdown(text):
    if not text:
        return text

    # Remove headers
    text = re.sub(r'^\s*#.*$', '', text, flags=re.MULTILINE)

    # Remove bold and italic
    text = re.sub(r'\*\*|__|\*|_', '', text)

    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove blockquotes
    text = re.sub(r'^\s*>.*$', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# Only use if generations .json file does not exist or you wish to create a new one

# Load the dataset
ds = load_dataset("yahma/alpaca-cleaned")
instructions = ds['train']['instruction']

max_tokens = 256
total_tokens_used = 0
max_total_tokens = 1000000

def generate_ai_content(prompt):
    global total_tokens_used
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who does not use markdown in their responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        content = completion.choices[0].message.content.strip()
        tokens_used = len(content.split())
        total_tokens_used += tokens_used
        return content, tokens_used
    except Exception as e:
        print(f"Error generating content for prompt '{prompt}': {e}")
        return None, 0

print("Generating AI content:")
original_texts = []
for prompt in tqdm(instructions[:10]):
    if total_tokens_used >= max_total_tokens:
        print(f"Reached token limit. Generated {len(original_texts)} examples.")
        break

    content, tokens = generate_ai_content(prompt)
    if content:
        content = remove_markdown(content)  # Remove markdown
        original_texts.append(content)
        print(f"Prompt: {prompt}")
        print(f"Generated content (markdown removed): {content[:100]}...")  # Print first 100 characters
        print(f"Tokens used: {tokens}")
        print(f"Total tokens used so far: {total_tokens_used}")
        print()
    else:
        print(f"Skipped prompt due to error: {prompt}")
        print()

print(f"Total examples generated: {len(original_texts)}")
print(f"Estimated total tokens used: {total_tokens_used}")

# Save generations to a .json file to load later.

# Define the save path
save_path = "/content/drive/My Drive/ai_generations_test1.json"

def save_progress(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

try:
    # Check if original_texts is defined and has content
    if 'original_texts' in globals() and original_texts:
        save_progress(original_texts, save_path)
        estimated_words = total_tokens_used * 0.75  # Estimated word count assuming 1 token equals ~3/4 words
        print(f"Total examples generated: {len(original_texts)}")
        print(f"Estimated total tokens used: {total_tokens_used}")
        print(f"Estimated words: {estimated_words}")
        print(f"Progress saved to {save_path}")
    else:
        print("No generated content found to save.")
except Exception as e:
    print(f"Error while saving progress: {e}")

# Only use to load already created generations .json file

# Define the save path
save_path = "/content/drive/My Drive/ai_generations_original.json"

# Load the dataset for prompts
ds = load_dataset("yahma/alpaca-cleaned")
instructions = ds['train']['instruction']

# Load existing generated data
try:
    with open(save_path, 'r') as f:
        original_texts = json.load(f)
    print(f"Loaded {len(original_texts)} examples from {save_path}")

    # If the loaded data is not a list, try to extract it from a dictionary
    if isinstance(original_texts, dict):
        if 'original_texts' in original_texts:
            original_texts = original_texts['original_texts']
        else:
            print("Warning: Unexpected data format. Using all loaded data as original texts.")

    # Ensure original_texts is a list
    if not isinstance(original_texts, list):
        raise TypeError("Loaded data is not in the expected format (list or dictionary with 'original_texts' key)")

except FileNotFoundError:
    print(f"No existing data found at {save_path}. Please generate content first.")
    original_texts = []
except json.JSONDecodeError:
    print(f"Error decoding JSON from {save_path}. Please check the file integrity.")
    original_texts = []
except TypeError as e:
    print(f"Error in data format: {e}")
    original_texts = []

print(f"Total examples: {len(original_texts)}")

# Print the first 10 examples or all if less than 10
print("\nFirst 10 examples (or all if less than 10):")
if len(original_texts) == 0:
    print("No examples to display. Please generate content first.")
else:
    for i, text in enumerate(original_texts[:10]):
        print(f"\nExample {i + 1}:")
        if i < len(instructions):
            print(f"Prompt: {instructions[i]}")
        else:
            print("Prompt: Not available")
        print(f"Generated text: {text[:100]}...")  # Print first 100 characters of generated text

# If no data was loaded, provide instructions on how to generate content
if len(original_texts) == 0:
    print("\nTo generate content, please run the content generation code first.")

def retrieve_from_undetectable(document_id, max_attempts=10, delay=5):
    for attempt in range(max_attempts):
        url = "https://api.undetectable.ai/document"
        payload = json.dumps({"id": document_id})
        headers = {
            'api-key': undetectable_api_key,
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        result = response.json()

        if result.get('status') == 'done':
            return result

        print(f"Attempt {attempt + 1}: Text still processing. Waiting {delay} seconds...")
        time.sleep(delay)

    return None

def submit_to_undetectable(content):
    url = "https://api.undetectable.ai/submit"
    payload = json.dumps({
        "content": content,
        "readability": "University",
        "purpose": "General Writing",
        "strength": "More Human"
    })
    headers = {
        'api-key': undetectable_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

print("\nProcessing texts with Undetectable.ai:")
undetectable_texts = []
for idx, text in enumerate(tqdm(original_texts)):
    print(f"\nProcessing text {idx + 1}:")
    if text:
        response_data = submit_to_undetectable(text)
        print(f"Submission response: {response_data}")
        if 'id' in response_data:
            document_id = response_data['id']
            result = retrieve_from_undetectable(document_id)
            if result:
                undetectable_texts.append(result.get('output', None))
                print(f"Original (no markdown): {text[:50]}...")
                print(f"Undetectable: {result.get('output', 'N/A')[:50]}...")
            else:
                print("Failed to retrieve processed text after multiple attempts")
                undetectable_texts.append(None)
        else:
            print(f"Failed to get document ID. Response: {response_data}")
            undetectable_texts.append(None)
    else:
        print("Skipping due to missing original text")
        undetectable_texts.append(None)
    time.sleep(8.0)  # 3 requests per second and to perform successful retrieval on first try

def save_undetectable_responses(responses, save_path):
    with open(save_path, 'w') as f:
        json.dump(responses, f)
    print(f"Undetectable responses saved to {save_path}")

# Define the save path
save_path = "/content/drive/My Drive/undetectable_responses_original.json"

# Save the undetectable responses
save_undetectable_responses(undetectable_texts, save_path)

print(f"Saved {len(undetectable_texts)} undetectable responses.")

def get_ai_detection_scores(text):
    try:
        url = "https://api.gptzero.me/v2/predict/text"
        payload = {
            "document": text,
            "version": "2024-04-04"
        }
        headers = {
            "x-api-key": gptzero_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        if data['documents']:
            return data['documents'][0]['class_probabilities']
        else:
            return {'ai': 0, 'human': 0, 'mixed': 0}
    except Exception as e:
        print(f"Error getting detection scores: {e}")
        return {'ai': 0, 'human': 0, 'mixed': 0}

# Get AI detection scores
print("\nGetting AI detection scores:")
original_scores = []
undetectable_scores = []
for idx, (original, undetectable) in enumerate(tqdm(zip(original_texts, undetectable_texts))):
    print(f"\nProcessing pair {idx + 1}:")
    if original and undetectable:
        print(f"Original text (first 50 chars): {original[:50]}...")
        print(f"Undetectable text (first 50 chars): {undetectable[:50]}...")

        orig_score = get_ai_detection_scores(original)
        undet_score = get_ai_detection_scores(undetectable)

        original_scores.append(orig_score)
        undetectable_scores.append(undet_score)

        print(f"Original scores: {orig_score}")
        print(f"Undetectable scores: {undet_score}")
    else:
        print("Skipping this pair due to missing text.")

    print("-" * 50)

print(f"\nTotal pairs processed: {len(original_scores)}")
print(f"Total original scores: {len(original_scores)}")
print(f"Total undetectable scores: {len(undetectable_scores)}")

# Save results to CSV
results = []
for idx, (original_text, undetectable_text, original_score, undetectable_score) in enumerate(zip(original_texts, undetectable_texts, original_scores, undetectable_scores)):
    results.append({
        "prompt": instructions[idx],
        "original_text": original_text,
        "undetectable_text": undetectable_text,
        "original_scores": original_score,
        "undetectable_scores": undetectable_score
    })

# Save to CSV
df = pd.DataFrame(results)
csv_path = '/content/drive/My Drive/ai_text_analysis_results_original.csv'
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
