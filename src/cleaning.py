import pandas as pd
import re
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the emotion classification pipeline with GPU support
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0)  # device=0 for GPU

# Function to clean text data
def clean_text(text):
    """Clean the input text by removing URLs, mentions, hashtags, and non-alphabetical characters."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#|\d+', '', text)  # Remove mentions, hashtags, and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

# Function to split the text into chunks that fit the model's input length
def split_text_into_chunks(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_length:
            chunks.append(' '.join(current_chunk[:-1]))  # Add the chunk (excluding the last word)
            current_chunk = [current_chunk[-1]]  # Start a new chunk with the last word

    if current_chunk:
        chunks.append(' '.join(current_chunk))  # Add the remaining chunk

    return chunks

# Function to assign emotion based on text
def assign_emotion(post):
    # Split the post into chunks
    chunks = split_text_into_chunks(post)
    
    # Batch process the chunks using the pipeline
    emotion_predictions = emotion_classifier(chunks)  # Predict for all chunks in parallel
    
    # Combine predictions to find the most frequent emotion (voting)
    emotion_labels = [pred['label'] for pred in emotion_predictions]
    most_frequent_emotion = Counter(emotion_labels).most_common(1)[0][0]

    # Calculate average score for each emotion (if needed for further analysis)
    emotion_scores = [pred['score'] for pred in emotion_predictions]
    emotion_score_map = {}
    for label, score in zip(emotion_labels, emotion_scores):
        if label in emotion_score_map:
            emotion_score_map[label].append(score)
        else:
            emotion_score_map[label] = [score]

    average_scores = {label: np.mean(scores) for label, scores in emotion_score_map.items()}
    final_predicted_emotion = max(average_scores, key=average_scores.get)

    return final_predicted_emotion

# Function to assign emoji based on the emotion
def assign_emoji(emotion):
    emoji_map = {
        'anger': '😡',
        'fear': '😨',
        'joy': '😊',
        'love': '❤️',
        'sadness': '😢',
        'surprise': '😲',
        'neutral': '😐'
    }
    return emoji_map.get(emotion.lower(), '🤷‍♂️')  # Default emoji if no match

# Load the dataset
print("Loading dataset...") 
df = pd.read_csv('dataset(cleaned too)\\mbti_1.csv')  # Replace with your dataset file
print("Dataset loaded successfully.\n")

# Clean the 'posts' column
print("Cleaning the text data in 'posts' column. This may take a few moments...\n")
df['cleaned_posts'] = df['posts'].apply(clean_text)
print("Text cleaning complete.\n")

# Use tqdm to display the progress bar for assigning emotions and emojis
print("Assigning emotions and emojis to the dataset...\n")
tqdm.pandas()  # Enable tqdm with pandas

# Assign emotions and emojis with progress bar
df['emotion'] = df['cleaned_posts'].progress_apply(assign_emotion)
df['emoji'] = df['emotion'].progress_apply(assign_emoji)

# Check the distribution of emotions and types before balancing
print("Emotion distribution before balancing:")
print(df['emotion'].value_counts())
print("\nType distribution before balancing:")
print(df['type'].value_counts())

# Visualize the emotion and type distribution before balancing
plt.figure(figsize=(12, 6))

# Plot emotion distribution
plt.subplot(1, 2, 1)
df['emotion'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Emotion Distribution Before Balancing')
plt.xlabel('Emotion')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)

# Plot type distribution
plt.subplot(1, 2, 2)
df['type'].value_counts().plot(kind='bar', color='salmon')
plt.title('Type Distribution Before Balancing')
plt.xlabel('MBTI Type')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Balance the dataset by oversampling underrepresented emotions and types
print("Balancing the dataset by oversampling underrepresented emotions and types...\n")

# Find the maximum count for emotions and types to balance the dataset
max_count_emotion = df['emotion'].value_counts().max()
max_count_type = df['type'].value_counts().max()

# Apply oversampling to the minority classes (emotions and types)
balanced_df = pd.concat([ 
    resample(df[df['emotion'] == emotion], 
             replace=True,  # Sample with replacement
             n_samples=max_count_emotion,  # Match the max emotion count
             random_state=42)  # Ensure reproducibility
    for emotion in df['emotion'].unique()
])

balanced_df = pd.concat([ 
    resample(balanced_df[balanced_df['type'] == type_], 
             replace=True,  # Sample with replacement
             n_samples=max_count_type,  # Match the max type count
             random_state=42)  # Ensure reproducibility
    for type_ in balanced_df['type'].unique()
])

# Shuffle the balanced DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset balanced successfully.\n")

# Check the distribution of emotions and types after balancing
print("Emotion distribution after balancing:")
print(balanced_df['emotion'].value_counts())
print("\nType distribution after balancing:")
print(balanced_df['type'].value_counts())

# Visualize the emotion and type distribution after balancing
plt.figure(figsize=(12, 6))

# Plot emotion distribution
plt.subplot(1, 2, 1)
balanced_df['emotion'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Emotion Distribution After Balancing')
plt.xlabel('Emotion')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)

# Plot type distribution
plt.subplot(1, 2, 2)
balanced_df['type'].value_counts().plot(kind='bar', color='salmon')
plt.title('Type Distribution After Balancing')
plt.xlabel('MBTI Type')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Save the cleaned and balanced data with emotions and emojis to a new CSV file
cleaned_filename = 'cleaned_balanced_mbti_with_emotions_and_emojis.csv'
balanced_df.to_csv(cleaned_filename, index=False)

print(f"Cleaned and balanced data saved to '{cleaned_filename}' with separate emotion and emoji columns.")













































