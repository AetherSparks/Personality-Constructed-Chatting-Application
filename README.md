# Personality Detection and Chatting Application

This project uses Natural Language Processing (NLP) and Machine Learning techniques to predict personality types based on text data. The dataset used for this project is the `mbti_1.csv` file, which contains user posts from the MBTI (Myers-Briggs Type Indicator) dataset.

## Project Overview

The goal of this project is to process, clean, and analyze the `mbti_1.csv` dataset and use the cleaned data to make predictions about users' MBTI personality types using a chatbot.

### Key Features:
- **Data Cleaning:** Clean and preprocess the raw `mbti_1.csv` dataset for NLP tasks.
- **Personality Prediction Bot:** A chatbot that predicts the personality type of a user based on their conversation.
- **Emotion Recognition:** The bot also predicts emotions and suggests emojis corresponding to the conversation.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Dataset Overview](#dataset-overview)
3. [Data Cleaning Process](#data-cleaning-process)
4. [Bot Usage](#bot-usage)
5. [Contributing](#contributing)
6. [License](#license)

---

## Prerequisites

Before starting, make sure you have the following installed on your local machine:

- **Python 3.x** (preferably Python 3.7 or higher)
- **Libraries:** Install the following libraries using pip:
  
  ```bash
  pip install pandas numpy nltk sklearn tensorflow transformers




Kaggle Account: To download the dataset, you need a Kaggle account. If you don't have one, sign up on Kaggle.
Dataset Overview
The dataset used in this project is mbti_1.csv, which contains the following columns:

type: MBTI personality type (e.g., ENFP, ISTJ, etc.)
posts: Text data containing user posts
cleaned_posts: Text data after preprocessing (you will generate this column after cleaning the dataset)
The dataset can be downloaded from Kaggle here.

Data Cleaning Process
The data cleaning process involves the following steps:

1. Download the Dataset
First, download the mbti_1.csv file from Kaggle. Save it in your working directory.

2. Load the Dataset
Use Pandas to load the dataset into a DataFrame:

python
Copy code
import pandas as pd

# Load the dataset
df = pd.read_csv('mbti_1.csv')
print(df.head())
3. Data Preprocessing
The goal of preprocessing is to clean and prepare the text for analysis. This includes:

Lowercasing: Convert all text to lowercase to ensure uniformity.
Removing special characters and punctuation: Remove any unnecessary symbols or punctuation.
Tokenization: Split text into individual words.
Removing stop words: Remove common words like "and", "the", "is", etc., which don't contribute much to analysis.
Stemming/Lemmatization: Reduce words to their root form (e.g., "running" to "run").
You can use the following code to clean the posts column:

python
Copy code
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters (i.e., special characters)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    words = text.split()
    
    # Remove stop words and lemmatize each word
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Rejoin words into a single string
    return ' '.join(words)

# Apply cleaning function to the 'posts' column
df['cleaned_posts'] = df['posts'].apply(clean_text)

# Save cleaned data to a new CSV file
df.to_csv('cleaned_mbti.csv', index=False)
print(df.head())
4. Handling Imbalanced Data
If your dataset has an imbalance (e.g., one MBTI type is overrepresented), you can either:

Oversample the minority class using techniques like SMOTE (Synthetic Minority Oversampling Technique).
Undersample the majority class by randomly selecting samples from the overrepresented classes.
This can be done using libraries like imbalanced-learn.

Bot Usage
Once the data is cleaned, you can use the processed data with your chatbot for personality type prediction.

1. Load the Pretrained Model
You'll need a trained model for MBTI personality type classification. The model should take the cleaned_posts column as input and predict the personality type.

python
Copy code
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pretrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to predict MBTI type
def predict_personality(post):
    inputs = tokenizer(post, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1)
    return predicted_class.item()

# Example usage
test_post = "I enjoy socializing with new people."
predicted_personality = predict_personality(test_post)
print(predicted_personality)
You can also modify the chatbot to include emotion recognition and emoji predictions as part of the response.



