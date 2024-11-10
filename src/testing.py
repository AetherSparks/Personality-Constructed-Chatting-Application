import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Load the dataset (adjust the path based on where you store the CSV file locally)
dataset_path = "dataset(cleaned too)\\cleaned_balanced_mbti_with_emotions_and_emojis.csv"  # Adjust path as needed
df = pd.read_csv(dataset_path)

# Encode MBTI labels (if not already encoded)
label_encoder = LabelEncoder()
label_encoder.fit(df['type'])  # Assuming 'type' column has the MBTI types (e.g., 'INTJ', 'ENFP')

# Load the fine-tuned model and tokenizer (adjust to your model's path)
model_path = 'models\\fine_tuned_bert_cleaned_model'  # Adjust to your model's path where you have saved the fine-tuned model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Check if a GPU is available and move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict MBTI type from user input
def predict_mbti_type(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move input tensors to the same device as the model (CPU or GPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Perform inference without gradients
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted class
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Define a function to convert numerical class back to MBTI type
def get_mbti_type(predicted_class):
    # Convert predicted class to MBTI type using label_encoder
    return label_encoder.classes_[predicted_class]

# Example usage with a single input (can be tested in `testing.py`)
def test_prediction(text):
    predicted_class = predict_mbti_type(text)
    mbti_type = get_mbti_type(predicted_class)
    return mbti_type

# Example: Test a single text input
sample_text = "iwtf"
predicted_mbti = test_prediction(sample_text)
print(f"The predicted MBTI type is: {predicted_mbti}")

# Optional: If you want to process the whole dataset (for testing purposes)
def test_on_dataset(df):
    predictions = []
    for text in df['cleaned_posts']:  # Assuming 'posts' column contains text
        predicted_class = predict_mbti_type(text)
        mbti_type = get_mbti_type(predicted_class)
        predictions.append(mbti_type)
    
    # Add the predictions as a new column
    df['predicted_type'] = predictions
    return df

# Test the model on the entire dataset (uncomment to test)
result_df = test_on_dataset(df)
print(result_df[['cleaned_posts', 'predicted_type']].head())
