# import os
# import time
# from groq import Groq
# import os
# from groq import Groq

# # Set your API key here
# os.environ["GROQ_API_KEY"] = "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"

# client = Groq(api_key=os.environ["GROQ_API_KEY"])


# # Function to generate a dynamic response using Groq's chat completion method
# def get_groq_response(conversation_history):
#     try:
#         # Combine conversation history to provide context for the chat model
#         prompt = "\n".join(conversation_history)

#         # Generate the response from Groq's chat model
#         chat_completion = client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="llama3-8b-8192"
#         )

#         # Extract and return the bot's reply
#         bot_reply = chat_completion.choices[0].message.content
#         return bot_reply

#     except Exception as e:
#         return f"Error: {e}"
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer

# # Assuming you've already loaded your model and tokenizer
# model = BertForSequenceClassification.from_pretrained('models\\fine_tuned_bert_cleaned_model')
# tokenizer = BertTokenizer.from_pretrained('models\\fine_tuned_bert_cleaned_model')

# # Function to predict personality from conversation history
# def predict_personality_from_conversation(conversation_history):
#     # Join all the conversation into a single string
#     conversation_text = " ".join(conversation_history)

#     # Tokenize the conversation text and prepare input for the model
#     inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, padding=True)

#     # Use the model to make a prediction
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)  # Get the predicted class
    
#     # List of possible personality types (this should match your model's output classes)
#     personality_types = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", 
#                          "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]

#     # Return the predicted personality type based on the prediction
#     return personality_types[predictions.item()]

# # Chatbot function to handle user interaction and dynamic conversation
# # Chatbot function to handle user interaction and dynamic conversation
# def chatbot():
#     print("Hello! I am your personality chatbot. Let's chat! Type 'bye', 'end', or 'quit' when you're ready to finish.")
    
#     conversation_history = []  # Track the conversation
    
#     while True:
#         user_input = input("You: ")  # Get user input
        
#         # Check for exit words like 'bye', 'end', etc.
#         if any(bye_word in user_input.lower() for bye_word in ['bye', 'end', 'quit', 'exit', 'goodbye']):
#             print("Bot: Nice talking with you!")
#             print("Bot: Analyzing your personality...")

#             # Predict personality based on the conversation
#             personality = predict_personality_from_conversation(conversation_history)
#             print(f"Bot: Based on our conversation, your personality type is: {personality}")
#             break  # Exit the loop after displaying the personality prediction
        
#         # Add user input to the conversation history
#         conversation_history.append(f"You: {user_input}")
        
#         # Generate a dynamic response based on the conversation history
#         bot_response = get_groq_response(conversation_history)
        
#         # Add bot response to the conversation history
#         conversation_history.append(f"Bot: {bot_response}")
        
#         # Display bot's response
#         print(f"Bot: {bot_response}")
        
#         # Optional: Simulate a pause between responses
#         time.sleep(1)

# # Main function to start the chatbot
# if __name__ == "__main__":
#     chatbot()





















import os
import time
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from groq import Groq

# Initialize Groq client
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Load the CSV data
csv_file_path = 'dataset(cleaned too)\\cleaned_balanced_mbti_with_emotions_and_emojis.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file_path)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('models\\fine_tuned_bert_cleaned_model')
tokenizer = BertTokenizer.from_pretrained('models\\fine_tuned_bert_cleaned_model')

# Global conversation history
conversation_history = []

# Function to get a response from Groq
def get_groq_response(conversation_history):
    try:
        prompt = "\n".join(conversation_history)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        bot_reply = chat_completion.choices[0].message.content
        return bot_reply
    except Exception as e:
        return f"Error: {e}"

# Function to predict personality, emotion, and emoji
def predict_personality_and_emotion(conversation_text):
    # Tokenize and prepare input for the model
    inputs = tokenizer(conversation_text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    # Check if the prediction index matches the mapping correctly
    personality_types = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", 
                         "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]

    personality = personality_types[predictions.item()] if 0 <= predictions.item() < len(personality_types) else "Unknown"
    
    # Ensure the emotion and emoji are within bounds and properly mapped
    if not data.empty and predictions.item() < len(data):
        emotion = data['emotion'].iloc[predictions.item()]
        emoji = data['emoji'].iloc[predictions.item()]
    else:
        emotion = "neutral"
        emoji = "😐"

    return personality, emotion, emoji

# Function to predict emotion for each segment of the conversation
def predict_emotions_per_segment(conversation_history):
    segment_emotions = []
    
    for segment in conversation_history:
        inputs = tokenizer(segment, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)
        
        # Ensure the emotion and emoji are within bounds and properly mapped
        if not data.empty and prediction.item() < len(data):
            emotion = data['emotion'].iloc[prediction.item()]
            emoji = data['emoji'].iloc[prediction.item()]
        else:
            emotion = "neutral"
            emoji = "😐"
        
        segment_emotions.append((emotion, emoji))
    
    return segment_emotions

# Function to analyze the conversation history without averaging
def clean_and_analyze_conversation():
    segment_emotions = predict_emotions_per_segment(conversation_history)
    
    # Count the frequency of each emotion
    emotion_counts = {}
    for emotion, emoji in segment_emotions:
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 1
        else:
            emotion_counts[emotion] += 1
    
    # Find the most common emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    corresponding_emoji = next(emoji for emotion, emoji in segment_emotions if emotion == dominant_emotion)
    
    # Combine conversation for personality prediction
    combined_text = " ".join(conversation_history)
    personality, _, _ = predict_personality_and_emotion(combined_text)
    
    print(f"Bot: Based on our conversation, your personality type is: {personality}")
    print(f"Bot: The dominant emotion is: {dominant_emotion} {corresponding_emoji}")

# Chatbot function for interaction
def chatbot():
    print("Hello! I am your personality chatbot. Let's chat! Type 'bye', 'end', or 'quit' when you're ready to finish.")
    
    global conversation_history  # Use the global conversation history list
    
    while True:
        user_input = input("You: ")  # Get user input
        
        if any(bye_word in user_input.lower() for bye_word in ['bye', 'end', 'quit', 'exit', 'goodbye']):
            print("Bot: Nice talking with you!")
            print("Bot: Analyzing your personality and emotion...")
            
            # Clean and analyze the conversation
            clean_and_analyze_conversation()
            break
        
        # Add user input to the conversation history
        conversation_history.append(user_input)
        
        # Generate a dynamic response from Groq without repeating history
        bot_response = get_groq_response(conversation_history[-1:])
        print(f"Bot: {bot_response}")
        
        # Add bot's response to the conversation history
        conversation_history.append(bot_response)
        
        time.sleep(1)

# Run the chatbot
if __name__ == "__main__":
    chatbot()
