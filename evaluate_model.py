from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model path
model_path = './results'

# Check if model directory exists
if not os.path.exists(model_path):
    logging.error(f"Model path '{model_path}' does not exist.")
    exit()

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    exit()

# Define evaluation inputs
evaluation_prompts = [
    "Describe Tanjiro Kamado's personality.",
    "What is Zenitsu Agatsuma like?",
    "Tell me about Muzan Kibutsuji."
]

# Generate and print responses
for prompt in evaluation_prompts:
    try:
        response = text_generator(prompt, max_length=50)
        logging.info(f"Input: {prompt}\nOutput: {response[0]['generated_text']}\n")
    except Exception as e:
        logging.error(f"Error generating response for prompt '{prompt}': {e}")


