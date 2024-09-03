from transformers import pipeline

# Load the fine-tuned model
model_path = './results'
text_generator = pipeline('text-generation', model=model_path)

# Generate a response
input_text = "Tell me about Tanjiro Kamado."
response = text_generator(input_text, max_length=50)

print(response[0]['generated_text'])
