import json
from datasets import Dataset

# Load the JSON data
with open('./data/MapTraits.json', 'r') as f:
    data = json.load(f)

# Prepare the dataset
texts = []
for character in data['characters']:
    text = f"{character['name']} has the following traits:\n"
    text += "Personality Traits: " + ", ".join(character['personality_traits']) + "\n"
    text += "Behavioral Traits: " + ", ".join(character['behavioral_traits'])
    texts.append(text)

dataset = Dataset.from_dict({"text": texts})

# Save the dataset to disk
dataset.save_to_disk('./data/MapTraitsDataset')

print("Data preparation complete.")
