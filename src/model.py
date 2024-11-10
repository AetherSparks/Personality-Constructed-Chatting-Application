import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, BertModel
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
print("Loading dataset...")
df = pd.read_csv('dataset(cleaned too)/cleaned_balanced_mbti_with_emotions_and_emojis.csv')
print("Dataset loaded successfully.\n")

# Encode MBTI types, emotions, and emojis to numerical labels
print("Encoding labels for MBTI types, emotions, and emojis...")
label_encoder_type = LabelEncoder()
df['label_type'] = label_encoder_type.fit_transform(df['type'])

label_encoder_emotion = LabelEncoder()
df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])

label_encoder_emoji = LabelEncoder()
df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
print("Encoding complete.\n")

# Split into training and validation sets
print("Splitting dataset into training and validation sets...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("Dataset split complete.\n")

# Load tokenizer
print("Loading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded.\n")

# Tokenization function with progress tracking
def tokenize_function(posts):
    print("Tokenizing data...")
    encodings = {"input_ids": [], "attention_mask": []}
    for post in tqdm(posts, desc="Tokenizing posts"):
        encoded = tokenizer(post, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        encodings['input_ids'].append(encoded['input_ids'][0])
        encodings['attention_mask'].append(encoded['attention_mask'][0])
    return {key: torch.stack(val) for key, val in encodings.items()}

# Tokenize datasets
train_encodings = tokenize_function(train_df['cleaned_posts'].tolist())
val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
print("Tokenization complete.\n")

# Custom multi-output model class using BERT
class MultiOutputBERT(nn.Module):
    def __init__(self, num_labels_type, num_labels_emotion, num_labels_emoji):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier_type = nn.Linear(self.bert.config.hidden_size, num_labels_type)
        self.classifier_emotion = nn.Linear(self.bert.config.hidden_size, num_labels_emotion)
        self.classifier_emoji = nn.Linear(self.bert.config.hidden_size, num_labels_emoji)

    def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        logits_type = self.classifier_type(pooled_output)
        logits_emotion = self.classifier_emotion(pooled_output)
        logits_emoji = self.classifier_emoji(pooled_output)

        loss = None
        if labels_type is not None and labels_emotion is not None and labels_emoji is not None:
            loss_fct = CrossEntropyLoss()  # No class weights specified
            loss_type = loss_fct(logits_type, labels_type)
            loss_emotion = loss_fct(logits_emotion, labels_emotion)
            loss_emoji = loss_fct(logits_emoji, labels_emoji)
            loss = loss_type + loss_emotion + loss_emoji

        return (loss, logits_type, logits_emotion, logits_emoji) if loss is not None else (logits_type, logits_emotion, logits_emoji)

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # Save the underlying BERT model
        self.bert.save_pretrained(save_directory)
        # Save custom components as a checkpoint
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        print(f"Model saved to {save_directory}")

# Custom dataset class for multi-output
class MultiOutputMBTIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_type, labels_emotion, labels_emoji):
        self.encodings = encodings
        self.labels_type = labels_type
        self.labels_emotion = labels_emotion
        self.labels_emoji = labels_emoji

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels_type'] = torch.tensor(self.labels_type[idx], dtype=torch.long)
        item['labels_emotion'] = torch.tensor(self.labels_emotion[idx], dtype=torch.long)
        item['labels_emoji'] = torch.tensor(self.labels_emoji[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels_type)

# Create dataset objects
train_dataset = MultiOutputMBTIDataset(
    train_encodings,
    train_df['label_type'].tolist(),
    train_df['label_emotion'].tolist(),
    train_df['label_emoji'].tolist()
)
val_dataset = MultiOutputMBTIDataset(
    val_encodings,
    val_df['label_type'].tolist(),
    val_df['label_emotion'].tolist(),
    val_df['label_emoji'].tolist()
)

# Instantiate the model
model = MultiOutputBERT(
    num_labels_type=len(label_encoder_type.classes_),
    num_labels_emotion=len(label_encoder_emotion.classes_),
    num_labels_emoji=len(label_encoder_emoji.classes_)
)
model.to(device)

# Define training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir='./models/checkpoints',
    eval_strategy="epoch",  # Use `eval_strategy` instead of deprecated `evaluation_strategy`
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=5,
    fp16=torch.cuda.is_available(),  # Enable FP16 if GPU is available
    logging_dir='./logs',
    report_to="none",
    load_best_model_at_end=True
)
print("Training arguments set.\n")

# Initialize Trainer
print("Initializing the Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
print("Trainer initialized.\n")
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
save_dir = './models/fine_tuned_bert_cleaned_model'
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Model and tokenizer saved successfully.")















































































































# import os
# import torch
# import torch.nn as nn
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments ,AutoModel   
# from sklearn.metrics import classification_report
# from tqdm import tqdm
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight
# from torch.nn import CrossEntropyLoss

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load dataset
# print("Loading dataset...")
# try:
#     df = pd.read_csv('dataset(cleaned too)/cleaned_balanced_mbti_with_emotions_and_emojis.csv')
#     print("Dataset loaded successfully.\n")
# except Exception as e:
#     print(f"Error loading dataset: {e}")
#     raise

# # Encode MBTI types, emotions, and emojis to numerical labels
# print("Encoding labels for MBTI types, emotions, and emojis...")
# try:
#     label_encoder_type = LabelEncoder()
#     df['label_type'] = label_encoder_type.fit_transform(df['type'])
#     print(f"Label encoder classes for MBTI types: {label_encoder_type.classes_}")

#     label_encoder_emotion = LabelEncoder()
#     df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])
#     print(f"Label encoder classes for emotions: {label_encoder_emotion.classes_}")

#     label_encoder_emoji = LabelEncoder()
#     df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
#     print(f"Label encoder classes for emojis: {label_encoder_emoji.classes_}")

#     print("Encoding complete.\n")
# except Exception as e:
#     print(f"Error during label encoding: {e}")
#     raise

# # Split into training and validation sets
# print("Splitting dataset into training and validation sets...")
# try:
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
#     print("Dataset split complete.\n")
# except Exception as e:
#     print(f"Error splitting dataset: {e}")
#     raise

# # Load tokenizer for RoBERTa
# print("Loading RoBERTa tokenizer...")
# try:
#     tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#     print("Tokenizer loaded.\n")
# except Exception as e:
#     print(f"Error loading tokenizer: {e}")
#     raise

# # Tokenization function with progress tracking
# def tokenize_function(posts):
#     print("Tokenizing data...")
#     try:
#         encodings = {"input_ids": [], "attention_mask": []}
#         for post in tqdm(posts, desc="Tokenizing posts"):
#             encoded = tokenizer(post, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#             encodings['input_ids'].append(encoded['input_ids'][0])
#             encodings['attention_mask'].append(encoded['attention_mask'][0])
#         print("Tokenization successful.")
#         return {key: torch.stack(val) for key, val in encodings.items()}
#     except Exception as e:
#         print(f"Error during tokenization: {e}")
#         raise

# # Tokenize datasets
# try:
#     train_encodings = tokenize_function(train_df['cleaned_posts'].tolist())
#     val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
#     print("Tokenization complete.\n")
# except Exception as e:
#     print(f"Error during tokenization of datasets: {e}")
#     raise

# # Custom multi-output model class using RoBERTa
# class MultiOutputRoBERTa(nn.Module):
#     def __init__(self, num_labels_type, num_labels_emotion, num_labels_emoji):
#         super(MultiOutputRoBERTa, self).__init__()
#         print("Initializing MultiOutputRoBERTa model...")
#         self.roberta = AutoModel.from_pretrained('roberta-base')
#         self.dropout = nn.Dropout(0.3)
#         self.classifier_type = nn.Linear(self.roberta.config.hidden_size, num_labels_type)
#         self.classifier_emotion = nn.Linear(self.roberta.config.hidden_size, num_labels_emotion)
#         self.classifier_emoji = nn.Linear(self.roberta.config.hidden_size, num_labels_emoji)
#         print("Model initialized successfully.")

#     def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None):
#         try:
#             outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#             pooled_output = self.dropout(outputs.pooler_output)

#             logits_type = self.classifier_type(pooled_output)
#             logits_emotion = self.classifier_emotion(pooled_output)
#             logits_emoji = self.classifier_emoji(pooled_output)

#             loss = None
#             if labels_type is not None and labels_emotion is not None and labels_emoji is not None:
#                 loss_fct = CrossEntropyLoss()
#                 loss_type = loss_fct(logits_type, labels_type)
#                 loss_emotion = loss_fct(logits_emotion, labels_emotion)
#                 loss_emoji = loss_fct(logits_emoji, labels_emoji)
#                 loss = loss_type + loss_emotion + loss_emoji

#             return (loss, logits_type, logits_emotion, logits_emoji) if loss is not None else (logits_type, logits_emotion, logits_emoji)
#         except Exception as e:
#             print(f"Error during forward pass: {e}")
#             raise

#     def save_pretrained(self, save_directory):
#         try:
#             if not os.path.exists(save_directory):
#                 os.makedirs(save_directory)
#             self.roberta.save_pretrained(save_directory)
#             torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
#             print(f"Model saved to {save_directory}")
#         except Exception as e:
#             print(f"Error saving model: {e}")
#             raise

# # Compute class weights based on the training data distribution
# print("Computing class weights for emotion labels...")
# try:
#     unique_classes = np.unique(df['label_emotion'])
#     class_weights = compute_class_weight('balanced', classes=unique_classes, y=df['label_emotion'])
#     class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
#     print(f"Computed class weights: {class_weights}")

#     num_classes = len(unique_classes)
#     assert num_classes == 7, f"Expected 7 classes for emotions, but found {num_classes}"
#     assert len(class_weights) == num_classes, f"Expected class weights tensor size to be {num_classes}, but got {len(class_weights)}"
#     print("Class weights verified.\n")
# except Exception as e:
#     print(f"Error computing class weights: {e}")
#     raise

# # Custom dataset class for multi-output
# class MultiOutputMBTIDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels_type, labels_emotion, labels_emoji):
#         self.encodings = encodings
#         self.labels_type = labels_type
#         self.labels_emotion = labels_emotion
#         self.labels_emoji = labels_emoji

#     def __getitem__(self, idx):
#         try:
#             item = {key: val[idx] for key, val in self.encodings.items()}
#             item['labels_type'] = torch.tensor(self.labels_type[idx], dtype=torch.long)
#             item['labels_emotion'] = torch.tensor(self.labels_emotion[idx], dtype=torch.long)
#             item['labels_emoji'] = torch.tensor(self.labels_emoji[idx], dtype=torch.long)
#             return item
#         except Exception as e:
#             print(f"Error retrieving item at index {idx}: {e}")
#             raise

#     def __len__(self):
#         return len(self.labels_type)

# # Create dataset objects
# try:
#     train_dataset = MultiOutputMBTIDataset(
#         train_encodings,
#         train_df['label_type'].tolist(),
#         train_df['label_emotion'].tolist(),
#         train_df['label_emoji'].tolist()
#     )
#     val_dataset = MultiOutputMBTIDataset(
#         val_encodings,
#         val_df['label_type'].tolist(),
#         val_df['label_emotion'].tolist(),
#         val_df['label_emoji'].tolist()
#     )
#     print("Dataset objects created successfully.\n")
# except Exception as e:
#     print(f"Error creating dataset objects: {e}")
#     raise

# # Instantiate the model
# try:
#     model = MultiOutputRoBERTa(
#         num_labels_type=len(label_encoder_type.classes_),
#         num_labels_emotion=len(label_encoder_emotion.classes_),
#         num_labels_emoji=len(label_encoder_emoji.classes_)
#     )
#     model.to(device)
#     print("Model instantiated and moved to device.\n")
# except Exception as e:
#     print(f"Error instantiating the model: {e}")
#     raise

# # Define training arguments
# print("Setting up training arguments...")
# try:
#     training_args = TrainingArguments(
#         output_dir='./models/checkpoints',
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         learning_rate=5e-5,
#         num_train_epochs=5,
#         fp16=torch.cuda.is_available(),
#         logging_dir='./logs',
#         report_to="none",
#         load_best_model_at_end=True
#     )
#     print("Training arguments set.\n")
# except Exception as e:
#     print(f"Error setting up training arguments: {e}")
#     raise

# print("Setup complete. You can now proceed with training.")

# # Initialize Trainer
# print("Initializing the Trainer...")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )
# print("Trainer initialized.\n")

# # Start training
# print("Starting training...")
# try:
#     trainer.train()
#     print("Training completed.\n")
# except Exception as e:
#     print(f"Error during training: {e}")
#     raise
# # Save the fine-tuned model
# print("Saving the fine-tuned model...")
# save_dir = './models/fine_tuned_roberta_cleaned_model'
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)
# print("Model and tokenizer saved successfully.")


































































# import os
# import torch
# import torch.nn as nn
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
# from sklearn.utils.class_weight import compute_class_weight
# from tqdm import tqdm

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load dataset
# print("Loading dataset...")
# try:
#     df = pd.read_csv('dataset(cleaned too)/cleaned_balanced_mbti_with_emotions_and_emojis.csv')
#     print("Dataset loaded successfully.\n")
# except Exception as e:
#     print(f"Error loading dataset: {e}")
#     raise

# # Encode MBTI types, emotions, and emojis to numerical labels
# print("Encoding labels for MBTI types, emotions, and emojis...")
# try:
#     label_encoder_type = LabelEncoder()
#     df['label_type'] = label_encoder_type.fit_transform(df['type'])
#     print(f"Label encoder classes for MBTI types: {label_encoder_type.classes_}")

#     label_encoder_emotion = LabelEncoder()
#     df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])
#     print(f"Label encoder classes for emotions: {label_encoder_emotion.classes_}")

#     label_encoder_emoji = LabelEncoder()
#     df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
#     print(f"Label encoder classes for emojis: {label_encoder_emoji.classes_}")

#     print("Encoding complete.\n")
# except Exception as e:
#     print(f"Error during label encoding: {e}")
#     raise

# # Split into training and validation sets
# print("Splitting dataset into training and validation sets...")
# try:
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
#     print("Dataset split complete.\n")
# except Exception as e:
#     print(f"Error splitting dataset: {e}")
#     raise

# # Load tokenizer for DistilGPT2
# print("Loading DistilGPT2 tokenizer...")
# try:
#     tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
#     tokenizer.pad_token = tokenizer.eos_token
#     print("Tokenizer loaded.\n")
# except Exception as e:
#     print(f"Error loading tokenizer: {e}")
#     raise

# # Tokenization function with progress tracking
# def tokenize_function(posts):
#     print("Tokenizing data...")
#     try:
#         encodings = {"input_ids": [], "attention_mask": []}
#         for post in tqdm(posts, desc="Tokenizing posts"):
#             encoded = tokenizer(post, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#             encodings['input_ids'].append(encoded['input_ids'][0])
#             encodings['attention_mask'].append(encoded['attention_mask'][0])
#         print("Tokenization successful.")
#         return {key: torch.stack(val) for key, val in encodings.items()}
#     except Exception as e:
#         print(f"Error during tokenization: {e}")
#         raise

# # Tokenize datasets
# try:
#     train_encodings = tokenize_function(train_df['cleaned_posts'].tolist())
#     val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
#     print("Tokenization complete.\n")
# except Exception as e:
#     print(f"Error during tokenization of datasets: {e}")
#     raise

# # Custom multi-output model class using DistilGPT2
# class MultiOutputDistilGPT2(nn.Module):
#     def __init__(self, num_labels_type, num_labels_emotion, num_labels_emoji):
#         super(MultiOutputDistilGPT2, self).__init__()
#         print("Initializing MultiOutputDistilGPT2 model...")
        
#         # Use DistilGPT2 instead of RoBERTa
#         self.gpt2 = AutoModel.from_pretrained('distilgpt2')
        
#         # A dropout layer
#         self.dropout = nn.Dropout(0.3)
        
#         # Classifiers for each output
#         self.classifier_type = nn.Linear(self.gpt2.config.hidden_size, num_labels_type)
#         self.classifier_emotion = nn.Linear(self.gpt2.config.hidden_size, num_labels_emotion)
#         self.classifier_emoji = nn.Linear(self.gpt2.config.hidden_size, num_labels_emoji)
        
#         print("Model initialized successfully.")

#     def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None , **kwargs):
#         try:
#             # Forward pass through DistilGPT2
#             outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            
#             # Use the hidden state of the last token (for classification)
#             last_hidden_state = outputs.last_hidden_state
#             pooled_output = last_hidden_state[:, -1]  # Select last token's hidden state

#             # Apply dropout
#             pooled_output = self.dropout(pooled_output)

#             # Get logits for each output
#             logits_type = self.classifier_type(pooled_output)
#             logits_emotion = self.classifier_emotion(pooled_output)
#             logits_emoji = self.classifier_emoji(pooled_output)

#             # Calculate loss if labels are provided
#             loss = None
#             if labels_type is not None and labels_emotion is not None and labels_emoji is not None:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss_type = loss_fct(logits_type, labels_type)
#                 loss_emotion = loss_fct(logits_emotion, labels_emotion)
#                 loss_emoji = loss_fct(logits_emoji, labels_emoji)
#                 loss = loss_type + loss_emotion + loss_emoji

#             return (loss, logits_type, logits_emotion, logits_emoji) if loss is not None else (logits_type, logits_emotion, logits_emoji)
        
#         except Exception as e:
#             print(f"Error during forward pass: {e}")
#             raise

#     def save_pretrained(self, save_directory):
#         try:
#             if not os.path.exists(save_directory):
#                 os.makedirs(save_directory)
#             self.gpt2.save_pretrained(save_directory)
#             torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
#             print(f"Model saved to {save_directory}")
#         except Exception as e:
#             print(f"Error saving model: {e}")
#             raise

# # Custom dataset class for multi-output
# class MultiOutputMBTIDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels_type, labels_emotion, labels_emoji):
#         self.encodings = encodings
#         self.labels_type = labels_type
#         self.labels_emotion = labels_emotion
#         self.labels_emoji = labels_emoji

#     def __getitem__(self, idx):
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item['labels_type'] = torch.tensor(self.labels_type[idx], dtype=torch.long)
#         item['labels_emotion'] = torch.tensor(self.labels_emotion[idx], dtype=torch.long)
#         item['labels_emoji'] = torch.tensor(self.labels_emoji[idx], dtype=torch.long)
#         return item

#     def __len__(self):
#         return len(self.labels_type)

# # Create dataset objects
# try:
#     train_dataset = MultiOutputMBTIDataset(
#         train_encodings,
#         train_df['label_type'].tolist(),
#         train_df['label_emotion'].tolist(),
#         train_df['label_emoji'].tolist()
#     )
#     val_dataset = MultiOutputMBTIDataset(
#         val_encodings,
#         val_df['label_type'].tolist(),
#         val_df['label_emotion'].tolist(),
#         val_df['label_emoji'].tolist()
#     )
#     print("Dataset objects created successfully.\n")
# except Exception as e:
#     print(f"Error creating dataset objects: {e}")
#     raise

# # Instantiate the model
# try:
#     model = MultiOutputDistilGPT2(
#         num_labels_type=len(label_encoder_type.classes_),
#         num_labels_emotion=len(label_encoder_emotion.classes_),
#         num_labels_emoji=len(label_encoder_emoji.classes_)
#     )
#     model.to(device)
#     print("Model instantiated and moved to device.\n")
# except Exception as e:
#     print(f"Error instantiating the model: {e}")
#     raise

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./models/checkpoints',
#     eval_strategy="epoch",  # Make sure this matches save_strategy
#     save_strategy="epoch",  # Set both to 'epoch'
#     learning_rate=5e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     save_steps=500,
#     load_best_model_at_end=True,
# )

# # Trainer initialization
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Training the model
# try:
#     print("Starting training...")
#     trainer.train()
#     print("Training complete.\n")
# except Exception as e:
#     print(f"Error during training: {e}")
#     raise

# # Save the fine-tuned model
# print("Saving the fine-tuned model...")
# save_dir = './models/fine_tuned_distillgpt2_cleaned_model'
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)
# print(f"Model saved to {save_dir}")

















































# import os
# import torch
# import torch.nn as nn
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoModel
# from sklearn.utils.class_weight import compute_class_weight
# from tqdm import tqdm

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load dataset
# print("Loading dataset...")
# try:
#     df = pd.read_csv('dataset(cleaned too)\\cleaned_balanced_mbti_with_emotions_and_emojis.csv')
#     print("Dataset loaded successfully.\n")
# except Exception as e:
#     print(f"Error loading dataset: {e}")
#     raise

# # Encode MBTI types, emotions, and emojis to numerical labels
# print("Encoding labels for MBTI types, emotions, and emojis...")
# try:
#     label_encoder_type = LabelEncoder()
#     df['label_type'] = label_encoder_type.fit_transform(df['type'])
#     label_encoder_emotion = LabelEncoder()
#     df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])
#     label_encoder_emoji = LabelEncoder()
#     df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
#     print("Encoding complete.\n")
# except Exception as e:
#     print(f"Error during label encoding: {e}")
#     raise

# # Split into training and validation sets
# print("Splitting dataset into training and validation sets...")
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# print("Dataset split complete.\n")

# # Load tokenizer for DeBERTa
# print("Loading DeBERTa tokenizer...")
# try:
#     tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', use_fast=False)
#     if tokenizer.pad_token is None: 
#         tokenizer.pad_token = tokenizer.eos_token  # Assign eos_token as the pad_token
#     print("Tokenizer loaded.\n")
# except Exception as e:
#     print(f"Error loading tokenizer: {e}")
#     raise

# # Tokenization function
# def tokenize_function(posts):
#     print("Tokenizing data...")
#     encodings = tokenizer(posts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#     print("Tokenization successful.")
#     return encodings

# # Tokenize datasets
# train_encodings = tokenize_function(train_df['cleaned_posts'].tolist())
# val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
# print("Tokenization complete.\n")

# # Custom multi-output model class using DeBERTa
# class MultiOutputDeBERTa(nn.Module):
#     def __init__(self, num_labels_type, num_labels_emotion, num_labels_emoji):
#         super(MultiOutputDeBERTa, self).__init__()
#         print("Initializing MultiOutputDeBERTa model...")
#         self.deberta = AutoModel.from_pretrained('microsoft/deberta-v3-small')
#         self.dropout = nn.Dropout(0.3)
#         self.classifier_type = nn.Linear(self.deberta.config.hidden_size, num_labels_type)
#         self.classifier_emotion = nn.Linear(self.deberta.config.hidden_size, num_labels_emotion)
#         self.classifier_emoji = nn.Linear(self.deberta.config.hidden_size, num_labels_emoji)
#         print("Model initialized successfully.")

#     def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None):
#         outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, 0]  # Get CLS token's hidden state
#         pooled_output = self.dropout(pooled_output)
#         logits_type = self.classifier_type(pooled_output)
#         logits_emotion = self.classifier_emotion(pooled_output)
#         logits_emoji = self.classifier_emoji(pooled_output)

#         loss = None
#         if labels_type is not None and labels_emotion is not None and labels_emoji is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss_type = loss_fct(logits_type, labels_type)
#             loss_emotion = loss_fct(logits_emotion, labels_emotion)
#             loss_emoji = loss_fct(logits_emoji, labels_emoji)
#             loss = loss_type + loss_emotion + loss_emoji

#         return (loss, logits_type, logits_emotion, logits_emoji) if loss is not None else (logits_type, logits_emotion, logits_emoji)

#     def save_pretrained(self, save_directory):
#         os.makedirs(save_directory, exist_ok=True)
#         self.deberta.save_pretrained(save_directory)
#         torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
#         print(f"Model saved to {save_directory}")

# # Custom dataset class for multi-output
# class MultiOutputMBTIDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels_type, labels_emotion, labels_emoji):
#         self.encodings = encodings
#         self.labels_type = labels_type
#         self.labels_emotion = labels_emotion
#         self.labels_emoji = labels_emoji

#     def __getitem__(self, idx):
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item['labels_type'] = torch.tensor(self.labels_type[idx], dtype=torch.long)
#         item['labels_emotion'] = torch.tensor(self.labels_emotion[idx], dtype=torch.long)
#         item['labels_emoji'] = torch.tensor(self.labels_emoji[idx], dtype=torch.long)
#         return item

#     def __len__(self):
#         return len(self.labels_type)

# # Instantiate datasets
# train_dataset = MultiOutputMBTIDataset(
#     train_encodings,
#     train_df['label_type'].tolist(),
#     train_df['label_emotion'].tolist(),
#     train_df['label_emoji'].tolist()
# )
# val_dataset = MultiOutputMBTIDataset(
#     val_encodings,
#     val_df['label_type'].tolist(),
#     val_df['label_emotion'].tolist(),
#     val_df['label_emoji'].tolist()
# )

# # Instantiate the model
# model = MultiOutputDeBERTa(
#     num_labels_type=len(label_encoder_type.classes_),
#     num_labels_emotion=len(label_encoder_emotion.classes_),
#     num_labels_emoji=len(label_encoder_emoji.classes_)
# )
# model.to(device)

# # Training arguments with memory-saving techniques
# training_args = TrainingArguments(
#     output_dir='./models/checkpoints',
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=16,  # Reduced batch size
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     save_steps=500,
#     load_best_model_at_end=True,
#     fp16=True,  # Use mixed precision
#     gradient_accumulation_steps=2  # Accumulate gradients to simulate a larger batch size
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Start training
# print("Starting training from the last checkpoint...")
# trainer.train(resume_from_checkpoint='models/checkpoints/checkpoint-1864')
# print("Training complete.\n")

# # Save the fine-tuned model
# print("Saving the fine-tuned model...")
# save_dir = './models/fine_tuned_deberta_model'
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)
# print(f"Model saved to {save_dir}")
