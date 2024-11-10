import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('dataset(cleaned too)/cleaned_balanced_mbti_with_emotions_and_emojis.csv')
print("Dataset loaded successfully.\n")

# Encode labels
print("Encoding labels...")
label_encoder_type = LabelEncoder()
df['label_type'] = label_encoder_type.fit_transform(df['type'])

label_encoder_emotion = LabelEncoder()
df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])

label_encoder_emoji = LabelEncoder()
df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
print("Encoding complete.\n")

# Split dataset
print("Splitting dataset...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("Dataset split complete.\n")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded.\n")

# Tokenization function
def tokenize_function(posts):
    print("Tokenizing data...")
    return tokenizer(posts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# Tokenize validation dataset
val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
print("Tokenization complete.\n")

# Custom dataset class
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

# Create validation dataset object
val_dataset = MultiOutputMBTIDataset(
    val_encodings,
    val_df['label_type'].tolist(),
    val_df['label_emotion'].tolist(),
    val_df['label_emoji'].tolist()
)

# Multi-output BERT model
class MultiOutputBERT(nn.Module):
    def __init__(self, num_labels_type, num_labels_emotion, num_labels_emoji):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier_type = nn.Linear(self.bert.config.hidden_size, num_labels_type)
        self.classifier_emotion = nn.Linear(self.bert.config.hidden_size, num_labels_emotion)
        self.classifier_emoji = nn.Linear(self.bert.config.hidden_size, num_labels_emoji)

    def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None, **kwargs):
        # Pass input IDs and attention mask to the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        logits_type = self.classifier_type(pooled_output)
        logits_emotion = self.classifier_emotion(pooled_output)
        logits_emoji = self.classifier_emoji(pooled_output)

        loss = None
        if labels_type is not None and labels_emotion is not None and labels_emoji is not None:
            loss_fct = nn.CrossEntropyLoss()  # Standard cross-entropy loss
            loss_type = loss_fct(logits_type, labels_type)
            loss_emotion = loss_fct(logits_emotion, labels_emotion)
            loss_emoji = loss_fct(logits_emoji, labels_emoji)
            loss = loss_type + loss_emotion + loss_emoji

        return (loss, logits_type, logits_emotion, logits_emoji) if loss is not None else (logits_type, logits_emotion, logits_emoji)


# Load pre-trained model
print("Loading pre-trained model...")
model_path = './models/fine_tuned_bert_cleaned_model'
model = MultiOutputBERT(
    num_labels_type=len(label_encoder_type.classes_),
    num_labels_emotion=len(label_encoder_emotion.classes_),
    num_labels_emoji=len(label_encoder_emoji.classes_),
)
model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
model.to(device)
print("Model loaded.\n")

# Initialize DataLoader for validation set
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Evaluate model and collect predictions
print("Generating predictions...")

model.eval()
y_true_type = val_df['label_type'].tolist()
y_true_emotion = val_df['label_emotion'].tolist()
y_true_emoji = val_df['label_emoji'].tolist()

y_pred_type = []
y_pred_emotion = []
y_pred_emoji = []

# Iterate over batches
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass through the model
        logits_type, logits_emotion, logits_emoji = model(input_ids=input_ids, attention_mask=attention_mask)[:3]

        # Get predictions (use argmax to get the class with highest probability)
        pred_type = torch.argmax(logits_type, dim=1)
        pred_emotion = torch.argmax(logits_emotion, dim=1)
        pred_emoji = torch.argmax(logits_emoji, dim=1)

        # Store predictions
        y_pred_type.extend(pred_type.cpu().numpy())
        y_pred_emotion.extend(pred_emotion.cpu().numpy())
        y_pred_emoji.extend(pred_emoji.cpu().numpy())

# Generate classification reports for type, emotion, and emoji predictions
print("Generating classification reports for all outputs...")

# Generate classification report for MBTI type
print("\nClassification Report for MBTI Type:")
print(classification_report(
    y_true_type,
    y_pred_type,
    target_names=label_encoder_type.classes_,
    labels=np.unique(y_true_type)
))

# Generate classification report for emotions
print("\nClassification Report for Emotions:")
print(classification_report(
    y_true_emotion,
    y_pred_emotion,
    target_names=label_encoder_emotion.classes_,
    labels=np.unique(y_true_emotion)
))

# Generate classification report for emojis
print("\nClassification Report for Emojis:")
print(classification_report(
    y_true_emoji,
    y_pred_emoji,
    target_names=label_encoder_emoji.classes_,
    labels=np.unique(y_true_emoji)
))

# Analyze and conclude relationships between MBTI types, emotions, and emojis
print("\nAnalyzing relationships between MBTI types and predictions...")

# Create a DataFrame to aggregate predictions and true labels
results_df = pd.DataFrame({
    'true_type': label_encoder_type.inverse_transform(y_true_type),
    'pred_type': label_encoder_type.inverse_transform(y_pred_type),
    'true_emotion': label_encoder_emotion.inverse_transform(y_true_emotion),
    'pred_emotion': label_encoder_emotion.inverse_transform(y_pred_emotion),
    'true_emoji': label_encoder_emoji.inverse_transform(y_true_emoji),
    'pred_emoji': label_encoder_emoji.inverse_transform(y_pred_emoji)
})

# Group by predicted MBTI types and analyze the corresponding emotions and emojis
type_emotion_relation = results_df.groupby('pred_type')['pred_emotion'].value_counts(normalize=True).unstack()
type_emoji_relation = results_df.groupby('pred_type')['pred_emoji'].value_counts(normalize=True).unstack()

# Print top relationships for analysis
print("\nTop predicted emotions associated with each MBTI type:")
print(type_emotion_relation)

print("\nTop predicted emojis associated with each MBTI type:")
print(type_emoji_relation)




# Function to plot classification report as a bar chart
def plot_classification_report(report, title="Classification Report"):
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=report_df.index, y=report_df['f1-score'], palette="viridis")
    plt.title(f"{title} - F1 Score")
    plt.xlabel('Classes')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Generate and plot classification report for MBTI type, emotions, and emojis
def generate_and_plot_classification_report(y_true, y_pred, labels, title):
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True
    )
    plot_classification_report(report, title)

# Generate classification reports and plot for each category
generate_and_plot_classification_report(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type Classification Report")
generate_and_plot_classification_report(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion Classification Report")
generate_and_plot_classification_report(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji Classification Report")


def plot_class_distribution(y_true, y_pred, labels, title="Class Distribution Comparison"):
    true_counts = pd.Series(y_true).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()

    df = pd.DataFrame({
        'True': true_counts,
        'Predicted': pred_counts
    }).reindex(labels, fill_value=0)

    df.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot class distribution comparison for MBTI, emotion, and emoji
plot_class_distribution(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type Class Distribution")
plot_class_distribution(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion Class Distribution")
plot_class_distribution(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji Class Distribution")

def plot_distribution(df, title="Prediction Distribution"):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pred_emotion'], kde=True, color='blue', label='Predicted Emotion')
    plt.title(title)
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

plot_distribution(results_df, "Prediction Distribution of Emotions")


def plot_most_common_predictions(df, title="Most Common Predictions"):
    # Top 10 predicted emotions and emojis
    top_emotions = df['pred_emotion'].value_counts().head(10)
    top_emojis = df['pred_emoji'].value_counts().head(10)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(x=top_emotions.index, y=top_emotions.values, ax=axes[0], palette='Blues')
    axes[0].set_title("Top 10 Predicted Emotions")
    axes[0].set_xlabel('Emotions')
    axes[0].set_ylabel('Count')

    sns.barplot(x=top_emojis.index, y=top_emojis.values, ax=axes[1], palette='Greens')
    axes[1].set_title("Top 10 Predicted Emojis")
    axes[1].set_xlabel('Emojis')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

# Plot top 10 most common emotions and emojis
plot_most_common_predictions(results_df)








































# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, RobertaModel
# from sklearn.metrics import classification_report
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import numpy as np
# from torch import nn
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report
# import numpy as np
# import pandas as pd
# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the dataset
# print("Loading dataset...")
# df = pd.read_csv('dataset(cleaned too)/cleaned_balanced_mbti_with_emotions_and_emojis.csv')
# print("Dataset loaded successfully.\n")

# # Encode labels
# print("Encoding labels...")
# label_encoder_type = LabelEncoder()
# df['label_type'] = label_encoder_type.fit_transform(df['type'])

# label_encoder_emotion = LabelEncoder()
# df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])

# label_encoder_emoji = LabelEncoder()
# df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
# print("Encoding complete.\n")

# # Split dataset
# print("Splitting dataset...")
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# print("Dataset split complete.\n")

# # Load tokenizer
# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# print("Tokenizer loaded.\n")

# # Tokenization function
# def tokenize_function(posts):
#     print("Tokenizing data...")
#     return tokenizer(posts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# # Tokenize validation dataset
# val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
# print("Tokenization complete.\n")

# # Custom dataset class
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

# # Create validation dataset object
# val_dataset = MultiOutputMBTIDataset(
#     val_encodings,
#     val_df['label_type'].tolist(),
#     val_df['label_emotion'].tolist(),
#     val_df['label_emoji'].tolist()
# )

# # Multi-output RoBERTa model
# class MultiOutputRoBERTa(nn.Module):
#     def __init__(self, num_labels_type, num_labels_emotion, num_labels_emoji):
#         super(MultiOutputRoBERTa, self).__init__()
#         self.roberta = RobertaModel.from_pretrained('roberta-base')
#         self.dropout = nn.Dropout(0.3)
#         self.classifier_type = nn.Linear(self.roberta.config.hidden_size, num_labels_type)
#         self.classifier_emotion = nn.Linear(self.roberta.config.hidden_size, num_labels_emotion)
#         self.classifier_emoji = nn.Linear(self.roberta.config.hidden_size, num_labels_emoji)

#     def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None, **kwargs):
#         # Pass input IDs and attention mask to the RoBERTa model
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         # Use the first token ([CLS] equivalent) hidden state as the pooled output
#         pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        
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

# # Load pre-trained model
# print("Loading pre-trained model...")
# model_path = './models/fine_tuned_roberta_cleaned_model'
# model = MultiOutputRoBERTa(
#     num_labels_type=len(label_encoder_type.classes_),
#     num_labels_emotion=len(label_encoder_emotion.classes_),
#     num_labels_emoji=len(label_encoder_emoji.classes_)
# )
# model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
# model.to(device)
# print("Model loaded.\n")

# # Initialize DataLoader for validation set
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# # Evaluate model and collect predictions
# print("Generating predictions...")

# model.eval()
# y_true_type = val_df['label_type'].tolist()
# y_true_emotion = val_df['label_emotion'].tolist()
# y_true_emoji = val_df['label_emoji'].tolist()

# y_pred_type = []
# y_pred_emotion = []
# y_pred_emoji = []

# # Iterate over batches
# with torch.no_grad():
#     for batch in tqdm(val_loader, desc="Evaluating"):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)

#         # Forward pass through the model
#         logits_type, logits_emotion, logits_emoji = model(input_ids=input_ids, attention_mask=attention_mask)[:3]

#         # Get predictions
#         pred_type = torch.argmax(logits_type, dim=1)
#         pred_emotion = torch.argmax(logits_emotion, dim=1)
#         pred_emoji = torch.argmax(logits_emoji, dim=1)

#         # Store predictions
#         y_pred_type.extend(pred_type.cpu().numpy())
#         y_pred_emotion.extend(pred_emotion.cpu().numpy())
#         y_pred_emoji.extend(pred_emoji.cpu().numpy())

# # Generate classification reports for type, emotion, and emoji predictions
# print("Generating classification reports for all outputs...")

# # Generate classification report for MBTI type
# print("\nClassification Report for MBTI Type:")
# print(classification_report(
#     y_true_type,
#     y_pred_type,
#     target_names=label_encoder_type.classes_,
#     labels=np.unique(y_true_type)
# ))

# # Generate classification report for emotions
# print("\nClassification Report for Emotions:")
# print(classification_report(
#     y_true_emotion,
#     y_pred_emotion,
#     target_names=label_encoder_emotion.classes_,
#     labels=np.unique(y_true_emotion)
# ))

# # Generate classification report for emojis
# print("\nClassification Report for Emojis:")
# print(classification_report(
#     y_true_emoji,
#     y_pred_emoji,
#     target_names=label_encoder_emoji.classes_,
#     labels=np.unique(y_true_emoji)
# ))

# # Analyze and conclude relationships between MBTI types, emotions, and emojis
# print("\nAnalyzing relationships between MBTI types and predictions...")

# # Create a DataFrame to aggregate predictions and true labels
# results_df = pd.DataFrame({
#     'true_type': label_encoder_type.inverse_transform(y_true_type),
#     'pred_type': label_encoder_type.inverse_transform(y_pred_type),
#     'true_emotion': label_encoder_emotion.inverse_transform(y_true_emotion),
#     'pred_emotion': label_encoder_emotion.inverse_transform(y_pred_emotion),
#     'true_emoji': label_encoder_emoji.inverse_transform(y_true_emoji),
#     'pred_emoji': label_encoder_emoji.inverse_transform(y_pred_emoji)
# })

# # Group by predicted MBTI types and analyze the corresponding emotions and emojis
# type_emotion_relation = results_df.groupby('pred_type')['pred_emotion'].value_counts(normalize=True).unstack()
# type_emoji_relation = results_df.groupby('pred_type')['pred_emoji'].value_counts(normalize=True).unstack()

# # Print top relationships for analysis
# print("\nTop predicted emotions associated with each MBTI type:")
# print(type_emotion_relation)

# print("\nTop predicted emojis associated with each MBTI type:")
# print(type_emoji_relation)




# # Function to plot classification report as a bar chart
# def plot_classification_report(report, title="Classification Report"):
#     report_df = pd.DataFrame(report).transpose()
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=report_df.index, y=report_df['f1-score'], palette="viridis")
#     plt.title(f"{title} - F1 Score")
#     plt.xlabel('Classes')
#     plt.ylabel('F1-Score')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# # Generate and plot classification report for MBTI type, emotions, and emojis
# def generate_and_plot_classification_report(y_true, y_pred, labels, title):
#     report = classification_report(
#         y_true,
#         y_pred,
#         target_names=labels,
#         output_dict=True
#     )
#     plot_classification_report(report, title)

# # Generate classification reports and plot for each category
# generate_and_plot_classification_report(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type Classification Report")
# generate_and_plot_classification_report(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion Classification Report")
# generate_and_plot_classification_report(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji Classification Report")


# def plot_class_distribution(y_true, y_pred, labels, title="Class Distribution Comparison"):
#     true_counts = pd.Series(y_true).value_counts()
#     pred_counts = pd.Series(y_pred).value_counts()

#     df = pd.DataFrame({
#         'True': true_counts,
#         'Predicted': pred_counts
#     }).reindex(labels, fill_value=0)

#     df.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
#     plt.title(title)
#     plt.xlabel('Classes')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# # Plot class distribution comparison for MBTI, emotion, and emoji
# plot_class_distribution(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type Class Distribution")
# plot_class_distribution(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion Class Distribution")
# plot_class_distribution(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji Class Distribution")

# def plot_distribution(df, title="Prediction Distribution"):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df['pred_emotion'], kde=True, color='blue', label='Predicted Emotion')
#     plt.title(title)
#     plt.xlabel('Predicted Emotion')
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.show()

# plot_distribution(results_df, "Prediction Distribution of Emotions")


# def plot_most_common_predictions(df, title="Most Common Predictions"):
#     # Top 10 predicted emotions and emojis
#     top_emotions = df['pred_emotion'].value_counts().head(10)
#     top_emojis = df['pred_emoji'].value_counts().head(10)

#     # Plotting
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#     sns.barplot(x=top_emotions.index, y=top_emotions.values, ax=axes[0], palette='Blues')
#     axes[0].set_title("Top 10 Predicted Emotions")
#     axes[0].set_xlabel('Emotions')
#     axes[0].set_ylabel('Count')

#     sns.barplot(x=top_emojis.index, y=top_emojis.values, ax=axes[1], palette='Greens')
#     axes[1].set_title("Top 10 Predicted Emojis")
#     axes[1].set_xlabel('Emojis')
#     axes[1].set_ylabel('Count')

#     plt.tight_layout()
#     plt.show()

# # Plot top 10 most common emotions and emojis
# plot_most_common_predictions(results_df)






























# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, BertModel, Trainer, TrainingArguments , AutoModelForCausalLM ,AutoModel
# from sklearn.metrics import classification_report
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import numpy as np
# from torch import nn

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the dataset
# print("Loading dataset...")
# df = pd.read_csv('dataset(cleaned too)\\cleaned_balanced_mbti_with_emotions_and_emojis.csv')
# print("Dataset loaded successfully.\n")

# # Encode labels
# print("Encoding labels...")
# label_encoder_type = LabelEncoder()
# df['label_type'] = label_encoder_type.fit_transform(df['type'])

# label_encoder_emotion = LabelEncoder()
# df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])

# label_encoder_emoji = LabelEncoder()
# df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
# print("Encoding complete.\n")

# # Split dataset
# print("Splitting dataset...")
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# print("Dataset split complete.\n")

# # Load tokenizer
# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# print("Tokenizer loaded.\n")

# # Tokenization function
# def tokenize_function(posts):
#     print("Tokenizing data...")
#     return tokenizer(posts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# # Tokenize validation dataset
# val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
# print("Tokenization complete.\n")

# # Custom dataset class
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

# # Create validation dataset object
# val_dataset = MultiOutputMBTIDataset(
#     val_encodings,
#     val_df['label_type'].tolist(),
#     val_df['label_emotion'].tolist(),
#     val_df['label_emoji'].tolist()
# )

# # Multi-output BERT model
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

#     def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None,**kwargs):
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


# # Load pre-trained model
# print("Loading pre-trained model...")
# model_path = './models/fine_tuned_deberta_model'
# model = MultiOutputDeBERTa(
#     num_labels_type=len(label_encoder_type.classes_),
#     num_labels_emotion=len(label_encoder_emotion.classes_),
#     num_labels_emoji=len(label_encoder_emoji.classes_),
# )
# model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
# model.to(device)
# print("Model loaded.\n")

# # Initialize DataLoader
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# # Evaluate model and collect predictions
# print("Generating predictions...")
# model.eval()
# y_true = val_df['label_emotion'].tolist()
# y_pred = []

# with torch.no_grad():
#     for batch in tqdm(val_loader, desc="Evaluating"):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         logits_type, logits_emotion, logits_emoji = model(input_ids=input_ids, attention_mask=attention_mask)[:3]
        
#         pred_emotion = torch.argmax(logits_emotion, dim=1)
#         y_pred.extend(pred_emotion.cpu().numpy())

# # Print classification report for emotion labels
# # Generate classification reports for type, emotion, and emoji predictions
# print("Generating classification reports for all outputs...")

# # True labels
# y_true_type = val_df['label_type'].tolist()
# y_true_emotion = val_df['label_emotion'].tolist()
# y_true_emoji = val_df['label_emoji'].tolist()

# # Prediction lists
# y_pred_type = []
# y_pred_emotion = []
# y_pred_emoji = []

# # Make predictions for all labels
# for batch in tqdm(val_dataset, desc="Evaluating"):
#     batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items() if key != 'labels_type' and key != 'labels_emotion' and key != 'labels_emoji'}
#     with torch.no_grad():
#         logits_type, logits_emotion, logits_emoji = model(**batch)[:3]

#     pred_type = torch.argmax(logits_type, dim=1)
#     pred_emotion = torch.argmax(logits_emotion, dim=1)
#     pred_emoji = torch.argmax(logits_emoji, dim=1)

#     y_pred_type.extend(pred_type.cpu().numpy())
#     y_pred_emotion.extend(pred_emotion.cpu().numpy())
#     y_pred_emoji.extend(pred_emoji.cpu().numpy())

# # Generate classification reports
# print("\nClassification Report for MBTI Type:")
# print(classification_report(
#     y_true_type,
#     y_pred_type,
#     target_names=label_encoder_type.classes_,
#     labels=np.unique(y_true_type)
# ))

# print("\nClassification Report for Emotions:")
# print(classification_report(
#     y_true_emotion,
#     y_pred_emotion,
#     target_names=label_encoder_emotion.classes_,
#     labels=np.unique(y_true_emotion)
# ))

# print("\nClassification Report for Emojis:")
# print(classification_report(
#     y_true_emoji,
#     y_pred_emoji,
#     target_names=label_encoder_emoji.classes_,
#     labels=np.unique(y_true_emoji)
# ))

# # Analyze and conclude relationships between MBTI types, emotions, and emojis
# print("\nAnalyzing relationships between MBTI types and predictions...")

# # Create a DataFrame to aggregate predictions and true labels
# results_df = pd.DataFrame({
#     'true_type': label_encoder_type.inverse_transform(y_true_type),
#     'pred_type': label_encoder_type.inverse_transform(y_pred_type),
#     'true_emotion': label_encoder_emotion.inverse_transform(y_true_emotion),
#     'pred_emotion': label_encoder_emotion.inverse_transform(y_pred_emotion),
#     'true_emoji': label_encoder_emoji.inverse_transform(y_true_emoji),
#     'pred_emoji': label_encoder_emoji.inverse_transform(y_pred_emoji)
# })

# # Group by predicted MBTI types and analyze the corresponding emotions and emojis
# type_emotion_relation = results_df.groupby('pred_type')['pred_emotion'].value_counts(normalize=True).unstack()
# type_emoji_relation = results_df.groupby('pred_type')['pred_emoji'].value_counts(normalize=True).unstack()

# # Print top relationships for analysis
# print("\nTop predicted emotions associated with each MBTI type:")
# print(type_emotion_relation)

# print("\nTop predicted emojis associated with each MBTI type:")
# print(type_emoji_relation)


# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report
# import numpy as np
# import pandas as pd

# # Function to plot classification report as a bar chart
# def plot_classification_report(report, title="Classification Report"):
#     report_df = pd.DataFrame(report).transpose()
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=report_df.index, y=report_df['f1-score'], palette="viridis")
#     plt.title(f"{title} - F1 Score")
#     plt.xlabel('Classes')
#     plt.ylabel('F1-Score')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# # Generate and plot classification report for MBTI type, emotions, and emojis
# def generate_and_plot_classification_report(y_true, y_pred, labels, title):
#     report = classification_report(
#         y_true,
#         y_pred,
#         target_names=labels,
#         output_dict=True
#     )
#     plot_classification_report(report, title)

# # Generate classification reports and plot for each category
# generate_and_plot_classification_report(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type Classification Report")
# generate_and_plot_classification_report(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion Classification Report")
# generate_and_plot_classification_report(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji Classification Report")

# def plot_pairplot(df, title="Pairplot of Features"):
#     sns.pairplot(df, hue="pred_emotion", palette="viridis")
#     plt.suptitle(title, size=16)
#     plt.tight_layout()
#     plt.show()

# # Assuming 'results_df' contains multiple features (e.g., 'pred_emotion', 'pred_emoji', etc.)
# plot_pairplot(results_df)



# def plot_correlation_heatmap(df, title="Correlation Heatmap"):
#     corr_matrix = df.corr()
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()

# # Assuming 'results_df' has continuous features like prediction scores
# plot_correlation_heatmap(results_df)
# def plot_class_distribution(y_true, y_pred, labels, title="Class Distribution Comparison"):
#     true_counts = pd.Series(y_true).value_counts()
#     pred_counts = pd.Series(y_pred).value_counts()

#     df = pd.DataFrame({
#         'True': true_counts,
#         'Predicted': pred_counts
#     }).reindex(labels, fill_value=0)

#     df.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
#     plt.title(title)
#     plt.xlabel('Classes')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# # Plot class distribution comparison for MBTI, emotion, and emoji
# plot_class_distribution(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type Class Distribution")
# plot_class_distribution(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion Class Distribution")
# plot_class_distribution(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji Class Distribution")



# def plot_relation_heatmap(df, title="Relation Heatmap"):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(df, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
#     plt.title(title)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.tight_layout()
#     plt.show()

# # Plot relationships between MBTI type and emotions, and between MBTI type and emojis
# plot_relation_heatmap(type_emotion_relation, title="MBTI Type to Emotion Relationship")
# plot_relation_heatmap(type_emoji_relation, title="MBTI Type to Emoji Relationship")

# # Function to visualize the most common predictions (Top 10)
# def plot_most_common_predictions(df, title="Most Common Predictions"):
#     # Top 10 predicted emotions and emojis
#     top_emotions = df['pred_emotion'].value_counts().head(10)
#     top_emojis = df['pred_emoji'].value_counts().head(10)

#     # Plotting
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#     sns.barplot(x=top_emotions.index, y=top_emotions.values, ax=axes[0], palette='Blues')
#     axes[0].set_title("Top 10 Predicted Emotions")
#     axes[0].set_xlabel('Emotions')
#     axes[0].set_ylabel('Count')

#     sns.barplot(x=top_emojis.index, y=top_emojis.values, ax=axes[1], palette='Greens')
#     axes[1].set_title("Top 10 Predicted Emojis")
#     axes[1].set_xlabel('Emojis')
#     axes[1].set_ylabel('Count')

#     plt.tight_layout()
#     plt.show()

# # Plot top 10 most common emotions and emojis
# plot_most_common_predictions(results_df)











































# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, DistilBertModel, AutoModel
# from sklearn.metrics import classification_report
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import numpy as np
# from torch import nn
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the dataset
# print("Loading dataset...")
# df = pd.read_csv('dataset(cleaned too)/cleaned_balanced_mbti_with_emotions_and_emojis.csv')
# print("Dataset loaded successfully.\n")

# # Encode labels
# print("Encoding labels...")
# label_encoder_type = LabelEncoder()
# df['label_type'] = label_encoder_type.fit_transform(df['type'])

# label_encoder_emotion = LabelEncoder()
# df['label_emotion'] = label_encoder_emotion.fit_transform(df['emotion'])

# label_encoder_emoji = LabelEncoder()
# df['label_emoji'] = label_encoder_emoji.fit_transform(df['emoji'])
# print("Encoding complete.\n")

# # Split dataset
# print("Splitting dataset...")
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# print("Dataset split complete.\n")

# # Load tokenizer
# # Corrected Tokenizer Setup
# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# # Set the pad_token to eos_token if not already defined
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as the padding token
# print("Tokenizer loaded.\n")

# # Tokenization function
# def tokenize_function(posts):
#     print("Tokenizing data...")
#     return tokenizer(posts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# # Tokenize validation dataset
# val_encodings = tokenize_function(val_df['cleaned_posts'].tolist())
# print("Tokenization complete.\n")


# # Custom dataset class
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

# # Create validation dataset object
# val_dataset = MultiOutputMBTIDataset(
#     val_encodings,
#     val_df['label_type'].tolist(),
#     val_df['label_emotion'].tolist(),
#     val_df['label_emoji'].tolist()
# )

# # Multi-output DistilGPT model
# class MultiOutputDistilGPT(nn.Module):
#     def __init__(self, num_labels_type, num_labels_emotion, num_labels_emoji):
#         super(MultiOutputDistilGPT, self).__init__()
#         self.distilgpt = AutoModel.from_pretrained('distilgpt2')
#         self.dropout = nn.Dropout(0.3)
#         self.classifier_type = nn.Linear(self.distilgpt.config.hidden_size, num_labels_type)
#         self.classifier_emotion = nn.Linear(self.distilgpt.config.hidden_size, num_labels_emotion)
#         self.classifier_emoji = nn.Linear(self.distilgpt.config.hidden_size, num_labels_emoji)

#     def forward(self, input_ids, attention_mask, labels_type=None, labels_emotion=None, labels_emoji=None, **kwargs):
#         # Pass input IDs and attention mask to the DistilGPT model
#         outputs = self.distilgpt(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = outputs.last_hidden_state
#         pooled_output = self.dropout(last_hidden_state[:, -1, :])  # Use the last hidden state for classification
        
#         logits_type = self.classifier_type(pooled_output)
#         logits_emotion = self.classifier_emotion(pooled_output)
#         logits_emoji = self.classifier_emoji(pooled_output)

#         loss = None
#         if labels_type is not None and labels_emotion is not None and labels_emoji is not None:
#             loss_fct = nn.CrossEntropyLoss()  # Standard cross-entropy loss
#             loss_type = loss_fct(logits_type, labels_type)
#             loss_emotion = loss_fct(logits_emotion, labels_emotion)
#             loss_emoji = loss_fct(logits_emoji, labels_emoji)
#             loss = loss_type + loss_emotion + loss_emoji

#         return (loss, logits_type, logits_emotion, logits_emoji) if loss is not None else (logits_type, logits_emotion, logits_emoji)

# # Ensure that the model's configuration matches the tokenizer's padding
# print("Loading pre-trained model...")
# model_path = './models/fine_tuned_distillgpt2_cleaned_model'
# model = MultiOutputDistilGPT(
#     num_labels_type=len(label_encoder_type.classes_),
#     num_labels_emotion=len(label_encoder_emotion.classes_),
#     num_labels_emoji=len(label_encoder_emoji.classes_),
# )

# # Load state dictionary with strict set to False to allow for flexible loading
# model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device), strict=False)
# model.to(device)
# print("Model loaded.\n")

# # The rest of the code for evaluation and visualization remains the same as in your original BERT implementation.


# # Initialize DataLoader for validation set
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# # Evaluate model and collect predictions
# model.eval()
# y_true_type = val_df['label_type'].tolist()
# y_true_emotion = val_df['label_emotion'].tolist()
# y_true_emoji = val_df['label_emoji'].tolist()

# y_pred_type = []
# y_pred_emotion = []
# y_pred_emoji = []

# with torch.no_grad():
#     for batch in tqdm(val_loader, desc="Evaluating"):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         logits_type, logits_emotion, logits_emoji = model(input_ids=input_ids, attention_mask=attention_mask)[:3]
#         y_pred_type.extend(torch.argmax(logits_type, dim=1).cpu().numpy())
#         y_pred_emotion.extend(torch.argmax(logits_emotion, dim=1).cpu().numpy())
#         y_pred_emoji.extend(torch.argmax(logits_emoji, dim=1).cpu().numpy())

# # Generate classification reports
# def generate_classification_report(y_true, y_pred, labels, title):
#     report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
#     print(f"\n{title} Classification Report:")
#     print(classification_report(y_true, y_pred, target_names=labels))

#     return report

# generate_classification_report(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type")
# generate_classification_report(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion")
# generate_classification_report(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji")
# import pandas as pd
# import numpy as np

# # Assuming y_pred_type, y_pred_emotion, y_pred_emoji contain the model predictions
# # and val_df['label_type'], val_df['label_emotion'], val_df['label_emoji'] contain the true labels

# # Convert predictions and true labels back to their string representations
# pred_types = label_encoder_type.inverse_transform(y_pred_type)
# pred_emotions = label_encoder_emotion.inverse_transform(y_pred_emotion)
# pred_emojis = label_encoder_emoji.inverse_transform(y_pred_emoji)

# # Create a DataFrame to group and analyze predictions
# pred_df = pd.DataFrame({
#     'pred_type': pred_types,
#     'pred_emotion': pred_emotions,
#     'pred_emoji': pred_emojis
# })

# # Get the top predicted emotions for each MBTI type
# top_emotions = pred_df.groupby('pred_type')['pred_emotion'].value_counts(normalize=True).unstack().fillna(0)
# print("Top predicted emotions associated with each MBTI type:")
# print(top_emotions)

# # Get the top predicted emojis for each MBTI type
# top_emojis = pred_df.groupby('pred_type')['pred_emoji'].value_counts(normalize=True).unstack().fillna(0)
# print("\nTop predicted emojis associated with each MBTI type:")
# print(top_emojis)

# # Visualize results
# def plot_classification_report(report, title):
#     df = pd.DataFrame(report).transpose()
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=df.index, y=df['f1-score'], palette="viridis")
#     plt.title(f"{title} - F1 Score")
#     plt.xlabel('Classes')
#     plt.ylabel('F1-Score')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# plot_classification_report(generate_classification_report(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type"), "MBTI Type")
# plot_classification_report(generate_classification_report(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion"), "Emotion")
# plot_classification_report(generate_classification_report(y_true_emoji, y_pred_emoji, label_encoder_emoji.classes_, "Emoji"), "Emoji")
