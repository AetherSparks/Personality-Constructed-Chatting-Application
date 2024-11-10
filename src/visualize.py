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




# from sklearn.metrics import roc_curve, auc

# def plot_roc_curve(y_true, y_pred, title="ROC Curve"):
#     fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
#     roc_auc = auc(fpr, tpr)

#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(title)
#     plt.legend(loc='lower right')
#     plt.tight_layout()
#     plt.show()

# # For binary classification, plot the ROC curve
# plot_roc_curve(y_true_emotion, y_pred_emotion, "Emotion ROC Curve")


# def plot_distribution(df, title="Prediction Distribution"):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df['pred_emotion'], kde=True, color='blue', label='Predicted Emotion')
#     plt.title(title)
#     plt.xlabel('Predicted Emotion')
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.show()

# plot_distribution(results_df, "Prediction Distribution of Emotions")
# def plot_stacked_bar(y_true, y_pred, labels, title="Stacked Bar Chart of Confusion"):
#     confusion_matrix_df = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#     confusion_matrix_df = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0)

#     confusion_matrix_df.drop('All', axis=1).plot(kind='bar', stacked=True, figsize=(10, 6), color=sns.color_palette("Blues", len(labels)))
#     plt.title(title)
#     plt.xlabel('True Labels')
#     plt.ylabel('Proportion')
#     plt.tight_layout()
#     plt.show()

# plot_stacked_bar(y_true_emotion, y_pred_emotion, label_encoder_emotion.classes_, "Emotion Prediction Stacked Bar")

# def plot_error_heatmap(y_true, y_pred, labels, title="Error Heatmap"):
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     cm_error = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
#     cm_error = np.subtract(1, cm_error)  # Invert the matrix to highlight errors

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm_error, annot=True, fmt='.2f', cmap='Reds', xticklabels=labels, yticklabels=labels)
#     plt.title(title)
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.tight_layout()
#     plt.show()

# plot_error_heatmap(y_true_type, y_pred_type, label_encoder_type.classes_, "MBTI Type Error Heatmap")

# # Function to plot relationships between predicted and true labels (Heatmap)
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


