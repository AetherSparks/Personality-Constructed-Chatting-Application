from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import logging
from accelerate import init_empty_weights, dispatch_model
from transformers import AutoConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model configuration and initialization
model_name = "EleutherAI/gpt-neo-1.3B"  # Change to a smaller model if needed
config = AutoConfig.from_pretrained(model_name)

# Initialize model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
model.tie_weights()

# Dispatch model across available devices (CPU, GPU, etc.)
model = dispatch_model(model, device_map="auto")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load and preprocess the dataset
train_file = 'train_data.txt'

def preprocess_function(examples):
    logging.info("Tokenizing data...")
    try:
        tokenized_data = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    except Exception as e:
        logging.error("Tokenization error: %s", e)
        raise
    logging.info("Tokenization complete.")
    return tokenized_data

# Create a dataset
try:
    dataset = load_dataset('text', data_files={'train': train_file}, cache_dir=None)
    logging.info("Dataset loaded. Length: %d", len(dataset['train']))
except Exception as e:
    logging.error("Dataset loading error: %s", e)
    raise

# Preprocess the dataset
try:
    dataset = dataset.map(preprocess_function, batched=True)
    logging.info("Dataset processing complete.")
except Exception as e:
    logging.error("Dataset processing error: %s", e)
    raise

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps=50,  # Evaluate more frequently to monitor progress
    save_steps=500,  # Save model checkpoints more frequently
    save_total_limit=3,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size to lower memory usage
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch sizes
    warmup_steps=500,  # Adjusted warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    fp16=True,  # Enable mixed precision to reduce memory usage
    dataloader_num_workers=4,  # Adjust for parallel data loading
    disable_tqdm=False,  # Enable progress bar for visibility
    report_to="none"  # Disable integration with other platforms
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train']
)

# Train the model
trainer.train()
