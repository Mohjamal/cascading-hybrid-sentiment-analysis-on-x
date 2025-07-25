import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from tqdm import tqdm # Using standard tqdm

# Import functions from our utility modules
from utils.preprocessing import preprocess_tweet
from utils.roberta_model import train_roberta_model, get_roberta_predictions
from utils.vader_model import get_vader_compound_score, get_vader_binary_prediction

# --- Configuration ---
# IMPORTANT: Update this path based on your repository structure.
DATASET_PATH = 'data/sentiment140-dataset-800records.csv'

NUM_LABELS = 2 # Positive (1) and Negative (0)
MAX_LEN = 128
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 2e-5
EPSILON = 1e-8

# Hybrid Model Specific Thresholds
# RoBERTa confidence below this threshold might trigger VADER's filter
ROBERTA_LOW_CONFIDENCE_THRESHOLD = 0.60

# VADER compound score thresholds for binary classification
VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05


# Set device for GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. Data Collection & Initial Loading ---
print("Loading data...")
try:
    df = pd.read_csv(DATASET_PATH, encoding='latin-1', header=None,
                     names=['target', 'ids', 'date', 'flag', 'user', 'text'])
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}. Please ensure the file is in the 'data/' directory or the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# Retain only relevant columns: text and target
df = df[['text', 'target']]

# Sentiment Label Mapping: 4 (positive) -> 1, 0 (negative) remains 0
df['target'] = df['target'].replace(4, 1)

print(f"Initial dataset shape: {df.shape}")
print(f"Sentiment distribution:\n{df['target'].value_counts()}")

# --- 2. Data Preprocessing ---
print("\nPreprocessing data...")
df['cleaned_text'] = df['text'].apply(preprocess_tweet)
print("Preprocessing complete.")

# --- 2b. Dataset Splitting (60/20/20) ---
print("\nSplitting dataset into training, development, and testing sets (60/20/20)...")
train_dev_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['target']
)
train_df, dev_df = train_test_split(
    train_dev_df, test_size=0.25, random_state=42, stratify=train_dev_df['target']
)
print(f"Train set size: {len(train_df)}")
print(f"Development set size: {len(dev_df)}")
print(f"Test set size: {len(test_df)}")


# --- 3. RoBERTa Model Component (Training and Prediction on Test Set) ---
print("\n--- RoBERTa Model Training and Prediction ---")
trained_model, tokenizer = train_roberta_model(
    train_df, dev_df, NUM_LABELS, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, EPSILON, device
)

# Check if training failed (if train_roberta_model returned None, None)
if trained_model is None or tokenizer is None:
    print("RoBERTa model training failed. Skipping predictions and further evaluation.")
    roberta_predictions = []
    roberta_confidences = []
    roberta_true_labels = []
    accuracy_roberta = 0.0 # Default values for failed training
    f1_roberta = 0.0
else:
    roberta_predictions, roberta_confidences, roberta_true_labels = get_roberta_predictions(
        trained_model, tokenizer, test_df, MAX_LEN, BATCH_SIZE, device
    )

    # --- Evaluation for Standalone RoBERTa ---
    print("\n--- Standalone RoBERTa Model Performance on Test Set ---")
    accuracy_roberta = accuracy_score(roberta_true_labels, roberta_predictions)
    precision_roberta, recall_roberta, f1_roberta, _ = precision_recall_fscore_support(roberta_true_labels, roberta_predictions, average='binary', pos_label=1)

    print(f"Accuracy: {accuracy_roberta:.4f}")
    print(f"Precision: {precision_roberta:.4f}")
    print(f"Recall: {recall_roberta:.4f}")
    print(f"F1-score: {f1_roberta:.4f}")

# --- 4. VADER Model Component (Scoring on Test Set) ---
print("\n--- VADER Sentiment Scoring on Test Set ---")
test_df['vader_compound_score'] = test_df['cleaned_text'].apply(get_vader_compound_score)

vader_standalone_predictions = test_df['vader_compound_score'].apply(
    lambda x: get_vader_binary_prediction(x, VADER_POSITIVE_THRESHOLD, VADER_NEGATIVE_THRESHOLD)
)
vader_true_labels = test_df['target'].values # True labels are the same for all models

# --- Evaluation for Standalone VADER ---
print("\n--- Standalone VADER Model Performance on Test Set ---")
accuracy_vader = accuracy_score(vader_true_labels, vader_standalone_predictions)
precision_vader, recall_vader, f1_vader, _ = precision_recall_fscore_support(vader_true_labels, vader_standalone_predictions, average='binary', pos_label=1)

print(f"Accuracy: {accuracy_vader:.4f}")
print(f"Precision: {precision_vader:.4f}")
print(f"Recall: {recall_vader:.4f}")
print(f"F1-score: {f1_vader:.4f}")


# --- 5. Cascading Hybrid Prediction Refinement Logic ---
print("\n--- Applying Hybrid Prediction Refinement Logic ---")

hybrid_predictions = []

# Only proceed if RoBERTa model training was successful
if trained_model is not None and tokenizer is not None:
    for i in tqdm(range(len(test_df)), desc="Applying Hybrid Logic"):
        roberta_pred = roberta_predictions[i]
        roberta_conf = roberta_confidences[i]
        vader_compound = test_df['vader_compound_score'].iloc[i]

        # Determine VADER's binary polarity (1 for positive, 0 for negative, -1 for neutral)
        vader_polarity = -1 # Default to neutral
        if vader_compound >= VADER_POSITIVE_THRESHOLD:
            vader_polarity = 1
        elif vader_compound <= VADER_NEGATIVE_THRESHOLD:
            vader_polarity = 0

        final_pred = roberta_pred # Start with RoBERTa's prediction

        # Apply refinement logic: VADER intervenes if RoBERTa is low confidence AND VADER contradicts
        if roberta_conf < ROBERTA_LOW_CONFIDENCE_THRESHOLD:
            # If VADER has a strong, non-neutral opinion AND it contradicts RoBERTa
            if vader_polarity != -1 and vader_polarity != roberta_pred:
                final_pred = vader_polarity # VADER overrides

        hybrid_predictions.append(final_pred)
else:
    print("Skipping hybrid prediction refinement as RoBERTa predictions are not available (training failed).")


# --- 6. Evaluation for Hybrid Model ---
print("\n--- Hybrid Model Performance on Test Set ---")
hybrid_true_labels = test_df['target'].values

# Only proceed if hybrid_predictions were generated
if hybrid_predictions:
    accuracy_hybrid = accuracy_score(hybrid_true_labels, hybrid_predictions)
    precision_hybrid, recall_hybrid, f1_hybrid, _ = precision_recall_fscore_support(hybrid_true_labels, hybrid_predictions, average='binary', pos_label=1)

    print(f"Accuracy: {accuracy_hybrid:.4f}")
    print(f"Precision: {precision_hybrid:.4f}")
    print(f"Recall: {recall_hybrid:.4f}")
    print(f"F1-score: {f1_hybrid:.4f}")

else:
    print("Skipping hybrid model evaluation as hybrid predictions were not generated.")
