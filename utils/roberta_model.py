import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm # Using standard tqdm
import numpy as np

def tokenize_data(df_data, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    labels = df_data['target'].values

    for text in df_data['cleaned_text']:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_len,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return TensorDataset(input_ids, attention_masks, labels)

def train_roberta_model(train_df, dev_df, num_labels, max_len, batch_size, epochs, learning_rate, epsilon, device):
    print("Setting up RoBERTa model and tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', num_labels=num_labels, ignore_mismatched_sizes=True)
    model.to(device)

    train_dataset = tokenize_data(train_df, tokenizer, max_len)
    dev_dataset = tokenize_data(dev_df, tokenizer, max_len)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("Starting RoBERTa training...")
    for epoch_i in range(epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
        print("Training...")

        total_train_loss = 0
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training Batch")):
            try:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            except RuntimeError as e:
                print(f"\nRuntimeError during training batch {step}: {e}")
                print("This might be due to memory constraints or data issues. Consider reducing BATCH_SIZE.")
                return None, None
            except Exception as e:
                print(f"\nAn unexpected error occurred during training batch {step}: {e}")
                return None, None


        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.2f}")

        print("\nRunning RoBERTa Validation...")
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in tqdm(dev_dataloader, desc="Validation Batch"):
            try:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                batch_accuracy = accuracy_score(label_ids, np.argmax(logits, axis=1))
                total_eval_accuracy += batch_accuracy
            except RuntimeError as e:
                print(f"\nRuntimeError during validation batch: {e}")
                print("This might be due to memory constraints or data issues.")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred during validation batch: {e}")
                break


        avg_val_loss = total_eval_loss / len(dev_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation Accuracy: {avg_val_accuracy:.2f}")

    print("\nRoBERTa Training complete!")
    return model, tokenizer

def get_roberta_predictions(model, tokenizer, test_df, max_len, batch_size, device):
    test_dataset = tokenize_data(test_df, tokenizer, max_len)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    print("\nPerforming RoBERTa predictions on Test Set...")
    model.eval()

    predictions = []
    confidences = []
    true_labels = []

    for batch in tqdm(test_dataloader, desc="RoBERTa Test Prediction"):
        try:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            predicted_labels = np.argmax(probabilities, axis=1).flatten()
            batch_confidences = np.max(probabilities, axis=1).flatten()

            predictions.extend(predicted_labels)
            confidences.extend(batch_confidences)
            true_labels.extend(b_labels.to('cpu').numpy().flatten())
        except RuntimeError as e:
            print(f"\nRuntimeError during prediction batch: {e}")
            print("This might be due to memory constraints or data issues.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred during prediction batch: {e}")
            break

    return predictions, confidences, true_labels