import pandas as pd
import torch
import sys, os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import evaluate
import numpy as np
import sys, os
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = r"d:\Capstone\indobert_nlp"
sys.path.append(current_dir)
from nlp_config import *

def train_indobert():
    print("Memulai training IndoBERT...")
    
    if not INTENT_DATA_PATH.exists():
        print("Data training belum dibuat! Silakan run step 2 dulu.")
        return
        
    # 1. Load Data
    df = pd.read_csv(INTENT_DATA_PATH)
    
    # 2. Encode Labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['intent'])
    
    # Simpan mapping label
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    num_labels = len(label_mapping)
    print(f"Ditemukan {num_labels} intent: {label_mapping}")
    
    # 3. Split Dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )
    
    # 4. Initialize Tokenizer & Model
    print(f"Loading model: {INDOBERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        INDOBERT_MODEL_NAME, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # Simpan mapping label ke config model agar saat load mudah
    model.config.id2label = {int(id): label for label, id in label_mapping.items()}
    model.config.label2id = {label: int(id) for label, id in label_mapping.items()}
    
    # 5. Tokenization
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # 6. Setup Metrics
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",  # eval_strategy is used in recent transformers
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 8. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Mulai proses training (ini mungkin memakan waktu)...")
    trainer.train()
    
    # 9. Save Model
    print(f"Menyimpan model ke {MODEL_OUTPUT_DIR}...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    print("Selesai!")

if __name__ == "__main__":
    # Karena evaluate module bisa download script, pastikan koneksi internet ada
    train_indobert()
