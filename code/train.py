from sklearn.model_selection import train_test_split
from scipy.special import softmax
from datasets import Dataset
import argparse
import logging
import os
import pandas as pd
import evaluate
import numpy as np
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    IntervalStrategy,
    Trainer,
    EarlyStoppingCallback,
)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A_mono, A_multi, B).", type=str, choices=["A_mono", "A_multi", "B"])
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test.", type=str)
    parser.add_argument("--predictions_file_path", "-p", required=False, help="Path to the predictions file.", type=str)

    args = parser.parse_args()
    return args


def get_data(train_file_path, test_file_path, random_seed):
    """
    Read in data and split train_file_path into train and validation set.
    """

    train_df = pd.read_json(train_file_path, lines=True)
    test_df = pd.read_json(test_file_path, lines=True)
    
    train_df, valid_df = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=random_seed)

    return train_df, valid_df, test_df


def preprocess_function(examples, **fn_kwargs):
    tokenizer = fn_kwargs["tokenizer"]
    return tokenizer(examples["text"], truncation=True, max_length=512)


def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):
    """
    Fine-tune the model on train_df, save the best model for evaluation.
    """

    # Turn pandas dataframe into huggingface Dataset.
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # Get tokenizer and model from huggingface.
    tokenizer = AutoTokenizer.from_pretrained(model)     
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    
    # Tokenize data for train/valid.
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_valid_dataset = valid_dataset.map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Trainer. 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50, 
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.STEPS, 
        eval_steps=50,
        save_total_limit=2, 
        save_strategy=IntervalStrategy.STEPS,
        eval_accumulation_steps=1,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    # Save best model.
    best_model_path = checkpoints_path + "/best/"
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    trainer.save_model(best_model_path)


def test(test_df, model_path, id2label, label2id):
    
    # Load tokenizer from saved model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load best model.
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # Tokenize test data.     
    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test_dataset = test_dataset.map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Trainer.
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Get logits from predictions and evaluate results using classification report.
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # Return dictionary of classification report.
    return results, preds


if __name__ == "__main__":

    # Read in command line arguments.
    args = create_arg_parser()
    subtask = args.subtask
    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    model = args.model
    predictions_file_path = args.predictions_file_path

    # Check if files exist.
    logging.basicConfig(filename="logs.log", encoding="utf-8", level=logging.DEBUG)

    for file in (train_file_path, test_file_path):
        if not os.path.exists(file):
            logging.error(f"File doesn't exist: {file}")
            raise ValueError(f"File doesn't exist: {file}")

    # Get the id-labels depending on the subtask.
    if subtask == "B":
        id2label = {0: "human", 1: "chatGPT", 2: "cohere", 3: "davinci", 4: "bloomz", 5: "dolly"}
        label2id = {"human": 0, "chatGPT": 1,"cohere": 2, "davinci": 3, "bloomz": 4, "dolly": 5}
    else:
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}

    # Set random seed for reproducibility.
    random_seed = 0
    set_seed(random_seed)

    # Read data into train/valid/test dataframes. 
    print("Reading data...")
    train_df, valid_df, test_df = get_data(train_file_path, test_file_path, random_seed)

    # Fine-tune the model.
    print("Start fine-tuning...")
    fine_tune(train_df, valid_df, f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)

    # Test the model.
    print("Testing the fine-tuned model...")
    results, predictions = test(test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)
    
    # Save results.
    if not predictions_file_path:
       predictions_file_path = f"predictions_{subtask}.jsonl"

    logging.info(results)
    predictions_df = pd.DataFrame({"id": test_df["id"], "label": predictions})
    predictions_df.to_json(predictions_file_path, lines=True, orient="records")
    print("Successfully saved predictions!")
