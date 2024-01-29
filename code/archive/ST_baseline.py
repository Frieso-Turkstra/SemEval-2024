from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_device_train_batch", "-pdtb", required=True, help="Batch size per device during training.", type=int, default=8)
    parser.add_argument("--per_device_eval_batch", "-pdeb", required=True, help="Batch size per device during evaluation.", type=int, default=8)
    parser.add_argument("--num_train_epochs", "-nte", required=True, help="Number of training epochs.", type=int, default=3)
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
    return parser.parse_args()

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)

def get_data(train_path, test_path, random_seed):
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)
    return train_df, val_df, test_df

def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))
    return results

def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model, fp16=True, fp16_full_eval=True, gradient_accumulation_steps=4):
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=len(label2id), id2label=id2label, label2id=label2id)

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        save_total_limit=2,
        per_device_train_batch_size=args.per_device_train_batch,
        per_device_eval_batch_size=args.per_device_eval_batch,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=fp16,
        fp16_full_eval=fp16_full_eval,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_model_path = checkpoints_path+'/best/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    trainer.save_model(best_model_path)

def test(test_df, model_path, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id)
            
    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    return results, preds

if __name__ == '__main__':

    args = create_arg_parser()

    random_seed = 0
    train_path = args.train_file_path
    test_path = args.test_file_path
    model = args.model
    subtask = args.subtask
    prediction_path = args.prediction_file_path

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    fine_tune(train_df, valid_df, f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model, fp16=True, fp16_full_eval=True, gradient_accumulation_steps=4)

    results, predictions = test(test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)
    
    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')