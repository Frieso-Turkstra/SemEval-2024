from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import EarlyStoppingCallback, IntervalStrategy
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
from torch import nn
import json

# python transformer_custom.py -df -tr data/subtaskA_train_monolingual.jsonl -t data/subtaskA_dev_monolingual.jsonl -sb B -m "custom-model" -p data/predictions_custom_nodf.jsonl

class CustomModel(nn.Module):
    def __init__(self, model_path, label2id, id2label):
        super(CustomModel, self).__init__()
        self.num_labels = len(label2id)
        self.hidden_states = list()

        #Load Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=self.num_labels, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True, output_attentions=True, output_hidden_states=True
        )

        # 768 is size of hidden_state
        self.linear = nn.Linear(768, self.num_labels) # load and initialize weights

    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # the last hidden state will be of the shape (batch_size=16, max_length=x, hidden_size=768)


        for i in range(8):
            hidden_states = list()
            for j in range(13):
                hidden_state = outputs.hidden_states[j][:,0,:].view(-1, 768)
                hidden_states.append(hidden_state[i].tolist())
            self.hidden_states.append(hidden_states)
        
        logits = self.linear(hidden_state) # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df


def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     
    #model = AutoModelForSequenceClassification.from_pretrained(
    #   model, num_labels=len(label2id), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    #)
    model = CustomModel(model, label2id, id2label)
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1, 
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.STEPS, # "steps", was epoch
        eval_steps=50, # evaluate and save every 50 steps
        save_total_limit=2, # only last 2 models are saved
        save_strategy=IntervalStrategy.STEPS, # was epoch
        eval_accumulation_steps=1,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
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

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    trainer.save_model(best_model_path)


def test(test_df, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    #model = AutoModelForSequenceClassification.from_pretrained(
    #   model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    #)
    model = CustomModel(model_path, label2id, id2label)
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)

    # print(f"Number of hidden_states collected: {len(model.hidden_states)}, should be equal to number of input sentences.")
    # is equal to 5000 / 8 = 625? 
    # print(f"Length of a hidden state: {len(model.hidden_states[0][0])}, should be equal to 768.")
    # what is currently saved as a hidden state is actually a batch of 8 hidden states

    # save hidden states to a file
    with open('hidden_states.json', 'w') as f:
        json.dump(model.hidden_states, f, indent=4)
    
    # return dictionary of classification report
    return results, preds


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
    parser.add_argument("--disable_finetuning", "-df", action="store_true", help="Disable finetuning")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = create_arg_parser()
    
    random_seed = 0
    train_path = args.train_file_path # For example 'subtaskA_train_multilingual.jsonl'
    test_path = args.test_file_path # For example 'subtaskA_test_multilingual.jsonl'
    subtask = args.subtask # For example 'A'
    model = args.model # For example 'xlm-roberta-base'
    prediction_path = args.prediction_file_path # For example 'subtaskB_predictions.jsonl'

    logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.DEBUG)

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

    # get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)
    
    if not args.disable_finetuning:
        # train detector model, this step can be skipped when testing already finetuned models
        fine_tune(train_df, valid_df, f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)

    # test detector model
    results, predictions = test(test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)
    # Output the results
    print(results)
    logging.info(results)

    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')
