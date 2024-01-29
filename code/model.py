from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch import nn
import argparse
import logging
import os
import evaluate
import pandas as pd
import numpy as np
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
    EarlyStoppingCallback,
    AutoConfig,
)


class CustomModel(nn.Module):
    def __init__(self, model_path, label2id, id2label, N):
        super(CustomModel, self).__init__()
        self.num_labels = len(label2id)
        self.hidden_states = list()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=self.num_labels, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True, output_attentions=True, output_hidden_states=True
        )
        self.config = AutoConfig.from_pretrained(model_path)
        self.linear = nn.Linear(self.config.hidden_size, self.num_labels) 
        self.N = N
        
        if self.N > self.config.num_hidden_layers:
            raise ValueError(f'Cannot extract the last {self.N} layers; the model only has {self.config.num_hidden_layers}')
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # A hidden state is of the shape (batch_size, max_length, hidden_size)
        batch_size = len(outputs.hidden_states[-1][:,0,:].view(-1, self.config.hidden_size))
        print('Batch_size = {}'.format(batch_size))
        for i in range(batch_size): 
            hidden_states = list()
            for j in range(1, self.N+1):
                hidden_state = outputs.hidden_states[-j][:,0,:].view(-1, self.config.hidden_size)
                hidden_states.append(hidden_state[i].tolist())
            self.hidden_states.append(hidden_states)
        
        # this part is necessary to run predict()
        logits = self.linear(hidden_state) # calculate losses
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


def preprocess_function(examples, **fn_kwargs):
    tokenizer = fn_kwargs['tokenizer']
    return tokenizer(examples['text'], truncation=True, max_length=512)


def compute_metrics(eval_pred):

    f1_metric = evaluate.load('f1')

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels, average='micro'))

    return results


def extract_hidden_states(N, data_df, output_file, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load model with a custom forward method, extract the last N layers
    model = CustomModel(model_path, label2id, id2label, N)
    
    # load and tokenize data
    dataset = Dataset.from_pandas(data_df)
    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer}
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # calls forward function on each sample in dataset
    trainer.predict(tokenized_dataset)

    # save hidden states to a file
    df = pd.DataFrame(model.hidden_states)
    df.to_json(output_file, lines=True, orient='records')


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer}
    )
    tokenized_valid_dataset = valid_dataset.map(
        preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer}
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1, 
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.STEPS, 
        eval_steps=50,
        save_total_limit=2, 
        save_strategy=IntervalStrategy.STEPS,
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


def get_data(train_path, test_path, random_seed):
    """
    Read in data and split train_path into train and validation set
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, valid_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, valid_df, test_df


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file_path', '-tr', required=True, help='Path to the train file.', type=str)
    parser.add_argument('--test_file_path', '-t', required=True, help='Path to the test file.', type=str)

    # specify the model or automatically select best model depending on subtask
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subtask', '-sb', help='Subtask (A_mono, A_multi or B)', type=str, choices=['A_mono', 'A_multi', 'B'])
    group.add_argument('--model', '-m', help='Transformer to train and test', type=str)
    
    parser.add_argument('--num_extracted_layers', '-n', required=False, help='The number of last hidden layers to extract', type=int, default=4)
    parser.add_argument('--disable_finetuning', '-df', required=False, help='Disable finetuning', action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_arg_parser()

    random_seed = 0
    train_path = args.train_file_path
    test_path = args.test_file_path
    subtask = args.subtask
    # Select a model
    models = {
        'A_mono': 'bert-base-cased',
        'A_multi': 'intfloat/multilingual-e5-large',
        'B': 'FacebookAI/xlm-roberta-base',
        None: args.model
    }
    model = models[subtask]

    logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.DEBUG)

    if not os.path.exists(train_path):
        logging.error('File doesn\'t exist: {}'.format(train_path))
        raise ValueError('File doesn\'t exist: {}'.format(train_path))

    if not os.path.exists(test_path):
        logging.error('File doesn\'t exist: {}'.format(test_path))
        raise ValueError('File doesn\'t exist: {}'.format(test_path))

    if subtask == 'A_mono' or subtask == 'A_multi':
        id2label = {0: 'human', 1: 'machine'}
        label2id = {'human': 0, 'machine': 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error('Wrong subtask: {}. It should be A_mono, A_multi or B'.format(subtask))
        raise ValueError('Wrong subtask: {}. It should be A_mono, A_multi or B'.format(subtask))
    
    set_seed(random_seed)

    # get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    # skip finetuning with the -df flag (if you load an already finetuned model)
    if not args.disable_finetuning:
        fine_tune(train_df, valid_df, f'{model}/subtask{subtask}/{random_seed}', id2label, label2id, model)

    # extract the last N hidden layers for each sample
    trained_model = f'{model}/subtask{subtask}/{random_seed}/best'
    output_file = f'hidden_states_{subtask}.jsonl'
    extract_hidden_states(args.num_extracted_layers, train_df, output_file, trained_model, id2label, label2id)