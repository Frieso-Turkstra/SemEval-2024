from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
import argparse
import logging
import os
import pandas as pd
from torch import nn
from transformers import (
    set_seed,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    AutoModelForSequenceClassification,
    AutoConfig,
)


class CustomModel(nn.Module):
    def __init__(self, model_path, label2id, id2label, num_layers):
        super(CustomModel, self).__init__()

        # Load the model.
        self.num_labels = len(label2id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=self.num_labels, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True, output_attentions=True, output_hidden_states=True
        )

        # Extract information from the model's configuration file.
        config = AutoConfig.from_pretrained(model_path)
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        # Prepare hidden layer extraction.
        self.linear = nn.Linear(self.hidden_size, self.num_labels) 
        self.num_layers = num_layers
        self.hidden_layers = list()
        
        # Ensure that the model has enough layers.
        if self.num_layers > self.num_hidden_layers:
            raise ValueError(f'Cannot extract the last {self.num_layers} layers; the model only has {self.num_hidden_layers}')
    
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # A hidden state is of the shape (batch_size, max_length, hidden_size)
        batch_size = len(outputs.hidden_states[-1][:,0,:].view(-1, self.hidden_size))
        for i in range(batch_size): 
            hidden_states = list()
            for j in range(1, self.num_layers+1):
                hidden_state = outputs.hidden_states[-j][:,0,:].view(-1, self.hidden_size)
                hidden_states.append(hidden_state[i].tolist())
            self.hidden_layers.append(hidden_states)
        
        # The layers are already extracted at this point but `predict()` still 
        # expects a normal output so that is why this next part is here.
        logits = self.linear(hidden_state) 
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A_mono, A_multi, B).", type=str, choices=["A_mono", "A_multi", "B"])
    parser.add_argument("--input_file_path", "-i", required=True, help="Path to the file for which to extract the hidden layers.", type=str)
    parser.add_argument("--model", "-m", required=True, help="Model from which to extract the hidden layers.", type=str)
    parser.add_argument("--num_layers", "-n", required=True, help="The number of last hidden layers to extract", type=int)
    parser.add_argument("--output_file_path", "-o", required=False, help="Path to the file where the hidden layers are saved.", type=str)
    
    args = parser.parse_args()
    return args


def preprocess_function(examples, **fn_kwargs):
    tokenizer = fn_kwargs["tokenizer"]
    return tokenizer(examples["text"], truncation=True, max_length=512)


def extract_hidden_layers(input_file_path, num_layers, id2label, label2id, model_path):
    # Read the samples from the input file into a dataframe.
    data_df = pd.read_json(input_file_path, lines=True)

    # Load the tokenizer from saved model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model with a custom forward method which extracts the last N layers
    model = CustomModel(model_path, label2id, id2label, num_layers)
    
    # Load and tokenize the data.
    dataset = Dataset.from_pandas(data_df)
    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Trainer.
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Call the forward function once for each sample in the dataset.
    trainer.predict(tokenized_dataset)

    # Return the extracted hidden layers.
    return model.hidden_layers


if __name__ == "__main__":

    # Read in the command line arguments.
    args = create_arg_parser()
    subtask = args.subtask
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    model = args.model
    num_layers = args.num_layers

    # Check if input file exists.
    logging.basicConfig(filename="logs.log", encoding="utf-8", level=logging.DEBUG)

    if not os.path.exists(input_file_path):
        logging.error(f"File doesn't exist: {input_file_path}")
        raise ValueError(f"File doesn't exist: {input_file_path}")

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

    # Extract the last N hidden layers for each sample in the input file.
    hidden_layers = extract_hidden_layers(input_file_path, num_layers, id2label, label2id, model)
    hidden_layers_df = pd.DataFrame(hidden_layers)
    hidden_layers_df.to_json(f"hidden_layers_{subtask}.jsonl", lines=True, orient="records")
