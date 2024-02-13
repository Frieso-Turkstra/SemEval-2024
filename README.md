# SemEval-2024
This repository is dedicated to Task 8: Multigenerator, Multidomain, and Multilingual Black-Box Machine-Generated Text Detection.
We participated in subtask A (both the mono- and multilingual track) and subtask B.

Additional files are stored as a back-up on the following public drive: https://drive.google.com/drive/folders/1nFSLDokaWRDR-0dez6jvRBTPk_6cZXvX?usp=drive_link

## Train model

### Overview
Train the transformer model from which the hidden layers will be extracted.
These hidden layers are combined with the features as input for the svm.

### Command

```sh
python code/train.py --subtask B --train_file_path data/subtaskB_train.jsonl --test_file_path data/subtaskB_dev.jsonl --model FacebookAI/xlm-roberta-base
```

- **--subtask**: The subtask (A_mono, A_multi, B).
- **--train_file_path**: Path to the train file.
- **--test_file_path**: Path to the test file.
- **--model**: Transformer model to train and test.

## Extract hidden layers

### Overview
Extract hidden layers from the fine-tuned model.
These hidden layers are combined with the features as input for the svm.

### Command

```sh
python code/extract_hidden_layers.py --subtask B --input_file_path data/subtaskB_dev.jsonl --model FacebookAI/xlm-roberta-base/subtaskB/0/best/ --num_layers 4
```

- **--subtask**: The subtask (A_mono, A_multi, B).
- **--input_file_path**: Path to the file for which to extract the hidden layers.
- **--model**: Model from which to extract the hidden layers.
- **--num_layers**: The number of last hidden layers to extract.

## Calculate perplexity scores

### Overview
Calculate the perplexity for each text in the data set.
Depending on your hardware, you may want to split large entries. This can be done by calling
```split_too_large_entries()``` from ```code/utils.py``` on your data set. We recommend a maximum token length of 10 000. Do not forget
to call ```merge_on_id()``` (also from ```code/utils.py```) afterwards to recombine the previously split entries. If memory issues persist, there is also functionality in ```code/utils.py``` for splitting the data set into smaller parts and merging them later on.

### Command

```sh
python code/perplexity.py --subtask B --input_file_path data/subtaskB_dev.jsonl --model openai-community/gpt2-xl  
```

- **--subtask**: The subtask (A_mono, A_multi, B).
- **--input_file_path**: Path to the file for which to calculate the perplexity scores.
- **--model**: The model from which the perplexity is calculated.

## Calculate other features

### Overview
Calculate the other features for each text in the data set. It is assumed you already have the perplexity scores so they can be merged into the normalized feature vector - which, combined with the hidden layers, serves as the input for the svm. 

### Command

```sh
python code/features.py --subtask B --input_file_path data/subtaskB_dev.jsonl --perplexity_file_path perplexities_data-subtaskB_dev.jsonl
```

- **--subtask**: The subtask (A_mono, A_multi, B).
- **--input_file_path**: Path to the file for which to calculate the features.
- **--perplexity_file_path**: Path to the file with the perplexity scores for the input file.

## Make predictions

### Overview
Train an svm on the hidden layers combined with the features.
Outputs the final predictions file.

### Command

```sh
python code/predict.py --subtask B --train_file_path data/subtaskB_train.jsonl --train_hidden_layers_file_path hidden_layers_data-subtaskB_train.jsonl --train_features_file_path features_data-subtaskB_train.jsonl --test_file_path data/subtaskB_dev.jsonl --test_hidden_layers_file_path hidden_layers_data-subtaskB_dev.jsonl --test_features_file_path features_data-subtaskB_dev.jsonl --num_layers 2 
```

- **--subtask**: The subtask (A_mono, A_multi, B).
- **--train_file_path**: Path to the train file.
- **--train_hidden_layers_file_path**: Path to the hidden layers train file.
- **--train_features_file_path**: Path to the features train file.
- **--test_file_path**: Path to the test file.
- **--test_hidden_layers_file_path**: Path to the hidden layers test file.
- **--test_features_file_path**: Path to the features test file.
- **--num_layers**: The number of last hidden layers to consider during prediction.
