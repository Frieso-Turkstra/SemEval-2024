#!/bin/sh

subtask="A_mono"                                        
train_file_path="subtaskA_train_monolingual.jsonl"      
test_file_path="subtaskA_dev_monolingual.jsonl"  

python model.py -sb $subtask -tr $train_file_path -t $test_file_path 
python perplexity.py -sb $subtask -i $train_file_path 
python features.py -sb $subtask -i $train_file_path -p "perplexities_${subtask}.jsonl"
python lstm.py -h "hidden_states_${subtask}.jsonl" -f "features_${subtask}.jsonl" 
# outputs a file called 'predictions_${subtask}.jsonl' with the predictions
