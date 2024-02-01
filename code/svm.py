from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import argparse

# THESE COMMANDS ARE FOR TESTING, DO NOT USE THEM
# python svm.py -tr ..\data\subtaskA_dev_monolingual.jsonl -t ..\data\subtaskA_dev_monolingual.jsonl -trh hidden_states_A_mono_test.jsonl -th hidden_states_A_mono_test.jsonl -trf features_A_mono_dev.jsonl -tf features_A_mono_dev.jsonl 
# python svm.py -tr subtaskB_train.jsonl -t subtaskB_dev.jsonl -trh hidden_states_B_train.jsonl -th hidden_states_B_test.jsonl -trf features_B_train.jsonl -tf features_B_dev.jsonl

def get_data(data, features, hidden_states, num_hidden_states):
    # get the labels
    data_df = pd.read_json(data, lines=True)
    try:
        labels = data_df["label"]
    except: # there are no labels because it is the test set
        labels = []
    
    # get input vectors, only select the last num_hidden_states
    hidden_states_df = pd.read_json(hidden_states, lines=True)
    hidden_states_df = hidden_states_df.iloc[: , :num_hidden_states]
    features_df = pd.read_json(features, lines=True)

    # make sure there are no NaNs
    hidden_states_df = hidden_states_df.fillna(0)
    features_df = features_df.fillna(0)

    vectors = list()
    vectors_df = pd.DataFrame()

    # add ids, features, merge hidden states into one column and flatten
    vectors_df['feature_vector'] = features_df.values.tolist()
    vectors_df['hidden_states'] = hidden_states_df.values.tolist()
    vectors_df['hidden_states'] = vectors_df['hidden_states'].apply(np.ravel)

    # add ids to dataframe
    vectors_df.insert(0, "id", features_df["id"])
    features_df.drop(["id"], axis=1, inplace=True)

    for hidden_state, feature_vector in zip(vectors_df["hidden_states"], vectors_df["feature_vector"]):
        vectors.append(np.concatenate((hidden_state, feature_vector)))
    
    # also return the correct labels
    return vectors, labels

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtask', '-sb', required=True, help='Subtask (A_mono, A_multi or B)', type=str, choices=['A_mono', 'A_multi', 'B'])
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the training file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--train_hidden_states_file_path", "-trh", required=True, help="Path to the hidden states training file.", type=str)
    parser.add_argument("--train_features_file_path", "-trf", required=True, help="Path to the features training file.", type=str)
    parser.add_argument("--test_hidden_states_file_path", "-th", required=True, help="Path to the hidden states testing file.", type=str)
    parser.add_argument("--test_features_file_path", "-tf", required=True, help="Path to the features testing file.", type=str)
    parser.add_argument("--num_hidden_states", "-n", required=False, help="The number of hidden layers to consider", choices=range(5), type=int, default=4)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = create_arg_parser()
    
    print("Reading data...")
    #X_train, Y_train = get_data(args.train_file_path, args.train_features_file_path, args.train_hidden_states_file_path, args.num_hidden_states)
    
    train_file_1 = ""
    train_file_2 = ""
    feature_file_1 = ""
    feature_file_2 = ""
    hidden_states_1 = ""
    hidden_states_2 = ""
    X_train_1, Y_train_1 = get_data(train_file_1, feature_file_1, hidden_states_1, 2)
    X_train_2, Y_train_2 = get_data(train_file_2, feature_file_2, hidden_states_2, 2)
    X_test, Y_test = get_data(args.test_file_path, args.test_features_file_path, args.test_hidden_states_file_path, args.num_hidden_states)
    
    print("Training...")
    #classifier = LinearSVC()
    #classifier.fit(X_train, Y_train)
    classifier = SGDClassifier()
    classifier.partial_fit(X_train_1, Y_train_1)
    classifier.partial_fit(X_train_2, Y_train_2)

    # Use the fitted classifier to predict classes on the test data
    print("Predicting...")
    Y_pred = classifier.predict(X_test)

    # save predictions with their id to file in correct format
    results_df = pd.DataFrame()

    data_df = pd.read_json(args.test_file_path, lines=True)
    results_df["id"] = data_df["id"]
    results_df["label"] = Y_pred
    results_df.to_json(f"predictions_{args.subtask}_{args.num_hidden_states}.jsonl", lines=True, orient="records")
    print("Successfully saved predictions!")
