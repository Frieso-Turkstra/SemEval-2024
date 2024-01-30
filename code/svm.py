from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import argparse

# python svm.py -tr ..\data\subtaskA_dev_monolingual.jsonl -t ..\data\subtaskA_dev_monolingual.jsonl -trh hidden_states_A_mono_test.jsonl -th hidden_states_A_mono_test.jsonl -trf features_A_mono_dev.jsonl -tf features_A_mono_dev.jsonl 
# python svm.py -tr subtaskB_train.jsonl -t subtaskB_dev.jsonl -trh hidden_states_B_train.jsonl -th hidden_states_B_test.jsonl -trf features_B_train.jsonl -tf features_B_dev.jsonl

def get_data(data, hidden_states, features):
    # get the labels
    data_df = pd.read_json(data, lines=True)
    labels = data_df["label"]
    
    # get input vectors
    hidden_states_df = pd.read_json(hidden_states, lines=True)
    features_df = pd.read_json(features, lines=True)

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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = create_arg_parser()
    
    print("Reading data...")
    X_train, Y_train = get_data(args.train_file_path, args.train_hidden_states_file_path, args.train_features_file_path)
    X_test, Y_test = get_data(args.test_file_path, args.test_hidden_states_file_path, args.test_features_file_path)
   
    classifier = LinearSVC()
    print("Training...")
    classifier.fit(X_train, Y_train)

    # Use the fitted classifier to predict classes on the test data
    print("Predicting...")
    Y_pred = classifier.predict(X_test)

    # save predictions with their id to file in correct format
    results_df = pd.DataFrame()

    data_df = pd.read_json(args.test_file_path, lines=True)
    results_df["id"] = data_df["id"]
    results_df["label"] = Y_pred
    results_df.to_json(f"predictions_{args.subtask}.jsonl", lines=True, orient="records")
    print("Successfully saved predictions!")
