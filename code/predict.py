from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import argparse


def get_data(file_path, features_file_path, hidden_layers_file_path, num_layers):
    # Get the labels (only for train/dev), return empty labels for the test set.
    data_df = pd.read_json(file_path, lines=True)
    try:
        labels = data_df["label"]
    except:
        labels = []
    
    # Read in data.
    features_df = pd.read_json(features_file_path, lines=True)
    hidden_layers_df = pd.read_json(hidden_layers_file_path, lines=True)

    # Ensure there are enough hidden layers.
    if num_layers > len(hidden_layers_df.iloc[0]):
        raise ValueError(f"Cannot consider the last {num_layers} layers; {hidden_layers_file_path} only contains {len(hidden_layers_df.iloc[0])} layers.")
    
    # Only select the last `num__layers` layers.
    hidden_layers_df = hidden_layers_df.iloc[:,:num_layers]
    
    # Make sure there are no NaNs.
    hidden_layers_df.fillna(value=0, inplace=True)
    features_df.fillna(value=0, inplace=True)

    vectors = list()
    vectors_df = pd.DataFrame()

    # Add features, merge hidden states into one column and flatten.
    vectors_df["feature_vector"] = features_df.values.tolist()
    vectors_df["hidden_layers"] = hidden_layers_df.values.tolist()
    vectors_df["hidden_layers"] = vectors_df["hidden_layers"].apply(np.ravel)

    # Add ids to dataframe.
    vectors_df.insert(0, "id", features_df["id"])

    for hidden_layer, feature_vector in zip(vectors_df["hidden_layers"], vectors_df["feature_vector"]):
        vectors.append(np.concatenate((hidden_layer, feature_vector)))
    
    # Return the vectors and the correct labels.
    return vectors, labels


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A_mono, A_multi or B).", type=str, choices=["A_mono", "A_multi", "B"])
    
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the training file.", type=str)
    parser.add_argument("--train_hidden_layers_file_path", "-trh", required=True, help="Path to the hidden layers training file.", type=str)
    parser.add_argument("--train_features_file_path", "-trf", required=True, help="Path to the features training file.", type=str)
    
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--test_hidden_layers_file_path", "-th", required=True, help="Path to the hidden layers testing file.", type=str)
    parser.add_argument("--test_features_file_path", "-tf", required=True, help="Path to the features testing file.", type=str)
    
    parser.add_argument("--num_layers", "-n", required=True, help="The number of hidden layers to consider.", type=int)
    parser.add_argument("--predictions_file_path", "-p", required=False, help="Path to the predictions file.", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Read in the command line arguments.
    args = create_arg_parser()
    
    print("Reading data...")
    X_train, Y_train = get_data(args.train_file_path, args.train_features_file_path, args.train_hidden_layers_file_path, args.num_layers)
    X_test, Y_test = get_data(args.test_file_path, args.test_features_file_path, args.test_hidden_layers_file_path, args.num_layers)

    print("Training...")
    classifier = LinearSVC()
    classifier.fit(X_train, Y_train)

    # Use the fitted classifier to predict classes on the test data.
    print("Predicting...")
    Y_pred = classifier.predict(X_test)

    # Save predictions with their id to file in correct format.
    data_df = pd.read_json(args.test_file_path, lines=True)

    results_df = pd.DataFrame()
    results_df["id"] = data_df["id"]
    results_df["label"] = Y_pred

    predictions_file_path = args.predictions_file_path
    if not predictions_file_path:
        predictions_file_path = f"predictions_{args.subtask}_{args.num_layers}.jsonl"

    results_df.to_json(predictions_file_path, lines=True, orient="records")
    print("Successfully saved predictions!")
