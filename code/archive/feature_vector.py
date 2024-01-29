import argparse
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

def read_and_combine_features(file_paths):
    # Combine features from multiple JSONL files into a single dictionary
    combined_features = {}
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                print(f"Reading file: {file_path}")
                for line in file:
                    item = json.loads(line)
                    id = item.get('id')
                    if id is not None:
                        features = {key: value for key, value in item.items() if key != 'id'}
                        if id not in combined_features:
                            combined_features[id] = features
                        else:
                            combined_features[id].update(features)
                    else:
                        print(f"Warning: Missing 'id' in file {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {file_path}")
        except Exception as e:
            print(f"Unexpected error reading file {file_path}: {e}")
    return combined_features


def normalize_features(all_features, method='z-score'):
    # Normalize the numerical features in the dataset
    scaler = StandardScaler() if method == 'z-score' else MinMaxScaler()

    numerical_features = {}
    non_numerical_features = {}

    # Separate numerical and non-numerical features for normalization
    for id, features in all_features.items():
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                numerical_features.setdefault(id, {})[feature] = value
            else:
                non_numerical_features.setdefault(id, {})[feature] = value

    if not numerical_features:
        print("Warning: No numerical features found for normalization")
        return {}

    feature_matrix = [list(numerical_features[id].values()) for id in numerical_features]
    
    # Check if feature_matrix is properly formed for normalization
    if any(isinstance(x, (list, tuple)) for row in feature_matrix for x in row):
        raise ValueError("Feature matrix contains nested lists or non-numerical values")

    normalized_matrix = scaler.fit_transform(np.array(feature_matrix))

    # Combine the normalized numerical features with the non-numerical features
    normalized_features = {}
    for i, id in enumerate(numerical_features):
        normalized_dict = dict(zip(numerical_features[id].keys(), normalized_matrix[i]))
        normalized_features[id] = {**normalized_dict, **non_numerical_features.get(id, {})}

    return normalized_features

def write_combined_features(features, output_file):
    # Write the combined and normalized features to a JSON file
    if not features:
        print("No features to write")
        return

    try:
        with open(output_file, 'w') as file:
            for id, feature_data in features.items():
                output_line = {"id": id, "features": feature_data}
                file.write(json.dumps(output_line))
                file.write('\n')
            print(f"Features written to {output_file}")
    except Exception as e:
        print(f"Error writing to file {output_file}: {e}")

if __name__ == "__main__":
    # Parse command line arguments for specifying subtask and output file
    parser = argparse.ArgumentParser(description="Combine and normalize features from multiple files.")
    parser.add_argument('--subtask', required=True, help="Subtask name")
    parser.add_argument('--dataset', required=True, choices=['dev', 'train'], help="Dataset type (dev or train)")
    parser.add_argument('--output', required=True, help="Output file path")
    args = parser.parse_args()

    # Define the feature files based on the subtask name
    if args.subtask == 'subtaskAmono':
        subtask_features = ['variability', 'avg_sentence_length', 'avg_word_length', 'sentence_length_range', 'avg_pmi', 'avg_npmi', 'pos', 'spelling_error', 'ngrams']
    elif args.subtask in ['subtaskAmulti', 'subtaskB']:
        subtask_features = ['variability', 'avg_sentence_length', 'avg_word_length', 'sentence_length_range', 'avg_pmi', 'avg_npmi', 'pos', 'ngrams']
    else:
        raise ValueError("Invalid subtask specified")

    feature_files = [f"{args.subtask}_{args.dataset}_{feature}.jsonl" for feature in subtask_features]

    # Read and combine features from specified files
    all_features = read_and_combine_features(feature_files)
    
    # Normalize the combined features
    normalized_features = normalize_features(all_features, method='z-score')
    
    # Write the normalized features to the output file
    write_combined_features(normalized_features, args.output)
