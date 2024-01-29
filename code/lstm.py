import pandas as pd
import argparse

def get_inputs(hidden_states, features):
    inputs = list()
    for hidden_state, feature in zip(hidden_states, features):
        inputs.append(hidden_state + feature)
    return inputs

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states_file_path", "-hs", required=True, help="Path to the hidden states file.", type=str)
    parser.add_argument("--features_file_path", "-f", required=True, help="Path to the features file.", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = create_arg_parser()
    hidden_states_df = pd.read_json(args.hidden_states_file_path)
    features_df = pd.read_json(args.features_file_path)

    inputs = get_inputs(hidden_states_df, features_df["vector"])
    print(inputs)

