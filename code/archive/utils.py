import pandas as pd
import math
import json


def extract_file_name(file_path: str) -> str:
    """
    Removes file path and extension, returns file name.
    """
    file_path = file_path.replace('\\', '/')
    left_index = file_path.rfind('/')
    right_index = file_path.find('.')
    return file_path[left_index+1:right_index]


def split_file(file_path: str, batch_size: int) -> None:
    """
    file_path -> the json file to be split and where the split files are saved
    batch_size -> the size of each split file
    """
    df = pd.read_json(file_path, lines=True)

    num_files = math.ceil(len(df) / batch_size)
    file_path = file_path[:file_path.rfind('.')] # exclude file extension
    
    for i in range(num_files):
        file = df[i*batch_size:i*batch_size+batch_size]
        file.to_json(f'{file_path}_{i}.jsonl', lines=True, orient='records')

file_names = [
    "hidden_states_A_mono_test_0.jsonl",
    "hidden_states_A_mono_test_1.jsonl",
]

def merge_files(files: list, output_file: str) -> None:
    """
    files -> list of jsonl files to be merged
    output_file -> path to which the merged jsonl file will be saved
    """
    dfs = [pd.read_json(file, lines=True) for file in files]
    temp = pd.concat(dfs, ignore_index=True)
    temp.to_json(output_file, lines=True, orient='records')

#merge_files(file_names, "hidden_states_A_mono_test_merged.jsonl")

def merge_files_new():
    """
    files -> list of jsonl files to be merged
    output_file -> path to which the merged jsonl file will be saved
    """
    file_A = ""
    file_B = "" 
    output_file = "" 

    df = pd.read_json(file_A, lines=True)
    with open(output_file, "w") as outfile:
        for row in df.iterrows():
            print(json.dumps(row), file=outfile)

    df = pd.read_json(file_B, lines=True)
    with open(output_file, "a") as outfile:
        for row in df.iterrows():
            print(json.dumps(row), file=outfile)

merge_files_new()

def split_text(text, batch_size=10_000):
    """
    Split text into batches of batch_size
    """
    subtexts = []
    num_batches = math.ceil(len(text) / batch_size)
        
    for i in range(num_batches):
        subtext = text[i*batch_size:i*batch_size+batch_size]
        subtexts.append(subtext)

    return subtexts


def split_too_large_entries(files, max_size=10_000):
    """
    - Detects too large entries (texts larger than max_size)
    - Those entries are split into smaller parts and saved into the original file
    (For perplexity, don't forget to call the merge_on_id function afterwards.)

    files -> list of file_paths 
    max_size -> max number of tokens in one entry
    """
    for file in files:
        # read in original file
        df = pd.read_json(file, lines=True)
        # get rows whose text is too large and remove them from the original
        too_large = df.loc[df['text'].str.len() > max_size]
        df = df.drop(too_large.index)

        # split the texts
        split_entries = []
        for _, row in too_large.iterrows():
            subtexts = split_text(row['text'])
            for subtext in subtexts:
                split_entries.append({'text': subtext, 'id': row['id']})
        
        # add the split entries to the original file and save
        combined = pd.concat([df, pd.DataFrame(split_entries)], ignore_index=True) 
        combined.to_json("test_split.jsonl", lines=True, orient='records')

#split_too_large_entries(["..\data\subtaskA_test_multilingual.jsonl"], 5000)

def merge_on_id(file):
    """
    Merges previously split texts (because they were too large) into one entry
    Calculates perplexity as the mean score of each split part
    """
    df = pd.read_json(file, lines=True)
    unique_ids = set(df['id'])

    data = []
    for id in unique_ids:
        # calculate the mean perplexity
        mean_ppl = df.loc[df['id'] == id].perplexity.mean()
        data.append({'id': id, 'perplexity': mean_ppl})

    new_df = pd.DataFrame(data)
    new_df.to_json(file, lines=True, orient='records')


def create_vector(file_path: str) -> None:
    df = pd.read_json(file_path, lines=True)
    features = df["features"]

    vectors = []
    for i in range(len(features)):
        vector = list(features[i].values())
        ngrams = vector.pop(-1)
        pos_tags = vector.pop(-1)
        vector += ngrams
        vector += pos_tags.values()
        vectors.append(vector)

    df["vector"] = vectors
    df.to_json(file_path, lines=True, orient='records')
