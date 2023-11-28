import pandas as pd
import math


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
    for i in range(num_files):
        file = df[i*batch_size:i*batch_size+batch_size]
        file.to_json(f'{file_path}_{i}', lines=True, orient='records')


def merge_files(files: list, output_file: str) -> None:
    """
    files -> list of json files to be merged
    output_file -> path to which the merged jsonl file will be saved
    """
    dfs = [pd.read_json(file, lines=True) for file in files]
    temp = pd.concat(dfs, ignore_index=True)
    temp.to_json(output_file, lines=True, orient='records')
