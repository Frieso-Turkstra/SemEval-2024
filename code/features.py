from spellchecker import SpellChecker
from sklearn import preprocessing
from collections import Counter
import argparse
import pandas as pd
import numpy as np
import nltk
import math
import tqdm
import string


# Function to calculate text coherence using PMI and NPMI
def calculate_text_coherence(tokens, window_size=2):
    word_freq = Counter(tokens)
    pair_freq = Counter(zip(tokens, tokens[1:]))
    total_words = sum(word_freq.values())
    total_pairs = sum(pair_freq.values())
    pmi_values = {}
    npmi_values = {}

    for (word1, word2), pair_count in pair_freq.items():
        p_xy = pair_count / total_pairs
        p_x = word_freq[word1] / total_words
        p_y = word_freq[word2] / total_words

        if p_xy > 0 and p_x * p_y > 0:
            pmi = math.log(p_xy / (p_x * p_y))
            npmi = pmi / (-math.log(p_xy)) if p_xy != 1 else 0
        else:
            pmi = 0
            npmi = 0

        pmi_values[(word1, word2)] = pmi
        npmi_values[(word1, word2)] = npmi

    avg_pmi = sum(pmi_values.values()) / len(pmi_values) if pmi_values else 0
    avg_npmi = sum(npmi_values.values()) / len(npmi_values) if npmi_values else 0

    return avg_pmi, avg_npmi


def get_feature_vector(id, text, subtask):
    vector = [id]

    # calculate some common values that are used in different features
    words = nltk.word_tokenize(text.lower())
    sentences = nltk.sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]

    # variability in sentence length
    vector.append(np.std(sentence_lengths))

    # average sentence length
    vector.append(np.mean(sentence_lengths))

    # average word length
    words = [word for word in words if word not in string.punctuation]
    vector.append(np.mean([len(word) for word in words]))

    # sentence length range
    vector.append(max(sentence_lengths) - min(sentence_lengths))

    # text coherence measures
    avg_pmi, avg_npmi = calculate_text_coherence(words)
    vector.append(avg_pmi)
    vector.append(avg_npmi)

    # count of 10 most frequent bigrams
    most_common = Counter(nltk.bigrams(words)).most_common(10)
    vector += [item[1] for item in most_common]

    # subtask specific features
    if subtask == "A_multi":
        # pos_multi
        pass 
    else: 
        # spelling errors
        spell = SpellChecker()
        vector.append(len(spell.unknown(words)))
        # pos_mono
        pass
    
    return vector


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A_mono, A_multi, B)", type=str, choices=["A_mono", "A_multi", "B"])
    parser.add_argument("--input_file_path", "-i", required=True, help="Data for which to calculate features", type=str)
    parser.add_argument("--perplexity_file_path", "-p", required=True, help="path to file with perplexity scores", type=str)
    parser.add_argument("--output_file_path", "-o", required=False, help="Output file path", type=str, default="features.jsonl")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_arg_parser()
    
    # Read in the data
    data_df = pd.read_json(args.input_file_path, lines=True)
    
    # calculate features for each text
    # perplexity is already calculated separately beforehand for efficiency reasons
    features = [
        "id",
        "variability",
        "avg_sentence_length",
        "avg_word_length",
        "sentence_length_range",
        "avg_pmi",
        "avg_npmi",
    ] + [f"bigram{i}" for i in range(10)]

    # extra = ["pos_multi"] if args.subtask == "A_multi" else ["pos_mono", "spelling_error"]
    extra = ["spelling_error"] if args.subtask != "A_multi" else [] # TEMP
    features += extra

    # store all feature vectors in a dataframe
    features_df = pd.DataFrame(columns=features)
    i = 0
    for id, text in tqdm.tqdm(zip(data_df["id"], data_df["text"])):
        i += 1
        if i == 5:
            break
        feature_vector = get_feature_vector(id, text, args.subtask)
        features_df.loc[len(features_df.index)] = feature_vector
    
    # concatenate with the perplexity scores
    perplexity_df = pd.read_json(args.perplexity_file_path, lines=True)
    perplexity_df = perplexity_df.iloc[:4,:] # TEMP
    complete_df = pd.merge(features_df, perplexity_df, on="id")

    # drop and save ids before normalization (otherwise you get normalized ids)
    ids = complete_df["id"]
    complete_df.drop(["id"], axis=1, inplace=True)

    # normalize dataframe column-wise and re-add the ids
    normalized_df = (complete_df - complete_df.mean()) / complete_df.std()
    normalized_df = normalized_df.fillna(0)
    normalized_df.insert(0, "id", ids.astype(int))

    # save to file
    normalized_df.to_json(args.output_file_path, lines=True, orient="records")
            
