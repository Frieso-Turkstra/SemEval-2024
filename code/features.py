from spellchecker import SpellChecker
from collections import Counter
import argparse
import pandas as pd
import numpy as np
import nltk
import math
import tqdm
import string
import langdetect
import stanza
import bs4
import html


# Download necessary resources from NLTK
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')


def clean_html_text(text):
    """ Returns the text without any html tags """
    try:
        soup = bs4.BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()
        cleaned_text = html.unescape(cleaned_text)
    except Exception as e:
        print(f"Error in cleaning HTML text: {e}")
        cleaned_text = text
    return cleaned_text


def detect_language(text):
    """ Returns the language of the text """
    try:
        return langdetect.detect(text)
    except:
        return "en"


def load_stanza_model(language):
    """ Returns a loaded stanza model for a given language """
    try:
        return stanza.Pipeline(lang=language, processors='tokenize,pos')
    except Exception as e:
        print(f"Stanza model loading error for language {language}: {e}. Defaulting to English.")
        return stanza.Pipeline(lang='en', processors='tokenize,pos')


def pos_tag_proportions(text, all_tags, stanza_model=None):
    """ Calculates the proportions of POS tags in text """
    if stanza_model is not None:
        doc = stanza_model(text)
        pos_tags = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
    else:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
    
    tag_freq = nltk.FreqDist(tag for (word, tag) in pos_tags)
    total = sum(tag_freq.values())

    # Standardize the POS tag vector
    standardized_vector = [tag_freq[tag] / total if tag in tag_freq else 0 for tag in all_tags]
    return standardized_vector


def calculate_text_coherence(tokens, window_size=2):
    """ Returns (Normalized) Pointwise Mutual Information scores """
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


def get_feature_vector(id, text, subtask, all_tags, loaded_stanza_models):
    vector = [id]

    # pre-calculate some common values that are used in different features
    text = clean_html_text(text)
    words = nltk.word_tokenize(text.lower())
    sentences = nltk.sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    if not sentence_lengths:
        sentence_lengths = [0]

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
    most_common = [item[1] for item in most_common]
    if len(most_common) < 10: # pad with zeroes if not enough bigrams
        most_common += [0 for _ in range(10 - len(most_common))]
    vector += most_common

    # subtask specific features
    if subtask == "A_multi":
        # pos_multi
        language = detect_language(text)
        if language not in loaded_stanza_models:
            loaded_stanza_models[language] = load_stanza_model(language)
        stanza_model = loaded_stanza_models[language]
        vector += pos_tag_proportions(text, all_tags, stanza_model)
    else: 
        # spelling errors
        spell = SpellChecker()
        vector.append(len(spell.unknown(words)))
        
        # pos_mono
        vector += pos_tag_proportions(text, all_tags)
    
    return vector


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A_mono, A_multi, B)", type=str, choices=["A_mono", "A_multi", "B"])
    parser.add_argument("--input_file_path", "-i", required=True, help="Data for which to calculate features", type=str)
    parser.add_argument("--perplexity_file_path", "-p", required=True, help="path to file with perplexity scores", type=str)
    parser.add_argument("--output_file_path", "-o", required=False, help="Output file path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_arg_parser()
    
    # Read in the data
    data_df = pd.read_json(args.input_file_path, lines=True)
    
    # All unique POS tags
    all_tags_multi = ['DET', 'NUM', 'PART', 'ADJ', 'PROPN', 'CCONJ', 'PRON', 'PUNCT', 'SYM', 'SCONJ', 'X', 'INTJ', 'NOUN', 'AUX', 'ADV', 'ADP', 'VERB']
    all_tags_mono = ['MD', 'POS', 'PRP$', 'CD', '(', 'VBP', 'WP', '.', 'WP$', 'VBG', 'SYM', 'VB', 'IN', ')', 'UH', 'RB', 'FW', "''", ':', 'JJ', 'PRP', '#', 'WDT', 'EX', '$', 'LS', 'CC', 'RBS', 'NNPS', 'DT', 'NN', 'JJR', 'VBZ', 'RBR', 'VBN', '``', 'TO', 'VBD', 'PDT', 'RP', 'WRB', 'NNS', 'NNP', 'JJS', ',']
    all_tags = all_tags_multi if args.subtask == "A_multi" else all_tags_mono

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
    ] + [f"bigram{i}" for i in range(10)] + [tag for tag in all_tags]

    if args.subtask != "A_multi":
        features.append("spelling_error")

    # store all feature vectors in a dataframe
    features_df = pd.DataFrame(columns=features)
    loaded_stanza_models = {}

    # calculate features for each sample
    print("Calculating features...")
    for id, text in tqdm.tqdm(zip(data_df["id"], data_df["text"])):
        feature_vector = get_feature_vector(id, text, args.subtask, all_tags, loaded_stanza_models)
        features_df.loc[len(features_df.index)] = feature_vector
    
    # concatenate with the perplexity scores
    perplexity_df = pd.read_json(args.perplexity_file_path, lines=True)
    complete_df = pd.merge(features_df, perplexity_df, on="id")

    # drop and save ids before normalization (otherwise you get normalized ids)
    ids = complete_df["id"]
    complete_df.drop(["id"], axis=1, inplace=True)

    # normalize dataframe column-wise and re-add the ids
    normalized_df = (complete_df - complete_df.mean()) / complete_df.std()
    normalized_df = normalized_df.fillna(0)
    normalized_df.insert(0, "id", ids.astype(int))

    # save to file
    if not (output_file_path := args.output_file_path):
        output_file_path = f"features_{args.subtask}.jsonl"

    normalized_df.to_json(output_file_path, lines=True, orient="records")            
