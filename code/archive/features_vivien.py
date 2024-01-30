import argparse
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from bs4 import BeautifulSoup
import html
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
import stanza
from langdetect import detect
from collections import Counter
from nltk import bigrams
import string



# Download necessary resources from NLTK
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean HTML text using BeautifulSoup
def clean_html_text(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()
        cleaned_text = html.unescape(cleaned_text)
    except Exception as e:
        print(f"Error in cleaning HTML text: {e}")
        cleaned_text = text
    return cleaned_text

# Function to calculate sentence lengths
def get_sentence_lengths(text):
    sentences = nltk.sent_tokenize(text)
    return [len(sentence.split()) for sentence in sentences]

# Function to calculate variance of sentence lengths
def calculate_sentence_length_variance(sentence_lengths):
    return np.std(sentence_lengths)

# Function to calculate range of sentence lengths
def calculate_sentence_length_range(sentence_lengths):
    if not sentence_lengths:
        return 0
    return max(sentence_lengths) - min(sentence_lengths)

# Function to calculate average word length
def calculate_avg_word_length(text):
    words = word_tokenize(text)
    # Filter out punctuation
    words = [word for word in words if word not in string.punctuation]
    return np.mean([len(word) for word in words]) if words else 0

# Function to extract bigrams
def extract_all_bigrams(data):
    bigrams_text_list = []

    for item in data:
        text = item['text']
        tokens = word_tokenize(text)
        bigrams_list = list(bigrams(tokens))
        bigrams_text = ' '.join(['_'.join(bigram) for bigram in bigrams_list]) # Joining each bigram with an underscore and then all bigrams with spaces
        bigrams_text_list.append(bigrams_text)

    return bigrams_text_list

# Function to vectorize bigrams
def vectorize_bigrams(all_bigrams_text):
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(all_bigrams_text)
    return X, vectorizer.get_feature_names_out()

# Function to count spelling errors in text
def count_spelling_errors(text):
    spell = SpellChecker()
    words = word_tokenize(text)
    misspelled = spell.unknown(words)
    return len(misspelled)

# Function to calculate text coherence using PMI and NPMI
def calculate_text_coherence(text, window_size=2):
    tokens = word_tokenize(text.lower())
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

# Function to calculate proportions of POS tags in text
def pos_tag_proportions(text, all_tags, stanza_model=None):
    if stanza_model is not None:
        doc = stanza_model(text)
        pos_tags = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
    else:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
    
    tag_freq = FreqDist(tag for (word, tag) in pos_tags)
    total = sum(tag_freq.values())

    # Standardize the POS tag vector
    standardized_vector = {tag: (tag_freq[tag] / total if tag in tag_freq else 0) for tag in all_tags}
    return standardized_vector

# Function to detect the language of the text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Function to load Stanza model for a given language
def load_stanza_model(language):
    try:
        return stanza.Pipeline(lang=language, processors='tokenize,pos')
    except Exception as e:
        print(f"Stanza model loading error for language {language}: {e}. Defaulting to English.")
        return stanza.Pipeline(lang='en', processors='tokenize,pos')
        
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Function to create argument parser for command line interface
def create_argparser():
    parser = argparse.ArgumentParser(description='Run linguistic feature extraction based on mode.')
    parser.add_argument('--input_file', '-i', required=True, help='Path to the input file.', type=str)
    parser.add_argument('--output_file', '-o', required=True, help='Path to the output file.', type=str)
    parser.add_argument('--mode', required=True, choices=['variability', 'avg_sentence_length', 'avg_word_length', 'sentence_length_range', 'avg_pmi', 'avg_npmi', 'ngrams', 'pos_mono', 'spelling_error_mono', 'pos_multi'], help='Mode to execute')
    parser.add_argument('--limit', help='Limit the number of data items to process.', type=int, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = create_argparser()

    try:
        with open(args.input_file, 'r') as file:
            data = [json.loads(line) for line in file]
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)

    if args.limit is not None:
        data = data[:args.limit]

    loaded_stanza_models = {}
    output_data = []
    all_tags = set()

    # Preprocessing to collect all unique POS tags
    for item in data:
        text = clean_html_text(item['text'])
        language = detect_language(text)
        if language not in loaded_stanza_models:
            loaded_stanza_models[language] = load_stanza_model(language)
        stanza_model = loaded_stanza_models[language]
        pos_vector = pos_tag_proportions(text, set(), stanza_model)
        all_tags.update(pos_vector.keys())

    # Extract and vectorize all bigrams if the mode is 'ngrams'
    if args.mode == 'ngrams':
        all_bigrams_text = extract_all_bigrams(data)
        bigram_vectors, feature_names = vectorize_bigrams(all_bigrams_text)

    for item in data:
        text = clean_html_text(item['text'])
        feature_result = None

        if args.mode == 'variability':
            sentence_lengths = get_sentence_lengths(text)
            feature_result = calculate_sentence_length_variance(sentence_lengths)
        elif args.mode == 'avg_sentence_length':
            sentence_lengths = get_sentence_lengths(text)
            feature_result = np.mean(sentence_lengths) if sentence_lengths else 0
        elif args.mode == 'avg_word_length':
            feature_result = calculate_avg_word_length(text)
        elif args.mode == 'sentence_length_range':
            sentence_lengths = get_sentence_lengths(text)
            feature_result = calculate_sentence_length_range(sentence_lengths)
        elif args.mode == 'avg_pmi':
            avg_pmi, avg_npmi = calculate_text_coherence(text)
            feature_result = avg_pmi
        elif args.mode == 'avg_npmi':
            avg_pmi, avg_npmi = calculate_text_coherence(text)
            feature_result = avg_npmi
        elif args.mode == 'ngrams':
            N = 10  # Number of top bigrams to extract
            doc_idx = data.index(item)
            bigram_freqs = bigram_vectors[doc_idx].toarray()[0]
            top_bigrams_indices = np.argsort(bigram_freqs)[-N:][::-1]
            top_bigrams_frequencies = [bigram_freqs[idx] for idx in top_bigrams_indices]
            feature_result = top_bigrams_frequencies
        elif args.mode == 'pos_mono':
            feature_result = pos_tag_proportions(text, all_tags)
        elif args.mode == 'spelling_error_mono':
            feature_result = count_spelling_errors(text)
        elif args.mode == 'pos_multi':
            language = detect_language(text)
            if language not in loaded_stanza_models:
                loaded_stanza_models[language] = load_stanza_model(language)
            stanza_model = loaded_stanza_models[language]
            feature_result = pos_tag_proportions(text, all_tags, stanza_model)

        # Convert NumPy types to Python types before JSON serialization
        if isinstance(feature_result, np.number):  # This checks for any kind of NumPy number
            feature_result = feature_result.item()  # Converts to native Python type
        output_data.append({"id": item['id'], "model": item['model'], "feature": {args.mode: feature_result}})

    try:
        with open(args.output_file, 'w') as outfile:
            for item in output_data:
                feature_name = args.mode
                output_line = {"id": item['id'], feature_name: item['feature'][feature_name]}
                json.dump(output_line, outfile, cls=NumpyEncoder)
                outfile.write('\n')
    except Exception as e:
        print(f"Error writing output file: {e}")
        exit(1)