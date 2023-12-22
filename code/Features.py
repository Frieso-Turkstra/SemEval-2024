import argparse
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from bs4 import BeautifulSoup
import html
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import stanza
from langdetect import detect
from collections import Counter

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

# Function to create a command line argument parser
def create_argparser():
    parser = argparse.ArgumentParser(description='Run linguistic feature extraction for specified subtasks.')
    parser.add_argument('--input_file', '-i', required=True, help='Path to the input file.', type=str)
    parser.add_argument('--output_file', '-o', required=True, help='Path to the output file.', type=str)
    parser.add_argument('--subtask', '-s', required=True, choices=['SubtaskAmono', 'SubtaskAmulti', 'SubtaskB'], help='Subtask to execute', type=str)
    parser.add_argument('--limit', '-l', help='Limit the number of lines to process from the input file.', type=int, default=None)
    parser.add_argument('--ngram_size', '-n', help='Size of the n-grams to be extracted.', type=int, default=2)
    return parser.parse_args()

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
    return np.mean([len(word) for word in words]) if words else 0

# Function to extract top n-grams from data
def extract_top_ngrams(data, ngram_size=2, top_n=100):
    vectorizer = TfidfVectorizer(ngram_range=(ngram_size, ngram_size), stop_words=None)
    X = vectorizer.fit_transform(data)
    sorted_ngrams = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get, reverse=True)
    return set(sorted_ngrams[:top_n])

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

        if p_xy > 0:
            pmi = math.log(p_xy / (p_x * p_y))
            npmi = pmi / (-math.log(p_xy))
        else:
            pmi = 0  # Set pmi to 0 when p_xy is 0
            npmi = 0  # Set npmi to 0 when p_xy is 0

        pmi_values[(word1, word2)] = pmi
        npmi_values[(word1, word2)] = npmi

    avg_pmi = sum(pmi_values.values()) / len(pmi_values) if pmi_values else 0
    avg_npmi = sum(npmi_values.values()) / len(npmi_values) if npmi_values else 0

    return avg_pmi, avg_npmi

# Function to calculate proportions of POS tags in text
def pos_tag_proportions(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tag_freq = FreqDist(tag for (word, tag) in pos_tags)
    total = sum(tag_freq.values())
    return {tag: freq / total for tag, freq in tag_freq.items()}

# Function to calculate POS tag proportions using the Stanza library
def pos_tag_stanza(text, model):
    doc = model(text)
    pos_tags = [word.pos for sent in doc.sentences for word in sent.words]
    tag_freq = FreqDist(pos_tags)
    total = sum(tag_freq.values())
    return {tag: freq / total for tag, freq in tag_freq.items()}

# Function to detect the language of the text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Function to load Stanza model for a given language
def load_stanza_model(language):
    return stanza.Pipeline(lang=language, processors='tokenize,pos')

# Main function
if __name__ == '__main__':
    args = create_argparser()

    loaded_stanza_models = {}

    try:
        with open(args.input_file, 'r') as file:
            data = [json.loads(line) for line in file][:args.limit or None]
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)

    # Extract top n-grams and fit a TF-IDF vectorizer
    common_ngrams = extract_top_ngrams([item['text'] for item in data], ngram_size=args.ngram_size, top_n=100)
    vectorizer = TfidfVectorizer(ngram_range=(1, args.ngram_size), stop_words=None)
    vectorizer.fit_transform([' '.join(common_ngrams)])

    # Process each item in the data
    output_data = []
    for item in data:
        text = clean_html_text(item['text'])

        sentence_lengths = get_sentence_lengths(text)
        word_count = len(word_tokenize(text))
        features = {
            'variability': calculate_sentence_length_variance(sentence_lengths),
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'avg_word_length': calculate_avg_word_length(text),
            'sentence_length_range': calculate_sentence_length_range(sentence_lengths),
        }

        # Calculate text coherence
        avg_pmi, avg_npmi = calculate_text_coherence(text)
        features['text_coherence_avg_pmi'] = avg_pmi
        features['text_coherence_avg_npmi'] = avg_npmi

        # Transform text with TF-IDF vectorizer and add to features
        ngram_vector = vectorizer.transform([text]).toarray().flatten().tolist()
        features['ngrams'] = ngram_vector

        # Perform POS tagging and add relevant features
        if args.subtask == 'SubtaskAmono':
            pos_prop = pos_tag_proportions(text)
            features['POS_tag_proportions'] = pos_prop
            features['spelling_error_ratio'] = count_spelling_errors(text) / word_count if word_count else 0

        elif args.subtask in ['SubtaskAmulti', 'SubtaskB']:
            language = detect_language(text)
            if language not in loaded_stanza_models:
                try:
                    loaded_stanza_models[language] = load_stanza_model(language)
                except Exception as e:
                    print(f"Error loading Stanza model for language {language}: {e}")
                    continue
            pos_prop = pos_tag_stanza(text, loaded_stanza_models[language])
            features['POS_tag_proportions'] = pos_prop

        output_data.append({"id": item['id'], "model": item['model'], "features": features})

    # Write output data to file
    try:
        with open(args.output_file, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)
    except Exception as e:
        print(f"Error writing output file: {e}")
        exit(1)
