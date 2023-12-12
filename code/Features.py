import pandas as pd
import argparse
import json
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk import ngrams
from collections import Counter
from bs4 import BeautifulSoup
import html
import nltk
from spellchecker import SpellChecker
from collections import defaultdict

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', required=True, help='Path to the input file.', type=str)
    parser.add_argument('--output_file', '-o', required=True, help='Path to the output file.', type=str)
    parser.add_argument('--mode', '-m', required=True, choices=['variability', 'sentence_lengths', 'word_lengths', 'sentence_length_range', 'ngrams', 'pos_proportions', 'spelling_errors'], help='Operation mode', type=str)
    parser.add_argument('--ngram_size', '-n', help='Size of the n-grams for ngrams mode', type=int, default=1)
    return parser.parse_args()

def remove_html_artifacts(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    cleaned_text = html.unescape(cleaned_text)
    return cleaned_text

def raw_sentence_lengths(text):
    sentences = sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    return sentence_lengths

def range_of_sentence_lengths(text):
    sentences = sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    if not sentence_lengths:
        return (0, 0)  # In case there are no sentences
    return min(sentence_lengths), max(sentence_lengths)

def calculate_length_variance(sentence_lengths):
    length_variance = np.std(sentence_lengths)
    return length_variance

def word_lengths_in_sentence(sentence):
    return [len(word) for word in word_tokenize(sentence)]

def calculate_ngram_frequencies(text, n):
    tokens = word_tokenize(text.lower())
    ngrams_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngrams_list)
    return ngram_counts

def calculate_pos_proportions(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tag_counts = Counter(tag for word, tag in pos_tags)
    total_tags = sum(tag_counts.values())
    return {tag: count / total_tags for tag, count in tag_counts.items()}

def find_spelling_errors(text):
    spell = SpellChecker()
    words = word_tokenize(text)
    misspelled = spell.unknown(words)
    return len(misspelled), list(misspelled)


if __name__ == '__main__':
    args = create_argparser()

    data = []
    with open(args.input_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    processed_data = defaultdict(lambda: defaultdict(list))

    for item in data:
        item['text'] = remove_html_artifacts(item['text'])
        text_id = item['id']
        model = item['model']
        sentences = sent_tokenize(item['text'])

        if args.mode == 'variability':
            sentence_lengths = raw_sentence_lengths(item['text'])
            length_variance = calculate_length_variance(sentence_lengths)
            processed_data[text_id][model].append(length_variance)

        elif args.mode == 'sentence_lengths':
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            processed_data[text_id][model].append(sentence_lengths)

        elif args.mode == 'word_lengths':
            all_word_lengths = []
            for sentence in sentences:
                all_word_lengths.extend(word_lengths_in_sentence(sentence))
            processed_data[text_id][model].append(all_word_lengths)

        elif args.mode == 'sentence_length_range':
            length_range = range_of_sentence_lengths(item['text'])
            processed_data[text_id][model].append(length_range)

        elif args.mode == 'ngrams':
        n = args.ngram_size  
        ngrams_list = [calculate_ngram_frequencies(sentence, n) for sentence in sentences]
        processed_data[text_id][model].append(ngrams_list)


        elif args.mode == 'pos_proportions':
            pos_proportions_list = [calculate_pos_proportions(sentence) for sentence in sentences]
            processed_data[text_id][model].append(pos_proportions_list)

        elif args.mode == 'spelling_errors':
            spelling_errors_list = [find_spelling_errors(sentence) for sentence in sentences]
            processed_data[text_id][model].append(spelling_errors_list)


    final_data = []
    for text_id, models in processed_data.items():
        for model, data_list in models.items():
            for data in data_list:
                final_data.append({'id': text_id, 'data': data})

    results_df = pd.DataFrame(final_data)

    results_df.to_json(args.output_file, lines=True, orient='records')
