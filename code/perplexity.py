from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import argparse


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', required=True, help='Path to the input file.', type=str)

    # specify the model or automatically select best model depending on subtask
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subtask', '-sb', help='Subtask (A_mono, A_multi or B)', type=str, choices=['A_mono', 'A_multi', 'B'])
    group.add_argument('--model', '-m', help='Transformer to train and test', type=str)

    parser.add_argument('--output_file', '-o', required=False, help='Path to the output file.', type=str)
    parser.add_argument('--stride', '-s', required=False, help='Set stride of the sliding window', type=int, default=512)
    args = parser.parse_args()
    return args


def calculate_perplexity(encodings, model, stride, device='cuda'):
    perplexities = []

    for encoding in tqdm(encodings):
        seq_len = encoding.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encoding.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        perplexity = torch.exp(torch.stack(nlls).mean())
        perplexities.append(perplexity.item()) 

    return perplexities


if __name__ == '__main__':
    args = create_argparser()

    # Select a model
    models = {
        'A_mono': 'gpt2-xl',
        'A_multi': 'bigscience/bloomz-1b1',
        'B': 'gpt2-xl',
        None: args.model
    }
    model_path = models[args.subtask]

    # Set up model and tokenizer
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = tokenizer.model_max_length

    # Read data
    df = pd.read_json(args.input_file, lines=True)
    texts = list(df['text'])

    # Tokenize data
    encodings = [tokenizer(text, return_tensors='pt') for text in texts]

    # Calculate perplexities
    perplexities = calculate_perplexity(encodings, model, args.stride)

    # Save results to output file
    if not (output_file := args.output_file):
       output_file = f'perplexities_{args.subtask}.jsonl'

    perplexities_df = pd.DataFrame({'id': df['id'], 'perplexity': perplexities})
    perplexities_df.to_json(args.output_file, lines=True, orient='records')
