from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from utils import extract_file_name


# https://huggingface.co/spaces/evaluate-metric/perplexity
# https://huggingface.co/docs/transformers/perplexity 


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True, help='Transformer to train and test', type=str)
    parser.add_argument('--input_file', '-i', required=True, help='Path to the input file.', type=str)
    parser.add_argument('--output_file', '-o', help='Path to the output file.', type=str)
    parser.add_argument('--stride', '-s', help='Set stride of the sliding window', default=512, type=int)
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

    # Set up model and tokenizer
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
        file_name = extract_file_name(args.input_file)
        output_file = f'perplexities/{file_name}_{args.model}_{args.stride}.jsonl'
    
    perplexities_df = pd.DataFrame({'id': df['id'], 'perplexity': perplexities})
    perplexities_df.to_json(output_file, lines=True, orient='records')