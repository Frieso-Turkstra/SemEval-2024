from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import argparse


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A_mono, A_multi or B).", type=str, choices=["A_mono", "A_multi", "B"])
    parser.add_argument("--input_file_path", "-i", required=True, help="Path to the input file.", type=str)
    parser.add_argument("--model", "-m", required=True, help="Transformer model whose 'perplexity' is calculated.", type=str)
    parser.add_argument("--output_file_path", "-o", required=False, help="Path to the output file.", type=str)
    parser.add_argument("--stride", "-s", required=False, help="Set stride of the sliding window.", type=int, default=512)

    args = parser.parse_args()
    return args


def calculate_perplexity(encodings, model, stride, device="cuda"):
    """
    Sliding window implementation adopted from: https://huggingface.co/docs/transformers/perplexity.
    """
    perplexities = []

    for encoding in tqdm(encodings):
        seq_len = encoding.input_ids.size(1)
        nlls = [] # negative log-likelihoods
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


if __name__ == "__main__":

    # Read in command line arguments.
    args = create_argparser()
    subtask = args.subtask
    input_file_path = args.input_file_path
    model_path = args.model
    output_file_path = args.output_file_path
    stride = args.stride

    # Set up model and tokenizer.
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    max_length = tokenizer.model_max_length

    # Read in data.
    df = pd.read_json(input_file_path, lines=True)
    texts = list(df["text"])

    # Tokenize data.
    encodings = [tokenizer(text, return_tensors="pt") for text in texts]

    # Calculate perplexities.
    perplexities = calculate_perplexity(encodings, model, stride)

    # Save results to output file.
    if not output_file_path:
        input_file_name = input_file_path.replace("/", "-").replace("\\", "-")
        dot_index = input_file_name.rfind(".")
        output_file_path = f"perplexities_{input_file_name[:dot_index]}.jsonl"

    perplexities_df = pd.DataFrame({"id": df["id"], "perplexity": perplexities})
    perplexities_df.to_json(output_file_path, lines=True, orient="records")
