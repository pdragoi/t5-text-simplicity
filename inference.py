import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import Dataset
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--tokenizer", type=str, default="t5-small")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_file", type=str, default="./dataset/test.csv")
    parser.add_argument("--texts", type=str, default="./dataset/texts.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default="predictions.csv")

    return parser.parse_args()


def inference(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    test_data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 512,
    debug: bool = False,
):

    model.to(device)

    model.eval()
    with torch.no_grad():
        encoded_predictions = []
        decoded_predictions = []
        for text_a, text_b, question, answer in tqdm(test_data_loader, desc="Evaluating..."):
            inputs = list(
                map(
                    lambda tuple: f"""Given the following Text A and Text B:\nText A: {tuple[0]}\nText B: {tuple[1]}\nQuestion: {tuple[2]}""",
                    zip(text_a, text_b, question)
                )
            )

            encoded_input = tokenizer(
                inputs,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded_input, attention_mask = encoded_input.input_ids, encoded_input.attention_mask

            encoded_input = encoded_input.to(device)
            attention_mask = attention_mask.to(device)


            model_prediction = model.generate(
                input_ids=encoded_input,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
            )

            encoded_predictions.extend(model_prediction.tolist())


            
            for i, pred in tqdm(enumerate(model_prediction), desc="Decoding..."):
                
                decoded_prediction = tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_predictions.append(decoded_prediction)
                if debug:
                    print(inputs[i])
                    print(f"predicted: {decoded_prediction}")
                    print("=====================================")

    return decoded_predictions


if __name__ == "__main__":

    args = parse_args()

    torch.manual_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, model_max_length=args.max_length)
    model = T5ForConditionalGeneration.from_pretrained(args.model)

    test_dataset = Dataset(
        dataset=args.test_file,
        texts=args.texts,
        tokenizer=tokenizer,
    )
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    device = args.device

    predictions = inference(
        model,
        tokenizer,
        test_data_loader,
        device,
        args.max_length,
        args.debug,
    )

    dataset = pd.read_csv(args.test_file)
    dataset["Result"] = predictions
    dataset.to_csv(args.save_path, index=False)
