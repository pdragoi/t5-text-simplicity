import torch
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset import Dataset
from torch.utils.data import DataLoader
import argparse
from evaluation import evaluate
from utils import save_plot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="t5-small")
    parser.add_argument("--tokenizer", type=str, default="t5-small")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_file", type=str, default="./dataset/train.csv")
    parser.add_argument("--valid_file", type=str, default="./dataset/validation.csv")
    parser.add_argument("--texts", type=str, default="./dataset/texts.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="base")
    parser.add_argument("--save_all", type=bool, default=False)

    return parser.parse_args()


def train(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    optimizer: torch.optim.Optimizer,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 512,
    epochs: int = 10,
    batch_size: int = 32,
    shuffle: bool = False,
    model_name: str = "base",
    save_all: bool = False,
):

    model = torch.nn.DataParallel(model)
    model.to(device)

    acc_old = 0
    acc_hist = []
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_train_loss = 0
        epoch_val_loss = 0

        model.train()
        for text_a, text_b, question, answer in tqdm(train_data_loader, desc="Training..."):
            optimizer.zero_grad()

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

            encoded_target = tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            encoded_input, attention_mask = encoded_input.input_ids, encoded_input.attention_mask
            encoded_target, _ = encoded_target.input_ids, encoded_target.attention_mask

            encoded_input = encoded_input.to(device)
            attention_mask = attention_mask.to(device)
            encoded_target = encoded_target.to(device)

            outputs = model(
                input_ids=encoded_input,
                attention_mask=attention_mask,
                labels=encoded_target,
            )

            loss = outputs.loss.sum()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Epoch {epoch} train loss: {epoch_train_loss / len(train_data_loader.dataset)}")
        
        acc, val_loss = evaluate(
            model=model, 
            tokenizer=tokenizer, 
            test_data_loader=valid_data_loader,
            device=device, 
            max_length=max_length,
            training=True
            # batch_size=batch_size,
            # shuffle=shuffle
        )
        epoch_val_loss += val_loss * batch_size
        print(f"Validation Accuracy: {acc}, Validation Loss: {epoch_val_loss / len(valid_data_loader.dataset)}")
        acc_hist.append(acc)
        train_losses.append(epoch_train_loss / len(train_data_loader.dataset))
        val_losses.append(epoch_val_loss / len(valid_data_loader.dataset))
        
        if acc > acc_old:
            model.module.save_pretrained(f"results/{model_name}/model/best")
            tokenizer.save_pretrained(f"results/{model_name}/tokenizer/best")
            acc_old = acc
        
        if save_all:
            model.module.save_pretrained(f"results/{model_name}/model/{acc}")
            tokenizer.save_pretrained(f"results/{model_name}/tokenizer/{acc}")
        
        
    model.module.save_pretrained(f"results/{model_name}/model/checkpoint-{epoch + 1}")
    tokenizer.save_pretrained(f"results/{model_name}/tokenizer/checkpoint-{epoch + 1}")

    print("Training complete!")
    print(f"Best Accuracy: {acc}")

    save_plot(
        plots=[
            (train_losses, "Train Loss"),
            (val_losses, "Valid Loss")
        ],
        title="Losses",
        xlabel="Epochs",
        ylabel="Loss",
        save_path=f"plots/{model_name}/loss.png"
    )

    save_plot(
        plots=[
            (acc_hist, "Accuracy")
        ],
        title="Scores",
        xlabel="Epochs",
        ylabel="Score",
        save_path=f"plots/{model_name}/scores.png"
    )


if __name__ == "__main__":

    args = parse_args()

    torch.manual_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, model_max_length=args.max_length)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)

   
    train_dataset = Dataset(
        dataset=args.train_file,
        texts=args.texts,
        tokenizer=tokenizer,
    )

    valid_dataset = Dataset(
        dataset=args.valid_file,
        texts=args.texts,
        tokenizer=tokenizer,
    )

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = args.device
    
    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_name=args.model_name,
        save_all=args.save_all,
    )