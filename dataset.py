import torch
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import T5Tokenizer
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, texts, tokenizer):
        self.dataset = pd.read_csv(dataset)
        self.texts = pd.read_csv(texts, index_col="id")
        self.tokenizer = tokenizer

        self.text_a = self.texts.loc[self.dataset["Text A"].values].text.apply(lambda x: x.replace("\n", " ")).tolist()
        self.text_b = self.texts.loc[self.dataset["Text B"].values].text.apply(lambda x: x.replace("\n", " ")).tolist()
        self.results = self.dataset["Result"].apply(lambda x: min(x, 1)).astype(str).values.tolist()
        # self.results = self.dataset["Result"].apply(lambda x: "Text A" if x >= 1 else "Text B").values.tolist()

        if len(self.text_a) != len(self.text_b) or len(self.text_a) != len(self.results):
            raise ValueError("Something went wrong")


    def __len__(self):
        return len(self.results)


    def __getitem__(self, idx):

        return (
            self.text_a[idx],
            self.text_b[idx],
            "Which of Text A and Text B is easier to understand? 1 for Text A, 0 for Text B.",
            # "Which of Text A and Text B is easier to understand?",
            self.results[idx]
        )


    def _accuracy(self, pred, answer):
        return pred == answer


    def evaluate(self, preds, answers, debug_wrong=False):
        exact_match = 0

        for pred, answer in tqdm(zip(preds, answers)):
            removable_tokens = [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.mask_token_id
            ]

            pred = [token for token in pred if token not in removable_tokens]
            answer = [token for token in answer if token not in removable_tokens]
            
            exact_match += self._accuracy(pred, answer)
            if debug_wrong:
                if not self._accuracy(pred, answer):
                    decoded_prediction = self.tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    decoded_answer = self.tokenizer.decode(answer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    print(f"predicted: {decoded_prediction}; true: {decoded_answer}")
                    print("=====================================")

        exact_match = exact_match / len(preds)

        return exact_match



if __name__ == '__main__':
    dataset = Dataset(
        dataset="./dataset/train.csv",
        tokenizer=T5Tokenizer.from_pretrained('t5-small', model_max_length=512),
        texts="./dataset/texts.csv",
    )
