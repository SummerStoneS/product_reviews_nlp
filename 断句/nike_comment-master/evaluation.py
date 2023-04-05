from typing import Union
import torch as th
from common import Record
from bert import get_model
from preprocess import Labeler


class Splitter:
    def __init__(self, bert_name, model):
        self.base_model = get_model(bert_name)
        self.labeler = Labeler(bert_name)
        self.predictor = model

    def __call__(self, record: Union[str, Record]):
        if isinstance(record, str):
            record = self.labeler(record)
        with th.no_grad():
            x = th.tensor([record.index])
            hidden = self.base_model(x).last_hidden_state.transpose(2, 1)
            prediction = self.predictor(hidden).squeeze().detach().numpy() > 0
        words = self.labeler.decode(record.index[1:-1]).split(" ")
        result = []
        for word, sign in zip(words, prediction):
            result.append(word)
            if sign:
                result.append("|")
        return "".join(result[:-1]).split("|")


if __name__ == '__main__':
    import pickle
    import random
    model = th.load("checkpoint/conv1d-100k-001.pkl")
    splitter = Splitter("bert-base-chinese", model)

    with open("data/processed.pkl", "rb") as f:
        data = pickle.load(f)
    random.seed(0)
    random.shuffle(data)
    print(data[0].sentence)
    test_data = data[100000:101000]
    for i, record in enumerate(test_data):
        result = splitter(record)
        ground_truth = record.sentence
        print(i)
        print(result)
        print(ground_truth)
        print("=" * 20)
