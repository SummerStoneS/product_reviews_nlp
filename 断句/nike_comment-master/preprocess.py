import pickle
from tqdm import tqdm
from bert import get_tokenizer
from common import Record


def read_data(filename):
    with open(filename, "r") as f:
        return f.readlines()


def remove_simple_sentences(data):
    return filter(lambda sentence: "|" in sentence, data)


def split_too_long_sentences(data):
    for sentence in data:
        if len(sentence) > 512:
            sub_sentences = sentence.split("|")
            count = len(sub_sentences)
            split_n = count // 2
            yield "|".join(sub_sentences[:split_n])
            yield "|".join(sub_sentences[split_n:])
        else:
            yield sentence


class Labeler:
    def __init__(self, model_name):
        self.tokenizer = get_tokenizer(model_name)

    def __call__(self, sentence):
        label = []
        index = [self.tokenizer.cls_token_id]
        for sub_sentence in sentence.split("|"):
            idx = self.tokenizer.encode(sub_sentence, add_special_tokens=False)
            index.extend(idx)
            label.extend([0] * (len(idx) - 1))
            label.append(1)
        index.append(self.tokenizer.cls_token_id)
        assert len(index) - 2 == len(label)
        return Record(sentence, index, label)

    def decode(self, index):
        return self.tokenizer.decode(index)


if __name__ == '__main__':
    labeler = Labeler('bert-base-chinese')
    data = read_data("data/commentsList.txt")
    data = list(split_too_long_sentences(remove_simple_sentences(data)))
    result = []
    for item in tqdm(data):
        result.append(labeler(item))
    with open("data/processed.pkl", "wb") as f:
        pickle.dump(result, f)
