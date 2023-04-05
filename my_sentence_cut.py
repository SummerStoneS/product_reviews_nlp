import pickle
from tqdm import tqdm
from nlp_utils import get_tokenizer, get_model
import pickle
import random
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import re
import pandas as pd
from collections import namedtuple
from typing import Union

Record = namedtuple("Record", ["sentence", "index", "label"])

# 下载过的bert模型
bert_model_dict = {'hfl/chinese-roberta-wwm-ext': 'roberta',
                   'hfl/chinese-bert-wwm-ext': 'wwm',
                   'bert-base-chinese': 'base'
                  }


class Labeler:
    def __init__(self, model_name, is_first_time=True):
        self.tokenizer = get_tokenizer(model_name, is_first_time)

    def __call__(self, sentence):
        label = []
        index = [self.tokenizer.cls_token_id]        # 段落开始符
        for sub_sentence in sentence.split("|"):
            idx = self.tokenizer.encode(sub_sentence, add_special_tokens=False)
            index.extend(idx)
            label.extend([0] * (len(idx) - 1))
            label.append(1)
        index.append(self.tokenizer.cls_token_id)
        assert len(index) - 2 == len(label)
        return Record(sentence, index, label)     # tuple named record, with named three elements

    def decode(self, index):
        return self.tokenizer.decode(index)


class Splitter:
    """
       [unk]是找不到的字，现在有英语，'硌'...要找回原来的字，并且输出unk list 
    
    """
    def __init__(self, bert_name, model_version):
        self.base_model = get_model(bert_name, is_first_time=False)                   
        self.labeler = Labeler(bert_name, is_first_time=False)
        self.predictor = th.load(model_version)
        self.unk_token_list = set()
        
    def initial_word_token_list(self, sentence, decode_str):
        a = re.sub(r"([^\u4E00-\u9Fa5]+)", r" \1 ", sentence).strip()  # 把非中文字符连起来
        token_list = [[words] if re.search(r'[^\u4E00-\u9Fa5]', words) else list(words) for words in a.split(" ")]
        token_list = list(itertools.chain(*token_list))
        unk_set = set(token_list) - set(decode_str)
        self.unk_token_list = self.unk_token_list.union(unk_set)
        return token_list
    
    def get_prediction(self, token_ids):
        with th.no_grad():
            x = th.tensor([token_ids])
            hidden = self.base_model(x).last_hidden_state.transpose(2, 1)
            prediction = self.predictor(hidden).squeeze().detach().numpy() > 0.5  # 理论上应该又一层sigmoid，但是因为大于0等于sigmoid后大于0.5，所以此处省略了
        return prediction

    def __call__(self, record: Union[str, Record]):  # input type can be a string or a record
        if isinstance(record, str):          # if input type is str, then convert it into a record
            record = self.labeler(record)
        prediction = self.get_prediction(record.index)
        decode_record = self.labeler.decode(record.index[1:-1])
        words = decode_record.split(" ")
        if '[UNK]' in words:
            initial_words = self.initial_word_token_list(record.sentence, decode_record)
            if len(words) != len(initial_words):
                print(words)
                print(initial_words)
                raise ValueError("decode list is not the same as initial sentence' token list")
            words = initial_words
        result = []
        for i, (word, sign) in enumerate(zip(words, prediction)):
            result.append(word)
            if sign & (i < len(words)-1):
                result.append("|")
        return "".join(result).split("|")


def cut_sentences(test_data, model_path, bert_name):

	splitter = Splitter(bert_name, model_path)

	df = []
	for i, record in tqdm(enumerate(test_data)):
	    result = splitter(record)
	    print(i)
	    print(record)
	    print(result)
	    print("=" * 20)
	    result = [sentence for sentence in result if (re.search('[\u4e00-\u9fa5]', sentence) != None) & len(sentence) > 2] # 至少有中文和长度大于2句子才被保留
	    df.append((record, result))
	return pd.DataFrame(df, columns=['initial', 'cut_result'])


if __name__ == '__main__':
	sentence = '鞋子不错'
	model_path = "checkpoint/conv1d-all-002.pkl"
	splitter = Splitter('bert-base-chinese', model_path)
	splitter(sentence)

