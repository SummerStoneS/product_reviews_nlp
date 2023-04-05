
### 给断句模型准备语料
from utils import connectsepSentences, get_source_comments_data
import random

dataset = get_source_comments_data(is_first_time=False)
print("原始评价数：", len(dataset))
comments_list = dataset["评论内容"].dropna().tolist()
print("原始不为空评论数:", len(comments_list))
modeling_data_src = connectsepSentences(comments_list)

def filter_long_sentences(sentence_list):
    return filter(lambda x: ("|" not in x) & (len(x) <= 9) & (len(x) >= 4), sentence_list)

def connect_short_sentences(short_sentences):
    random_idx = [random.randrange(0, len(short_sentences)) for i in range(8)]
    random_sentences = [short_sentences[idx] for idx in random_idx]
    return "|".join(random_sentences)


long_sentences = list(filter(lambda x: "|" in x, modeling_data_src))
print(f"long sentences count: {len(long_sentences)}")
short_sentences = list(filter_long_sentences(modeling_data_src))
print("short sentences count:", len(short_sentences))
short_sentences.sort(key=len, reverse=True)

# 样本数据给模型训练（没服务器。。。）
random.shuffle(long_sentences)
long_samples = long_sentences[: round(len(long_sentences)*0.6)]
# 随机挑一些拼到一起
short_samples = [connect_short_sentences(short_sentences) for i in range(round(len(long_sentences)*0.4))]
final_train_samples = long_samples + short_samples
print(len(final_train_samples))
with open(r"断句\nike_comment-master\data\modeling_comments.txt", 'w+', encoding='utf-8') as f:
    for sentence in final_train_samples:
        f.write(f"{sentence}\n")

# test数据
long_test = long_sentences[round(len(long_sentences)*0.6):]
short_test = [connect_short_sentences(short_sentences) for i in range(round(len(long_test)*2/3))]
final_test_samples = long_test + short_test
print(len(final_test_samples))
with open(r"断句\nike_comment-master\data\modeling_test_comments.txt", 'w+', encoding='utf-8') as f:
    for sentence in final_test_samples:
        f.write(f"{sentence}\n")