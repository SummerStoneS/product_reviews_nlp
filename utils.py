import re
import glob
import pandas as pd
import numpy as np
import os
from langconv import Converter


def read_list(path):
    words = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    words = np.unique(words, axis=0) # 删除重复数据
    return words


def remove_useless_words(comment):
    useless_words = ['风格款式介绍', '鞋底材质', '鞋面材质','尺码推荐']   # 淘宝给的format词汇
    useless_words = '|'.join(useless_words)
    return re.sub(useless_words, '', comment)


def process_comment(initial_comment):
    comment = re.sub("[^\u4e00-\u9fa5^a-z^A-Z^0-9，。！,\.! ]", "", initial_comment)    # 只保留中英文和数字
    comment.strip()
    p = r"(?<=[^\d])(\d)(?=[^\d])"    # 为了把"1舒适度很好2透气性不错3好看"这样的句子按照1,2,3分开，每匹配到一个这样的组就在前面加一个空格，后面按照空格分开，序号后面不能是其他数字，避免把37码拆成3 7
    comment = re.sub(p, r" \1", comment)
    content = re.split("，| |。|!|！|,|\.|但是|就是|不过|但|而且|第一|第二|第三|第四|第五|丶|\丨", comment)  #按照标点符号和转折语断句
    # 繁体字转换，段落的词语数至少是2，段落里至少一个中文单词
    content = [Converter('zh-hans').convert(par) for par in content if (len(par) > 1)&(re.search('[\u4e00-\u9fa5]', par)!=None)]
    content = [re.sub(r"([\u4E00-\u9FA5]+?)(\1+)", r"\1", sentence) for sentence in content]  # 去叠字
    content = list(map(remove_useless_words, content))
    content = [sentence.strip() for sentence in content if len(sentence.strip()) > 1]
    return content    


# 依据标点符号分隔单句
def sepSentence(initial_comment):
    if str(initial_comment) == 'nan':
        return []
    sentences = []
    content = process_comment(initial_comment)
    for w in content:
        sentences.append(w)
    return sentences


# 依据标点符号分隔句子列表
def sepSentences(comments_list):
    sentences = []
    for initial_comment in comments_list:
        if str(initial_comment) == 'nan':
            continue
        content = process_comment(initial_comment)
        for w in content:
            sentences.append(w)
    sentences = [x for x in sentences if x]      # 去掉删除完空格就没有东西的行
    return sentences


def connectsepSentences(comments_list):
    sentences = []
    for initial_comment in comments_list:
        if str(initial_comment) == 'nan':
            continue
        content = process_comment(initial_comment)
        sentences.append('|'.join(content))  
    # sentences = [x for x in sentences if x]      # 去掉删除完空格就没有东西的行
    return sentences


def get_source_comments_data(is_first_time=True):
    if is_first_time:
        filenames = glob.glob(r"跑鞋评论数据更新1119\*.xlsx")
        dataset = pd.DataFrame()
        for file_name in filenames:
            file = pd.read_excel(file_name, header=[0,1])
            dataset = pd.concat([dataset, file])
        dataset.columns = new_col_names
        dataset.to_hdf("step_data/new_data.h5", key="src")
    else:
        dataset = pd.read_hdf("step_data/new_data.h5", key="src")
    return dataset


def join_df_sentence(df, col):
    """
        each row of df[col] is a string with sentences joined by ','
        return: a list of sentences that connect sentences in all rows together
    """
    return list(itertools.chain(*df[col].str.split(',').map(lambda x: [sentence.strip() for sentence in x]).tolist()))


class Finder:
    """
        find sentence in cluster result
    """
    def __init__(self, df, search_col):
        """
            each row of df[search_col] is a string with sentences joined by ','
        """
        self.df = df
        self.df['sentence_list'] = self.df[search_col].str.split(',')

    def find_sentence_location(self, sentence):
        for row, cols in self.df.iterrows():
            if sentence in cols[search_col]:
                print(cols)
                print(cols['cluster_sentences'])


class GetWordDict:
    """
        self-defined word
    """
    def __init__(self, folder_location=None):
        self.root = 'model_input/keywords/' if not folder_location else folder_location
        self.type_location_dict = {
        'keyword': 'keywords_sentimental_words.xlsx',
        'positive_comments': 'positive_comments.txt',
        'negative_comments': 'negative_comments.txt',
        'positive_sentiments': 'positive_sentiments',
        'negative_sentiments': 'negative_sentiments',
        'stopwords': 'cn_stopwords.txt'
        }

    def __call__(self, word_type):
        """
            type: keyword, positive comments/sentiments, negative comments/sentiments...
        """

        try:
            file_name = self.type_location_dict[word_type]
            keyword = self.read_files(file_name)
            setattr(self, word_type, keyword)
        except:
            raise KeyError(f'no key named {word_type}')
        return keyword

    def read_files(self, file_name):
        if file_name.split('.')[1] == 'xlsx':
            keyword = pd.read_excel(os.path.join(self.root, file_name))
        elif file_name.split('.')[1] == 'txt':
            keyword = read_list(os.path.join(self.root, file_name))
        else:
            raise ValueError('this type of file is not accepted')
        return keyword


def count_level_1_num(sentence, keyword_dict):
    """
        计算一句话有多少个不重复的一级词汇
    """
    word_cut = list(jieba.cut(sentence))
    word_cut = map(lambda x: keyword_dict[x]['一级'] if x in keyword_dict.keys() else '', word_cut)
    word_cut = [x for x in word_cut if x != '']
    return len(set(word_cut))

def filter_cutting_sentence(sentence_list):
    """
        句子字数大于8，且包含多个一级
    """
    keyword_manager = GetWordDict()
    keyword = keyword_manager('keyword')
    keyword = keyword.set_index('关键字')
    keyword_dict = keyword.T.to_dict('dict')
    jieba.load_userdict(keyword_dict.keys())
    
    len_gt_8 = list(filter(lambda x: len(x) >= 8, sentence_list))
    remain = list(filter(lambda x: len(x) < 8, sentence_list))
    len_gt_8_lv1_num = map(lambda x: count_level_1_num(x, keyword_dict), len_gt_8)
    sentence_to_cut = []
    for i, count in enumerate(len_gt_8_lv1_num):
        sentence = len_gt_8[i]
        if count > 1:
            if re.search(r'和|都', sentence) and (len(sentence) < 28):
                remain.append(sentence)                               # 舒适和颜值都很不错，不宜断、'款式材质都很不错'，'鞋子不论是质量还是款式都很好'
            else:
                sentence_to_cut.append(sentence)
        else:
            remain.append(sentence)
    return sentence_to_cut, remain