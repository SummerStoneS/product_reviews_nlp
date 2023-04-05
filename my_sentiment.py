import os
import pandas as pd
import numpy as np
import jieba
import re
from langconv import Converter
import glob
from utils import read_list

distance = 5  # 扫描关键词前后5个词
input_words_location = 'model_input/keywords/'
keyword_tag = pd.read_excel(os.path.join(input_words_location, 'keywords_sentimental_words.xlsx'))     # 关键词词表

# keyword_tag = pd.read_excel(r'model_input\keywords_v2\keywords_sentimental_words.xlsx', sheet_name='情感分析关键词补打标签专用')    # 关键词词表
keys = set(keyword_tag.index.to_list())

# 正面情感词
possentiment_path = os.path.join(input_words_location, 'positive_sentiments.txt')
# 负面情感词
negsentiment_path = os.path.join(input_words_location, 'negative_sentiments.txt')
# 中性情感词
neusentiment_path = os.path.join(input_words_location, 'neutral_sentiments.txt')
# 正面评价词
poscomment_path = os.path.join(input_words_location, 'positive_comments.txt')
# 负面评价词
negcomment_path = os.path.join(input_words_location, 'negative_comments.txt')
# 停用词
stopwords_path = os.path.join(input_words_location, 'cn_stopwords.txt')

# 读取词典
pos_sentiments = read_list(possentiment_path)
neg_sentiments = read_list(negsentiment_path)
neu_sentiments = read_list(neusentiment_path)
pos_comments = read_list(poscomment_path)
neg_comments = read_list(negcomment_path)
stopwords = read_list(stopwords_path)
remove = np.array(['好','不','不是','可','如果','不知','有','不如','一般','再','可以','还要','要','便于','也','又','还','比','和','无','也好','大','小','不怕','不能'])
stopwords = np.setdiff1d(stopwords,remove)


add_addwords = np.array(['如果不是','不知道','不知','如果不是','希望','软塑胶','不明显','体重轻','低配速', '跑鞋', '跑步鞋',
	'无压力','质量也轻','质量轻','不紧','不清楚','不说了','不好刷', '不好洗', '怀疑是不是', '没穿几次', '没穿几天', '配速',
	'没几次', '没几天', '没有穿几次', '没有穿几天', '没有想到', '从没想过', '不要犹豫', '不要太', '不挑'])  # 不要被jieba分开的词
addwords=np.append(pos_sentiments,neg_sentiments)
addwords=np.append(addwords,neu_sentiments)
addwords=np.append(addwords,pos_comments)
addwords=np.append(addwords,neg_comments)
addwords=np.append(addwords,add_addwords)
addwords=np.unique(addwords, axis=0)
# print(len(addwords))

# 前缀否定词
no_words = ['不', '不会', '不太会', '没觉得', "没有", "不是",'没','从没','从来没','不怎么','不容易','不会有','不能']
# 前缀中性词
neu_begin = ['不知','不知道','如果不是','希望', '是不是', '担心', '怕']

# 特殊comment匹配，comment-sentiment字典，特殊comment只能形容特定sentiment。E.g.【好看】只能用于形容['颜值','颜色','外观','款式','搭配’]
search_dict={'好看':['颜值','颜色','外观','款式','搭配'],
             '难看':['颜值','颜色','外观','款式','搭配'],
             '不好看':['颜值','颜色','外观','款式','搭配'],
             '不难看':['颜值','颜色','外观','款式','搭配'],
             '低':['性价比'],
             '低了':['性价比'],
             '帅':['颜值','颜色','外观','款式','搭配'],
             '美':['颜值','颜色','外观','款式','搭配'],
             '真高':['颜值','这颜值','性价比'],
             '很亮':['颜值','颜色','外观','款式','搭配'],
             '在线':['颜值','颜色','外观','款式','搭配','脚感','舒适度','质量','性能'],
             # '不怎么':['老气','好看'],
             # '不容易':['脏'],
             '丑':['颜值','颜色','外观','款式'],
             '太丑':['颜值','颜色','外观','款式'],
             '好丑':['颜值','颜色','外观','款式'],
             '老气':['颜值','颜色','外观','款式'],
             '提了':['配速'],
             '快了':['配速'],
             '提升':['配速'],
             '提高':['配速'],
             '很小':['弹性','回弹','回弹力','反弹','反弹力'],
             '足够':['弹性','回弹','回弹力','反弹','反弹力','舒适度','支撑'],
             '比较大':['弹性','回弹','回弹力','反弹','反弹力'],
             # '有点':['弹性','回弹','回弹力','反弹','反弹力'],
             '大':['弹性','回弹','回弹力','反弹','反弹力'],
             '小':['弹性','回弹','回弹力','反弹','反弹力'],
             '不用担心':neg_sentiments,
             '不怕':neg_sentiments,
             '没色差':['颜色'],
             '没有色差':['颜色'],
             '没什么色差':['颜色'],
             '无语':np.append(neu_sentiments, pos_sentiments),
             '廉价':np.append(neu_sentiments, pos_sentiments),
             '问题':np.append(neu_sentiments, pos_sentiments),
             '偏大':['尺码'],
             '严实':['包裹'],
             '再也不用担心':neg_sentiments,
             '再也不怕':neg_sentiments}

# 特殊词after中有否定也不不代表反面意思; i.e. neg_after_useless中的词在后面遇到neg_comment时直接忽略，E.g.线头处理得不好
neg_after_useless=['硬','胶','胶水','鞋胶','溢胶','脱胶','开胶','线头','褶皱','不磨脚', '胶味',
                   '破损','不软','不硬','不累','不紧','不疼']

# 特殊词after遇到特殊否定则不适用
situation = list(keyword_tag[keyword_tag['一级'] == '适用场景'].index)
cant = ['舒适','舒服','轻松','软','轻盈','很轻','舒适度','很软','轻','便宜','贴合',
        '挺舒服','轻便','合适','轻便','有弹力','轻巧','跟脚','透气','柔软','做工','帅气',
        '复古','搭配','百搭','有弹性','轻薄','轻','挺轻','蛮软','超软','挺弹','超弹',
        '富有弹性','简约','轻快','时尚','耐穿','微软','稍软','软乎','软绵','硬','超硬','巨硬',
        '较硬','码偏','偏小','偏大']
# '脚感','减震',
cant_all = np.append(situation, cant)

# comment-sentiment字典，以下keys(情感comment)不能在指定的values(关键词sentiment)后面做形容词 E.g.【一般】不能在后面形容【尺码】
after_useless_dict = {'有':['做工'],
                      # '不':cant_all,
                      '不':addwords,
                      '不会':cant_all,
                      '不觉得':cant_all,
                      '不容易':cant_all,
                      '没':cant_all,
                      '没有':cant_all,
                      # '再也不用担心':cant_all,
                      # '再也不怕':cant_all,
                      # '不是':cant_all,
                      '不适合':cant,
                      '一般':['尺码'],
                      '无':['材质'],
                      '不舒服':neg_sentiments,
                      '不行': neg_sentiments
                      }


just_before = ['有点']        # 只能在前面作为comment形容sentiment，有点弹性（弹性是中性），弹性有点差（这里的有点不能作为positive出现）


# 2-3 添加jieba自定义分词
for word in addwords:
    jieba.add_word(word)


def get_table(type=None):
    global keyword_tag, keys, situation

    if type:
        keyword_tag = pd.read_excel(os.path.join(input_words_location, 'keywords_sentimental_words.xlsx'), sheet_name=type)
        print(1)
    try:
        keyword_tag = keyword_tag.drop_duplicates(subset=['关键字']).set_index('关键字')
    except:
        pass
    keys = set(keyword_tag.index.to_list())
    situation = list(keyword_tag[keyword_tag['一级'] == '适用场景'].index)


def cal_coefficient_no_break(word,coefficient):
    if word in pos_comments:
        coefficient = 2
    if word in neg_comments:
        coefficient = -1
    return coefficient


 # 关键词前评价词得分
def cal_b_multiscore(before, b_multiscore=0, search_word=""):
    before_ori=before # 复制
    b_flag = 0
    for b_index,b_word in reversed(list(enumerate(before))):
        # 是否有情感词，有则更新before
        if b_word in keys:
            before=before[b_index:]
            break
        # 在before中查找评价词
    if (len(before)) == 0:
        bb_word=before_ori[-1]
    if (before_ori != before) | (before[0] in keys): # 有更新或before的第一个词为sentiment，即before中出现sentiment
        for bb_index,bb_word in enumerate(reversed(before)):
            if (bb_word in pos_comments) | (bb_word in neg_comments): # 找到一个评价词
                if bb_word in no_words:
                    b_multiscore=-1
                    break # 退出 bb_word in reversed(before)
                else:
                    if (before.index(bb_word)) > (int(len(before)/2)): # 该评价词离当前关键词近
                        if bb_word in search_dict.keys():
                            if search_word not in search_dict[bb_word]:
                                b_multiscore=100
                                break
                        if bb_word in pos_comments: # 正面
                            b_multiscore=2
                            break # 退出 bb_word in reversed(before)
                        else: # 负面
                            b_multiscore=-1
                            break # 退出 bb_word in reversed(before)            
                    else: #该评价词离当前关键词远
                        b_multiscore=100 # 标记已找到评价词，而不是没有评价词
                        break # 退出 bb_word in reversed(before) 
        if b_multiscore==0: # 在更新的before中没有评价词，可能出现【不会闷脚和硌脚】并搜索到硌脚
            # 前后关键词属于同一个sentiment或两个sentiment均不为负面时再往前搜索（可出现正正、中中、负负、正中、中正）
            if (keyword_tag.loc[search_word, '情感倾向'] == keyword_tag.loc[before[0], '情感倾向']) | (all([keyword_tag.loc[search_word, '情感倾向'] != '负面', keyword_tag.loc[before[0], '情感倾向'] != '负面'])):
                for k in before_ori[:distance - bb_index]:
                    if k in search_dict.keys():
                        if search_word not in search_dict[k]:
                            continue
                    b_multiscore=cal_coefficient_no_break(k,b_multiscore)
                if b_multiscore != 0:
                    b_flag = 1 # 为计算最终得分时遇到comment-sentiment-sentiment-comment做准备
    else: # 没有更新，before中无sentiment
        for k in before:
            if k in search_dict.keys():
                if search_word not in search_dict[k]:
                    # b_multiscore = 0
                    continue
            b_multiscore=cal_coefficient_no_break(k,b_multiscore)
    if b_multiscore==100:
        b_multiscore=0
    return b_multiscore, b_flag


# 关键词后评价词得分
def cal_a_multiscore(after, a_multiscore=0, b_multi=0, search_word=""):
    after_ori = after
    for a_index,a_word in enumerate(after):
        # 是否有情感词，有则更新after
        if a_word in keys:
            after=after[:a_index+1]
            break
    # 在after中查找评价词
    if (len(after)) == 0:
        aa_word=after_ori[0]
    if (after_ori != after) | (after[-1] in keys): # 有更新或after的最后一个词为sentiment，即after中出现sentiment
        for aa_index,aa_word in enumerate(after):
            if aa_word in just_before:
                continue
            if (aa_word in pos_comments) | (aa_word in neg_comments): # 找到一个评价词
                if aa_word in no_words:
                    a_multiscore=100 # 标记已找到评价词，而不是没有评价词
                    break # 退出 aa_word in after
                else:
                    if (after.index(aa_word)) <= (int(len(after)/2)): # 该评价词离当前关键词近
                        if aa_word in search_dict.keys(): # 是否满足特殊匹配
                            if search_word not in search_dict[aa_word]:
                                a_multiscore=100
                                break
                        '''
                        if aa_word in after_useless_dict.keys(): # aa_word是否能在after中形容当前sentiment
                            if search_word in after_useless_dict[aa_word]:
                                a_multiscore=100
                                break
                        '''
                        if aa_word in pos_comments: # 正面
                            a_multiscore=2
                            break # 退出 aa_word in after
                        else:
                            if search_word in neg_after_useless: # 当前sentiment后面是否能跟neg_comments，例如线头after中的neg_comment不适用
                                continue
                            a_multiscore=-1 # 负面
                            break # 退出 aa_word in after                        
                    else: #该评价词离当前关键词远
                        a_multiscore=100 # 标记已找到评价词，而不是没有评价词
                        break # 退出 aa_word in reversed(before)
        if a_multiscore == 0: # 在更新的after中没有评价词，可能出现【透气性和包裹性都很好】并搜索到透气性
            # 前后关键词属于同一个sentiment或两个sentiment均不为负面时再往后搜索（可出现正正、中中、负负、正中、中正）
            if (keyword_tag.loc[search_word, '情感倾向'] == keyword_tag.loc[after[-1], '情感倾向']) | (all([keyword_tag.loc[search_word, '情感倾向'] != '负面', keyword_tag.loc[after[-1], '情感倾向'] != '负面'])):
                for k in reversed(after_ori[aa_index+1:]):
                    if k in just_before:
                        continue
                    if k in search_dict.keys(): # 是否满足特殊匹配
                        if search_word not in search_dict[k]:
                            a_multiscore=0
                            # break
                            continue
                    if k in no_words:
                        a_multiscore = 100 # 289行是否删去一个tab有待商榷，先标记100
                        if k in after_useless_dict.keys(): # 若是after_useless_dict.keys中的no_word，则直接为0
                            if search_word in after_useless_dict[k]:
                                a_multiscore=0
                        continue
                    if k in after_useless_dict.keys(): # k是否能在after中形容当前sentiment
                        if search_word in after_useless_dict[k]:
                            a_multiscore=0
                            continue
                    a_multiscore=cal_coefficient_no_break(k,a_multiscore)
                    if (search_word in neg_after_useless) & (a_multiscore < 0): # 当前sentiment后面是否能跟neg_comments
                        a_multiscore = 0
                        continue

    else: # 没有更新，after中无sentiment
        for k in reversed(after):
            if k in just_before: # 只能在前面形容则continue
                continue
            if k in search_dict.keys(): # 是否满足特殊匹配
                if search_word not in search_dict[k]:
                    a_multiscore=0
                    # break
                    continue
            if k in after_useless_dict.keys(): # k是否能在after中形容当前sentiment
                if search_word in after_useless_dict[k]:
                    a_multiscore=0
                    # break
                    continue
            if k in pos_comments:
                a_multiscore=2
            if k in neg_comments:
                if search_word in neg_after_useless: # 当前sentiment后面是否能跟neg_comments
                    a_multiscore=0
                    continue
                if k in no_words:
                    a_multiscore=100 # 标记找到评价词
                else:
                    a_multiscore=-1
        if (a_multiscore == 100) & (b_multi == 0):
            a_multiscore=-1
    if a_multiscore==100:
        a_multiscore=0
    return a_multiscore


# 情感评分
def cal_score(corpus, orisen_tag='', keyword_sheet=None):
    """
        corpus: list of sentences
        orisen_tag: original sentiment tag NLP的情感分析结果 用来对比
        keyword_sheet: keywords_sentimental_words.xlsx 里的情感分析关键词 或者 情感分析关键词补打标签专用版
    """

    get_table(type=keyword_sheet)

    result = []
    no_keyword = set()
    
    for i in range(len(corpus)):
        keynum = 0
        flag = 0
        words = [x for x in jieba.cut(corpus[i]) if x not in stopwords]
        reswords = ' '.join(words)
        sentence = corpus[i]
        # 情感分析
        for s_index, word in enumerate(words):
            # 以不知、不知道、如果不是、希望等开头的句子都加到中性
            if word in neu_begin:
                keynum += 1
                result.append({
                    "sentence": sentence,
                    "segmentation": reswords,
                    "sentiment": "中性",
                    "orisen_tag": orisen_tag,
                })
                break  # 退出 word in words，直接到下一个段落
            
            score = 0
            b_multiscore = 0
            a_multiscore = 0
            
            if word not in keys: # 不是关键字
                continue

            keynum += 1
            
            result.append({
                "sentence": sentence,
                "segmentation": reswords,
                "keyword": word,
                "tag_1": keyword_tag.loc[word, "一级"],
                "tag_2": keyword_tag.loc[word, "二级"],
                "orisen_tag": orisen_tag,
            })

            # 计算得分——匹配情感词
            if word in pos_sentiments:
                score = 2
            elif word in neu_sentiments:
                score = 0.5
            elif word in neg_sentiments:
                score = -1
            # else:
                # continue

            # 计算得分——获取邻近词语
            if s_index < distance:
                before = words[0:s_index]
            else:
                before = words[s_index - distance : s_index]
            if s_index > len(words) - distance - 1:
                after = words[s_index + 1 : len(words)]
            else:
                after = words[s_index + 1 : s_index + distance + 1]

            # 计算得分——匹配评价词
            if len(before) == 0:
                b_multiscore=0
            else:
                b_multiscore, b_flag = cal_b_multiscore(before, b_multiscore=0, search_word=word)
            if (len(after) == 0) | (word in search_dict.keys()):
                a_multiscore=0
            else:
                a_multiscore=cal_a_multiscore(after,a_multiscore=0,b_multi=b_multiscore,search_word=word)

            # 计算得分——得最终score
            if (b_multiscore != 0) & (a_multiscore != 0):
                if b_flag == 1: # comment-sentiment-sentiment-comment，第二个sentiment有a_multiscore时第一个sentiment前comment所得b_multiscore对其不适用
                    score = score * a_multiscore
                else:
                    score=score * b_multiscore * a_multiscore
            elif (b_multiscore != 0) & (a_multiscore == 0):
                score = score * b_multiscore
            elif (b_multiscore == 0) & (a_multiscore != 0):
                score = score * a_multiscore
            else:
                score = score
            result[-1]['score'] = score

            # 情感分类
            if score > 0.5:
                result[-1]['sentiment'] = '正面'
            elif score < 0:
                result[-1]['sentiment'] = '负面'
            else:
                result[-1]['sentiment'] = '中性'
#             row += 1
        
        if keynum == 0: # 没有关键词
            # 判断是否有评价词
            score = 0
            this_result = {
                "sentence": sentence,
                "segmentation": reswords,
                "orisen_tag": orisen_tag,
                "score": score
            }
            for word in reversed(words):
                if word in pos_comments:
                    flag = 1
                    if score > 0:
                        score += 1
                    elif score < 0:
                        score *= 1
                    else:
                        score = 1
                elif word in neg_comments:
                    flag = 1
                    if score < 0:
                        if word in no_words:
                            score *= -1
                        else:
                            score += -1
                    elif score > 0:
                        score *= -1
                    else:
                        score = -1
            # 判断得分
            if score > 0:
                this_result['sentiment'] = '正面'
                this_result['score'] = score
                result.append(this_result)
            if score < 0:
                this_result['sentiment'] = '负面'
                this_result['score'] = score
                result.append(this_result)
        
        if (keynum == 0) & (flag == 0): # 没有关键词且没有评价词
            no_keyword.add((sentence, orisen_tag))

    return pd.DataFrame(result), pd.DataFrame(no_keyword, columns=['sentence', 'orisen_tag'])
    