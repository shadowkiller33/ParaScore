import numpy as np
import math
import rouge
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk.translate.meteor_score import meteor_score
from src.tokenizeChinese import tokenizeSentence

func = nltk.translate.bleu_score.SmoothingFunction()
def cal_bleu(refs, hyps, is_chinese):
    if is_chinese == 'bq':
        refs = [[line.strip().split() for line in tokenizeSentence(ref)] for ref in refs]
        hyps = [line.strip().split() for line in tokenizeSentence(hyps)]
    else:
        refs = [[nltk.word_tokenize(line.strip().lower()) for line in ref] for ref in refs]
        hyps = [nltk.word_tokenize(line.strip().lower()) for line in hyps]

    return corpus_bleu(refs, hyps,weights=(0.25,0.25,0.25,0.25),smoothing_function=func.method1)

def cal_sen_bleu(refs, hyps, is_chinese):

    refs = nltk.word_tokenize(refs.strip().lower())
    hyps = nltk.word_tokenize(hyps.strip().lower())

    return sentence_bleu(refs, hyps,weights=(0.25,0.25,0.25,0.25),smoothing_function=func.method1)

def cal_meteor(refs, hyps):
    scores = []
    for ref, hyp in zip(refs, hyps):
        scores.append(meteor_score(ref, hyp))
    return sum(scores) / len(scores)

def cal_sen_meteor(refs, hyps):
    scores = []
    for ref, hyp in tqdm(zip(refs, hyps)):
        scores.append(meteor_score(ref, hyp))
    return scores

def cal_rougeL(refs, hyps):
    evaluator = rouge.Rouge(metrics=['rouge-l'])
    scores = evaluator.get_scores(hyps, refs)
    score = []
    for x in scores:
        score.append(x['rouge-l']['f'])

    return score

def cal_rouge1(refs, hyps):
    evaluator = rouge.Rouge(metrics=['rouge-1'])
    scores = evaluator.get_scores(hyps, refs)
    score = []
    for x in scores:
        score.append(x['rouge-1']['f'])

    return score

def cal_rouge2(refs, hyps):
    evaluator = rouge.Rouge(metrics=['rouge-2'])
    scores = evaluator.get_scores(hyps, refs)
    score = []
    for x in scores:
        score.append(x['rouge-2']['f'])

    return score

def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    '''
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    '''
 
    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
 
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内
 
 
def conver_float(x):
    float_str = x
    return [float(f) for f in float_str]
 
def process_wordembe(path):
    '''
    :param path: a path of english word embbeding file 'glove.840B.300d.txt'
    :return: a list, element is a 301 dimension word embbeding, it's form like this
            ['- 0.12332 ... -0.34542\n', ', 0.23421 ... -0.456733\n', ..., 'you 0.34521 0.78905 ... -0.23123\n']
    '''
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    res = {}
    for line in tqdm(lines):
        line = line.strip().split()
        try:
            float(line[1])
            res[line[0]] = line[1:]
        except:
            pass

    return res
 
 
def word2vec(x, lines):
    '''
    :param x: a sentence/sequence, type is string, for example 'hello, how are you ?'
    :return: a list, the form like [[word_vector1],...,[word_vectorn]], save per word embbeding of a sentence.
    '''
    x = x.split()
    x_words = []
    for w in x:
        if w in lines:
            x_words.append(conver_float(lines[w]))
    return x_words
 
 
def greedy(x, x_words, y_words):
    '''
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    cosine = []  # 存放一个句子的一个词与另一个句子的所有词的余弦相似度
    sum_x = 0   # 存放最后得到的结果
    for x_v in x_words:
        for y_v in y_words:
            cosine.append(cosine_similarity(x_v, y_v))
        if cosine:
            sum_x += max(cosine)
            cosine = []
    sum_x = sum_x / len(x.split())
    return sum_x

def sentence_embedding(x_words):
    '''
    上面的第一个公式：computing sentence embedding by computing average of all word embeddings of sentence.
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    sen_embed = np.array([0 for _ in range(len(x_words[0]))])  # 存放句向量
 
    for x_v in x_words:
        x_v = np.array(x_v)
        sen_embed = np.add(x_v, sen_embed)
    sen_embed = sen_embed / math.sqrt(sum(np.square(sen_embed)))
    return sen_embed


def greedy_sent_score(x, y, lines):
    
    x_words = word2vec(x, lines)
    y_words = word2vec(y, lines)

    # greedy match
    sum_x = greedy(x, x_words, y_words)
    sum_y = greedy(y, y_words, x_words)
    score = (sum_x+sum_y)/2
    return score

def cal_greedy_match(ref, hyp, lines):
    res = []
    scores = []

    scores.append(greedy_sent_score(ref, hyp, lines))
    res.append(sum(scores) / len(scores))
    return res[0]

def sent_embd_score(x, y, lines):
    x_words = word2vec(x, lines)
    y_words = word2vec(y, lines)

    x_emb = sentence_embedding(x_words)
    y_emb = sentence_embedding(y_words)

    embedding_average = cosine_similarity(x_emb, y_emb)
    return embedding_average

def cal_embd_average(refs, hypss):
    lines = process_wordembe('./glove.6B.300d.txt')
    res = []
    for hyps in hypss:
        scores = []
        for ref, hyp in tqdm(zip(refs, hyps)):
            scores.append(sent_embd_score(ref[0], hyp, lines))
        res.append(sum(scores) / len(scores))
    return res

def vector_extrema(words):
    '''
    对应公式部分：computing vector extrema by compapring maximun value of all word embeddings in same dimension.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :return: a 300 dimension list, vector extrema
    '''
    vec_extre = np.max(np.array(words), axis=0)
    return vec_extre

def vec_sent_score(x, y, lines):
    x_words = word2vec(x, lines)
    y_words = word2vec(y, lines)

    vec_x = vector_extrema(x_words)
    vec_y = vector_extrema(y_words)
 
    similarity = cosine_similarity(vec_x, vec_y)
    return similarity

def cal_vec_extr(ref, hyp, lines):
    scores = []
    scores.append(vec_sent_score(ref, hyp, lines))
    return scores[0]