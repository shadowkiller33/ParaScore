import os
import pathlib
import random
import json
import numpy as np
from scipy import linalg
import sklearn.cluster
import nltk
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel
)

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# ----- tools about configuring model and loading data ----- #

def edit(x,y,chinese = 0):
    if chinese=='bq':
        x = x.replace(" ", "")
        y = y.replace(" ", "")
    a = len(x)
    b = len(y)
    dis = nltk.edit_distance(x,y)
    return dis/max(a,b)

def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

def average(lists):
    for i in range(len(lists)):
        lists[i] = [sum(lst) / len(lst) for lst in lists[i]]
    return lists

def read_data(file_path):
    querys, answers = [], []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            qa = line.split('\t')
            query, answer = qa[:2] if len(qa) > 1 else (qa[0], '')
            querys.append(query.strip())
            answers.append(answer.strip())
    return querys, answers

def get_model_configs(pretrained_model_path, is_chinese=False):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = AutoModel.from_pretrained(pretrained_model_path, return_dict=True)
    return tokenizer, model

def get_embeddings(querys, answers, tokenizer, model, batch_size, use_cuda=True):
    feats = []
    model.eval()
    if use_cuda:
        model.to('cuda')

    with torch.no_grad():
        num_batches = len(querys) // batch_size
        if len(querys) % batch_size > 0:
            num_batches += 1
        for i in tqdm(range(num_batches)):
            query = querys[i*batch_size : (i+1)*batch_size]
            answer = answers[i*batch_size : (i+1)*batch_size]
            if len(query)== 0 or len(answer) == 0:
                continue 
            inputs = tokenizer(query, answer, return_tensors='pt', padding=True)
            if use_cuda:
                inputs.to('cuda')
            outputs = model(**inputs)
            feats.append(outputs.last_hidden_state[:, 0].cpu().data)
    feats = torch.cat(feats).numpy()
    
    return feats

def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)



def _cluster_into_bins(eval_data, ref_data, num_clusters):
    """Clusters the union of the data points and returns the cluster distribution.
    Clusters the union of eval_data and ref_data into num_clusters using minibatch
    k-means. Then, for each cluster, it computes the number of points from
    eval_data and ref_data.
    Args:
        eval_data: NumPy array of data points from the distribution to be evaluated.
        ref_data: NumPy array of data points from the reference distribution.
        num_clusters: Number of cluster centers to fit.
    Returns:
        Two NumPy arrays, each of size num_clusters, where i-th entry represents the
        number of points assigned to the i-th cluster.
    """

    cluster_data = np.vstack([eval_data, ref_data])
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit(cluster_data).labels_

    eval_labels = labels[:len(eval_data)]
    ref_labels = labels[len(eval_data):]

    eval_bins = np.histogram(eval_labels, bins=num_clusters,
                             range=[0, num_clusters], density=True)[0]
    ref_bins = np.histogram(ref_labels, bins=num_clusters,
                            range=[0, num_clusters], density=True)[0]
    return eval_bins, ref_bins





# ----- details on data transformation ----- #

def read_vocab(file):
    vocab = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            vocab.append(line[0])
    return vocab

def read_dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        res = [line.strip() for line in f]
    return res

def read_dialogue(path):
    querys = []
    refs = []
    hyps = []
    human_scores = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            querys.append(line['src'])
            refs.append(line['refs'][0])

            for i, hyp in enumerate(line['hyps']):
                if len(hyps) < i + 1:
                    hyps.append([])
                hyps[i].append(hyp)

            for i, scores in enumerate(line['human_scores']):
                if len(human_scores) < i + 1:
                    human_scores.append([])
                for j, score in enumerate(scores):
                    if len(human_scores[i]) < j + 1:
                        human_scores[i].append([])
                    human_scores[i][j].append(score)
    
    return querys, refs, hyps, human_scores

def transform_qa_pairs(querys, answers, transform, ratio, noise_dict, repeat_dict):
    trans_answers = list(answers)
    trans_indexs = random.sample(list(range(len(querys))), int(len(querys) * ratio))

    if transform == 'noise':
        assert noise_dict != None
        vocab = read_vocab(noise_dict)
        for trans_index in trans_indexs:
            trans_answer = answers[trans_index].split()
            num_lower = len(trans_answer) // 4
            num_upper = max(len(trans_answer) // 2 + 1, len(trans_answer) // 4 + 2)
            num_list = list(range(num_lower, num_upper))
            num = random.choice(num_list)
            for _ in range(num):
                loc = random.randint(0, len(trans_answer))
                word = random.choice(vocab)
                trans_answer.insert(loc, word)
            trans_answers[trans_index] = ' '.join(trans_answer)

    elif transform == 'mismatch':
        indexs = sorted(trans_indexs)
        for index, trans_index in zip(indexs, trans_indexs):
            trans_answers[index] = answers[trans_index]

    elif transform == 'permutate':
        for trans_index in trans_indexs:
            trans_answer = answers[trans_index].split()
            random.shuffle(trans_answer)
            trans_answers[trans_index] = ' '.join(trans_answer)
    
    elif transform == 'repeat':
        assert repeat_dict != None
        repeat_dict = read_dict(repeat_dict)
        for trans_index in trans_indexs:
            trans_answers[trans_index] = random.choice(repeat_dict)

    else:
        raise RuntimeError('Unknown transformation: {}'.format(transform))

    return querys, trans_answers