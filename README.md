# On the Evaluation Metrics for Paraphrase Generation

This is the implementation of the paper [On the Evaluation Metrics for Paraphrase Generation](https://arxiv.org/abs/2202.08479).

## Quick links

* [Overview](#overview)
* [Requirements](#requirements)
* [Data](#prepare-the-data)
* [Run](#run)
* [Toolkit](#toolkit)
* [Citation](#citation)


## Overview
In this work we present ParaScore, ParaScore is an evaluation metric for Paraphrase Evaluation. It possesses the merits of reference-based and reference-free metrics and explicitly models lexical divergence. There are two versions of ParaScore: reference-based and reference-free version. 

(1) The reference-based version takes a *reference*, a *source* and a *candidate* as input, then it returns a score that indicates to what extent the candidate is good to be a paraphrase of *source*. 

(2) The reference-free version is used for scenarios where references are unavailable. It takes a *source* and a *candidate* as input, then returns the score to as paraphrase quality indicator. 

You can find more details of this work in our [paper](https://arxiv.org/abs/2202.08479).


## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to little different results from the paper. However, the trend should still hold no matter what versions of packages you use.


## Data
In a standard paraphrase evaluation paradigm, there're source sentences, references, and candidates.

In our paper, two benchmarks are selected: BQ-Para and Twitter-Para. Specifically, BQ-Para is the first Chinese paraphrase evaluation benchmark built by us, and Twitter-Para is adopted from [Code](https://github.com/lanwuwei/Twitter-URL-Corpus) or [Paper](https://aclanthology.org/D17-1126.pdf).

There're two pickles in each dataset, one contains source-reference pairs and the other one contains source-candidate pairs. The IO of benchmarks can refer to [data_utils.py](https://github.com/shadowkiller33/ParaScore/blob/master/src/data_utils.py).


## Run
Here's a simple script for running our codes.

```bash
python run.py \
    --metric parascore \
    --model_type bert-base-uncased \
    --batch_size 16 \
    --dataset_name twitter \
    --data_dir your data dir \
    --setting need \
    --extend True \
```

We further explain some of the script's arguments:

* `metric`: the metric name, including bleu | meteor | rouge | greedy | average | extrema | bert_score | parascore| ibleu 
* `model_type`: the pre-trained model name when using PLM-based metrics like BERTScore.
* `dataset_name`: select dataset
  * `bq`: BQ-para dataset (Chinese)
  * `twitter`: Twitter-para dataset (English)
* `setting`: reference-based or reference-free version
  * `need`: reference-based
  * `free`: reference-free
* `extend`: whether the dataset is extended

this codebase supports pre-trained models in Huggingface's `transformers`. You can check [Huggingface's website](https://huggingface.co/models) for available models and pass models with their names to `--model_type`. Some examples would be `bert-large-uncased`, `roberta-base`, `roberta-large`, etc.


## Toolkit
We also prepare a convinient toolkit for fast usage for our ParaScore, which is called **parascore**. You can install it as follows:

```
pip install parascore==1.0.5
```

Some documents and tutorial can refer to the [homepage](https://github.com/shadowkiller33/parascore_toolkit) for parascore_toolkit package. Here is a simple tutorial for parascore:

```python
from parascore import ParaScorer
scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased')
cands = ["A young person is skating."]
sources = ["There's a child on a skateboard."]
refs = ["A kid is skateboarding."]
score = scorer.base_score(cands, sources, refs, batch_size=16)
print(score)
[0.8152960109710693]
```


## Citation

Please cite our paper if you use ParaScore in your work:

```
@inproceedings{
  title = {On the Evaluation Metrics for Paraphrase Generation},
  author = {Lingfeng Shen, Lemao Liu, Haiyun Jiang, Shuming Shi},
  year = {2022},
  booktitle = {The 2022 Conference on Empirical Methods in Natural Language Processing}
}
```
