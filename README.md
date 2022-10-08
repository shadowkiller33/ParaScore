# On the Evaluation Metrics for Paraphrase Generation

The implementation of the paper [On the Evaluation Metrics for Paraphrase Generation](https://arxiv.org/abs/2202.08479).

ParaScore is an evaluation metric for Paraphrase Evaluation. It possesses the merits of reference-based and reference-free metrics and explicitly models lexical divergence. There are two versions of ParaScore: reference-based and reference-free version. 

(1) The reference-based version takes a *reference*, a *source* and a *candidate* as input, then it returns a score that indicates to what extent the candidate is good to be a paraphrase of *source*. 

(2) The reference-free version is used for scenarios where references are unavailable. It takes a *source* and a *candidate* as input, then returns the score to as paraphrase quality indicator. 




## Installation






## How to Cite

Please cite our EMNLP paper:

```
@inproceedings{,
  title = {},
  author = {},
  year = {2022},
  booktitle = {Proceedings of EMNLP}
}
```
