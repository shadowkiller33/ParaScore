import argparse
from src.eval_metric import *
from src.score import *

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, help='[bleu | meteor | rouge | greedy | average | extrema | bert_score | parascore| ibleu |]')
parser.add_argument('--model_type', type=str, default='', help='pretrained model type or path to pretrained model')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dataset_name', type=str, default=0, help='dataset name')
parser.add_argument('--data_dir', type=str,  help='data dir')
parser.add_argument('--setting', type=str, default='need', help='reference needed or free')
parser.add_argument('--alpha', type=float, default='0.2', help='alpha in ibleu')
parser.add_argument('--beta', type=float, default='3', help='beta in selfibleu')
parser.add_argument('--extend', type=bool, default=False, help='extended version')
args = parser.parse_args()


if __name__ == '__main__':
    assert args.metric in ['rougeL', 'rouge1', 'ibleu', 'rouge2', 'meteor', 'bert_score', 'bleu', 'bertibleu',
                           'bartscore', 'parascore']
    system_scores, seg_score_test = eval_metric(args)
    scorer(system_scores, seg_score_test, args)