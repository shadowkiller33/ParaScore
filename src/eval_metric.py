import argparse
from BARTScore.bart_score import BARTScorer
import bert_score
from src.baseline import cal_bleu, cal_meteor, cal_rougeL,cal_rouge1,cal_rouge2, cal_greedy_match, cal_embd_average, cal_vec_extr,cal_sen_bleu,cal_sen_meteor,process_wordembe
from src.data_utils import *
from src.utils import *
from scipy.stats import spearmanr, pearsonr, kendalltau

def eval_metric(args):
    Dataloader = DataHelper(args.data_dir, args.dataset_name, args.extend)
    hyps, refs, querys, scores, seg_score = Dataloader.get_data()
    hyp, ref, query = Dataloader.get_sample_level_data(hyps, refs, querys, args.dataset_name)
    hyp_dev, hyp_test, ref_dev, ref_test, query_dev, query_test, seg_score_dev, seg_score_test = Dataloader.get_dev_test_data(hyp, ref, query, seg_score)
    system_scores = []
    print("#-------------------------------------#")
    print(args.metric, args.model_type,args.extend)
    with open('log.txt','a+') as f:
        f.write("#-------------------------------------#\n")
        f.write(str(args.metric)+' '+str(args.model_type)+' '+str(args.dataset_name)+' '+str(args.setting)+' '+'extend'+':'+str(args.extend)+'\n')
        f.write("#-------------------------------------#\n")
    print("#-------------------------------------#")



    if args.setting == 'need':
        if args.metric == 'bert_score':
            score = bert_score.score(hyp_test, ref_test, model_type=args.model_type, batch_size=args.batch_size)
            system_scores = score[2].cpu().numpy().tolist()

        elif args.metric == 'bleu':
            for x,y in zip(ref_test, hyp_test):
                system_scores.append(cal_sen_bleu(x,y, args.dataset_name))

        elif args.metric == 'meteor':
            system_scores = cal_sen_meteor(ref_test, hyp_test)

        elif args.metric == 'rougeL':
            system_scores = cal_rougeL(ref_test, hyp_test)

        elif args.metric == 'rouge1':
            system_scores = cal_rouge1(ref_test, hyp_test)

        elif args.metric == 'rouge2':
            system_scores = cal_rouge2(ref_test, hyp_test)

        elif args.metric == 'ibleu':
            s1 = []
            s2 = []
            for x,y in zip(hyp_test,ref_test):
                s1.append(cal_sen_bleu(x,y, args.dataset_name))
            for x,y in zip(hyp_test,query_test):
                s2.append(cal_sen_bleu(x,y, args.dataset_name))
            system_scores = [a_i - args.alpha*b_i for a_i, b_i in zip(s1, s2)]

        elif args.metric == 'parascore':
            s2 = []
            thresh = 0.35
            for x, y in zip(query_test, hyp_test): #calculating diversity
                div = edit(x, y, args.dataset_name)
                if div >= thresh:
                    ss = thresh
                elif div < thresh:
                    ss = -1 + ((thresh+1)/thresh)*div
                s2.append(ss)

            score = bert_score.score(query_test, hyp_test, model_type=args.model_type, batch_size=args.batch_size)
            score2 = bert_score.score(ref_test, hyp_test, model_type=args.model_type, batch_size=args.batch_size)
            dis_query_hyp = score[2].cpu().numpy().tolist()
            dis_ref_hyp = score2[2].cpu().numpy().tolist()
            s1 = [max(x,y) for x,y in zip(dis_query_hyp, dis_ref_hyp)]

            s2_dev = []
            for x, y in zip(query_dev, hyp_dev): #calculating diversity
                div = edit(x, y, args.dataset_name)
                if div >= thresh:
                    ss = thresh
                elif div < thresh:
                    ss = -1 + ((thresh+1)/thresh)*div
                s2_dev.append(ss)
            score_dev = bert_score.score(query_dev, hyp_dev, model_type=args.model_type, batch_size=args.batch_size)
            score_dev2 = bert_score.score(ref_dev, hyp_dev, model_type=args.model_type, batch_size=args.batch_size)
            dev_dis_query_hyp = score_dev[2].cpu().numpy().tolist()
            dev_dis_ref_hyp = score_dev2[2].cpu().numpy().tolist()
            s1_dev = [max(x, y) for x, y in zip(dev_dis_query_hyp, dev_dis_ref_hyp)]
            dev_performance = []
            for i in range(100):
                system_scores_dev = [a_i + 0.01 *i* b_i for a_i, b_i in zip(s1_dev, s2_dev)]
                dev_performance.append((pearsonr(system_scores_dev, seg_score_dev)[0]))

            index = dev_performance.index(max(dev_performance))
            system_scores = [a_i + 0.01*(index+1)*b_i for a_i, b_i in zip(s1, s2)]

        elif args.metric == 'bartscore':
            # if args.is_chinese == 1:
            #     model = 'bart-cn.bin'
            # else:
            #     model = 'bart.pth'
            bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')#fnlp/bart-base-chinese
            bart_scorer.load(path='bart.pth')
            out = bart_scorer.score(ref_test, hyp_test, batch_size=16)
            system_scores = out


    elif args.setting == 'free':
        if args.metric == 'bert_score':
            # if args.is_chinese == 1:
            #     model = 'bert-base-chinese'
            # else:
            #     model = 'bert-base-uncased'
            score = bert_score.score(query_test, hyp_test, model_type=args.model_type, batch_size=args.batch_size)
            score111 = score[2].cpu().numpy().tolist()
            system_scores = score111

        elif args.metric == 'bleu':
            for x, y in zip(query_test, hyp_test):
                system_scores.append(cal_sen_bleu(x, y, args.dataset_name))

        elif args.metric == 'meteor':
            system_scores = cal_sen_meteor(query_test, hyp_test)

        elif args.metric == 'rougeL':
            system_scores = cal_rougeL(query_test, hyp_test)

        elif args.metric == 'rouge1':
            system_scores = cal_rouge1(query_test, hyp_test)

        elif args.metric == 'rouge2':
            system_scores = cal_rouge2(query_test, hyp_test)

        elif args.metric == 'ibleu':
            s1 = []
            s2 = []
            for (hyp, ref) in zip(ref_test, hyp_test):
                s1.append(cal_sen_bleu(ref, hyp, args.dataset_name))
            for (hyp, ref) in zip(query_test, hyp_test):
                s2.append(cal_sen_bleu(ref, hyp, args.dataset_name))
            system_scores = [a_i - args.alpha*b_i for a_i, b_i in zip(s1, s2)]

        elif args.metric == 'selfibleu':
            score = bert_score.score(query_test, hyp_test, model_type=args.model_type, batch_size=args.batch_size)
            score111 = score[2].cpu().numpy().tolist()
            s2 = []
            s1 = score111
            for x, y in zip(query_test, hyp_test):
                s2.append(cal_sen_bleu(x, y, args.dataset_name))
            beta = args.beta
            system_scores = [(beta + 1)/(beta/a_i + 1/(1+b_i))   for a_i, b_i in zip(s1, s2)]

        elif args.metric == 'parascore':
            diversity = []
            thresh = 0.35
            for x, y in zip(query_test, hyp_test):
                div = edit(x, y,args.dataset_name)
                if div>=thresh:
                    ss = thresh
                elif div < thresh:
                    ss = -1 + ((thresh+1)/thresh)*div
                diversity.append(ss)
            score = bert_score.score(query_test, hyp_test, model_type=args.model_type, batch_size=args.batch_size)
            similarity = score[2].cpu().numpy().tolist()

            diversity_dev = []
            for x, y in zip(query_dev, hyp_dev):
                div = edit(x, y,args.dataset_name)
                if div>=thresh:
                    ss = thresh
                elif div < thresh:
                    ss = -1 + ((thresh+1)/thresh)*div
                diversity_dev.append(ss)
            score_dev = bert_score.score(query_dev, hyp_dev, model_type=args.model_type, batch_size=args.batch_size)
            similarity_dev = score_dev[2].cpu().numpy().tolist()
            dev_performance = []
            for i in range(100):
                system_scores_dev = [a_i + 0.01 *i* b_i for a_i, b_i in zip(similarity_dev, diversity_dev)]
                dev_performance.append((pearsonr(system_scores_dev, seg_score_dev)[0]))
            index = dev_performance.index(max(dev_performance))

            system_scores = [a_i + 0.01*b_i*(index+1) for a_i, b_i in zip(similarity, diversity)]


        elif args.metric == 'bartscore':
            bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
            bart_scorer.load(path='bart.pth')
            out = bart_scorer.score(query_test, hyp_test, batch_size=16)
            system_scores = out

    return system_scores, seg_score_test





