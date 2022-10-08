from scipy.stats import spearmanr, pearsonr, kendalltau

def scorer(system_scores, seg_score_test, args):
    pearson_corrs = []
    spearman_corrs = []
    kendall_corrs = []

    pearson_corrs.append((pearsonr(system_scores, seg_score_test)[0]))
    spearman_corrs.append((spearmanr(system_scores, seg_score_test)[0]))
    kendall_corrs.append((kendalltau(system_scores, seg_score_test)[0]))
    print('The pearson correlation between {} and human score is {}'.format(args.metric, pearson_corrs))
    print('The spearman correlation between {} and human score is {}'.format(args.metric, spearman_corrs))
    #print('The kendall correlation between {} and human score is {}'.format(args.metric, kendall_corrs))
    with open('log.txt','a+') as f:
        f.write("#-------------------------------------#\n")
        f.write('In {} mode, The pearson correlation between {} and human score is {}\n'.format(args.setting, args.metric, pearson_corrs))
        f.write('In {} mode, The spearman correlation between {} and human score is {}\n'.format(args.setting, args.metric, spearman_corrs))
        #f.write('The kendall correlation between {} and human score is {}\n'.format(args.metric, kendall_corrs))
        f.write("#-------------------------------------#\n")