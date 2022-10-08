import os
from pathlib import Path
import pickle

class DataHelper:
    def __init__(self, data_dir, dataset_name, extend) -> None:
        # download data to specified dir if not already exist
        self.version = extend
        self.data_path = os.path.join(data_dir, dataset_name)
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(f"{self.data_path}/source-candidate.pickle"):
            raise ValueError(
                f"Download Dataset for {dataset_name} to {self.data_path} first!!"
            )

    def get_data(self):
        with open(f"{self.data_path}/source-reference.pickle", 'rb') as handle1:
            source_reference = pickle.load(handle1)
        with open(f"{self.data_path}/source-candidate.pickle", 'rb') as handle2:
            source_candidate = pickle.load(handle2)

        hyps = []
        refs = []
        querys = []
        scores = []
        seg_score = []
        extend = self.version
        for id, x in enumerate(source_reference):
            hyps_content = source_candidate[x]
            refs_content = source_reference[x]
            source = x
            ssss = [y[0] for y in hyps_content]
            if extend == True and id % 5 == 0:
                ssss.append(source)
            hyps.append(ssss)

            length = len([y[0] for y in hyps_content])
            if extend == True and id % 5 == 0:
                refs.append([refs_content[0]] * (length + 1))
            else:
                refs.append([refs_content[0]] * length)

            if extend == True and id % 5 == 0:
                querys.append([x] * (length + 1))
            else:
                querys.append([x] * length)
            ss = [float(y[1]) for y in hyps_content]
            if extend == True and id % 5 == 0:
                ss.append(0)
            scores.append(ss)

        for l in scores:
            for x in l:
                seg_score.append(x)

        return hyps , refs, querys, scores, seg_score

    def get_sample_level_data(self, hyps, refs, querys, name):
        hyp = []
        ref = []
        query = []

        for x in hyps:
            for s in x:
                if name == 'bq':
                    s = " ".join(s)
                hyp.append(s)
        for x in refs:
            for s in x:
                if name == 'bq':
                    s = " ".join(s)
                ref.append(s)
        for x in querys:
            for s in x:
                if name == 'bq':
                    s = " ".join(s)
                query.append(s)
        return hyp, ref, query

    def get_dev_test_data(self, hyp, ref, query, seg_score):

        hyp_dev = hyp[:len(hyp) // 10]
        hyp_test = hyp[len(hyp) // 10:]

        ref_dev = ref[:len(ref) // 10]
        ref_test = ref[len(ref) // 10:]

        query_dev = query[:len(query) // 10]
        query_test = query[len(query) // 10:]

        seg_score_dev = seg_score[:len(seg_score) // 10]
        seg_score_test = seg_score[len(seg_score) // 10:]

        return hyp_dev, hyp_test, ref_dev, ref_test, query_dev, query_test, seg_score_dev, seg_score_test


