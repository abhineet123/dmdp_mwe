import sys, os

sys.path.append(os.path.abspath(os.getcwd()))
import math
from collections import defaultdict
from evaluation.devkit.MOT.MOT_metrics import MOTMetrics
from evaluation.devkit.Evaluator import Evaluator, run_metrics
import multiprocessing as mp
import pandas as pd

import time
from os import path
import numpy as np


class MOT_evaluator(Evaluator):
    def __init__(self):
        Evaluator.__init__(self)
        
        self.type = "MOT"

    def eval(self):

        # print("Check prediction files")
        # error_message = ""
        for pred_file in self.tsfiles:
            # print(pred_file)
            # check if file is comma separated
            try:
                df = pd.read_csv(pred_file, header=None, sep=",")
            except pd.errors.EmptyDataError:
                print('empty file found: {}'.format(pred_file))
                continue

            if len(df.columns) == 1:
                f = open(pred_file, "r")
                error_message = "Submission %s not in correct form. Values in file must be comma separated." \
                                "Current form:<br>%s<br>%s<br>.........<br>" % (
                    pred_file.split("/")[-1], f.readline(), f.readline())
                raise Exception(error_message)

            df.groupby([0, 1]).size().head()
            count = df.groupby([0, 1]).size().reset_index(name='count')

            # check if any duplicate IDs
            if any(count["count"] > 1):
                doubleIDs = count.loc[count["count"] > 1][[0, 1]].values
                error_message = "Found duplicate ID/Frame pairs in sequence %s." % pred_file.split("/")[-1]
                for id in doubleIDs:
                    double_values = df[((df[0] == id[0]) & (df[1] == id[1]))]
                    for row in double_values.values:
                        error_message += "\n%s" % row
                raise Exception(error_message)

                # error_message += "<br> <!exc> "
        # if error_message != "":
        #     raise Exception(error_message)

        print("Files are ok!")
        arguments = []

        for seq, res, gt in zip(self.sequences, self.tsfiles, self.gtfiles):
            arguments.append({"metricObject": MOTMetrics(seq), "args": {
                "gtDataDir": os.path.join(self.datadir, seq),
                "sequence": str(seq),
                "pred_file": res,
                "gt_file": gt,
                "benchmark_name": self.benchmark_name}})
        # try:
        if self.MULTIPROCESSING:
            p = mp.Pool(self.NR_CORES)
            print("Evaluating on {} cpu cores".format(self.NR_CORES))
            processes = [p.apply_async(run_metrics, kwds=inp) for inp in arguments]
            self.results = [p.get() for p in processes]
            p.close()
            p.join()

        else:
            self.results = [run_metrics(**inp) for inp in arguments]

        # self.failed = False
        # except:
        #     self.failed = True
        #     raise Exception("<exc> MATLAB evalutation failed <!exc>")
        self.Overall_Results = MOTMetrics("OVERALL")


def get_file_lists(benchmark_name=None, gt_dir=None, res_dir=None, save_pkl=None, eval_mode="train",
                   seqmaps_dir="seqmaps"):
    """
    Params
    -----
    benchmark_name: Name of benchmark, e.g. MOT17
    gt_dir: directory of folders with gt data, including the c-files with sequences
    res_dir: directory with result files
        <seq1>.txt
        <seq2>.txt
        ...
        <seq3>.txt
    eval_mode:
    seqmaps_dir:
    seq_file: File name of file containing sequences, e.g. 'c10-train.txt'
    save_pkl: path to output directory for final results
    """

    benchmark_gt_dir = gt_dir
    seq_file = "{}-{}.txt".format(benchmark_name, eval_mode)

    res_dir = res_dir
    benchmark_name = benchmark_name
    seqmaps_dir = seqmaps_dir

    mode = eval_mode

    datadir = os.path.join(gt_dir, mode)

    # getting names of sequences to evaluate
    assert mode in ["train", "test", "all"], "mode: %s not valid " % mode

    print("Evaluating Benchmark: %s" % benchmark_name)

    # ======================================================
    # Handle evaluation
    # ======================================================

    # load list of all sequences
    seq_path = os.path.join(seqmaps_dir, seq_file)
    seq_path = os.path.abspath(seq_path)
    sequences = np.genfromtxt(seq_path, dtype='str', skip_header=True)

    gtfiles = []
    tsfiles = []
    for seq in sequences:
        # gtf = os.path.join(benchmark_gt_dir, mode, seq, 'gt/gt.txt')
        gtf = os.path.join(benchmark_gt_dir, '{}.txt'.format(seq))
        if path.exists(gtf):
            gtfiles.append(gtf)
        else:
            raise Exception("Ground Truth %s missing" % gtf)
        tsf = os.path.join(res_dir, "%s.txt" % seq)
        if path.exists(gtf):
            tsfiles.append(tsf)
        else:
            raise Exception("Result file %s missing" % tsf)

    return gtfiles, tsfiles, datadir, sequences


def main():
    eval = MOT_evaluator()

    benchmark_name = "2D_MOT_2015"
    # gt_dir = "C:/Datasets/MOT2015/2DMOT2015Labels"
    gt_dir = "C:/Datasets/MOT2015/Annotations"
    res_root_dir = 'C:/UofA/PhD/Code/deep_mdp/tracking_module/log'
    res_dir = "no_ibt_mot15_0_10_100_100/lk_wrapper_tmpls2_svm_min10_active_pt_svm/MOT15_0_10_100_100/max_lost0_trd34"
    res_dir = os.path.join(res_root_dir, res_dir)

    eval_mode = "train"
    seqmaps_dir = "../seqmaps"
    gtfiles, tsfiles, datadir, sequences = get_file_lists(
        benchmark_name=benchmark_name,
        gt_dir=gt_dir,
        res_dir=res_dir,
        seqmaps_dir=seqmaps_dir,
        eval_mode=eval_mode)

    eval.run(gtfiles, tsfiles, datadir, sequences, benchmark_name)


if __name__ == "__main__":
    main()
