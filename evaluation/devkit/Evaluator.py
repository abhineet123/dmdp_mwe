import sys, os

sys.path.append(os.getcwd())

import multiprocessing
import time
import numpy as np
from evaluation.devkit.MOT.MOT_metrics import MOTMetrics

class Evaluator(object):
    """ The `Evaluator` class runs evaluation per sequence and computes the overall performance on the benchmark"""

    def __init__(self):
        self.results = None
        self.Overall_Results = None

    def run(self, gtfiles, tsfiles, datadir, sequences, benchmark_name):
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

        print('Evaluating on {} ground truth files and {} test files.'.format(len(gtfiles), len(tsfiles)))
        print('\n'.join(gtfiles))
        print('\n'.join(tsfiles))

        start_time = time.time()

        self.gtfiles, self.tsfiles = gtfiles, tsfiles
        self.datadir = datadir
        self.sequences = sequences
        self.benchmark_name = benchmark_name

        # error_traceback = ""

        self.MULTIPROCESSING = 1
        MAX_NR_CORES = multiprocessing.cpu_count()
        # self.NR_CORES = MAX_NR_CORES
        # set number of core for mutliprocessing
        if self.MULTIPROCESSING:
            self.NR_CORES = np.minimum(MAX_NR_CORES, len(self.tsfiles))
        # try:

        """ run evaluation """
        self.eval()

        # calculate overall results
        results_attributes = self.Overall_Results.metrics.keys()

        for attr in results_attributes:
            """ accumulate evaluation values over all sequences """
            try:
                self.Overall_Results.__dict__[attr] = sum(obj.__dict__[attr] for obj in self.results)
            except:
                pass
        cache_attributes = self.Overall_Results.cache_dict.keys()
        for attr in cache_attributes:
            """ accumulate cache values over all sequences """
            try:
                self.Overall_Results.__dict__[attr] = self.Overall_Results.cache_dict[attr]['func'](
                    [obj.__dict__[attr] for obj in self.results])
            except:
                pass
        print("evaluation successful")

        # Compute clearmot metrics for overall and all sequences
        for res in self.results:
            res.compute_clearmot()

        self.Overall_Results.compute_clearmot()

        self.accumulate_df(type="mail")
        self.failed = False
        # error = None

        # except Exception as e:
        #     print(str(traceback.format_exc()))
        #     print("<br> Evaluation failed! <br>")
        #
        #     error_traceback += str(traceback.format_exc())
        #     self.failed = True
        #     self.summary = None

        end_time = time.time()

        self.duration = (end_time - start_time) / 60.

        # ======================================================
        # Collect evaluation errors
        # ======================================================
        # if self.failed:
        #
        #     startExc = error_traceback.split("<exc>")
        #     error_traceback = [m.split("<!exc>")[0] for m in startExc[1:]]
        #
        #     error = ""
        #
        #     for err in error_traceback:
        #         error += "Error: %s" % err
        #
        #     print("Error Message", error)
        #     self.error = error
        #     print("ERROR %s" % error)

        print("Evaluation Finished in {} sec".format(self.duration))
        print("Your Results")
        str_summary = self.render_summary()
        print(str_summary)
        # save results if path set
        # if save_pkl:
        #
        #     self.Overall_Results.save_dict(
        #         os.path.join(save_pkl, "%s-%s-overall.pkl" % (self.benchmark_name, self.mode)))
        #     for res in self.results:
        #         res.save_dict(os.path.join(save_pkl, "%s-%s-%s.pkl" % (self.benchmark_name, self.mode, res.seqName)))
        #     print("Successfully save results")

        return self.Overall_Results, self.results, self.summary, str_summary

    def eval(self):
        raise NotImplementedError

    def accumulate_df(self, type=None):
        """ create accumulated dataframe with all sequences """
        for k, res in enumerate(self.results):
            res.to_dataframe(display_name=True, type=type)
            if k == 0:
                summary = res.df
            else:
                summary = summary.append(res.df)
        summary = summary.sort_index()

        self.Overall_Results.to_dataframe(display_name=True, type=type)

        self.summary = summary.append(self.Overall_Results.df)

    def render_summary(self, buf=None):
        """Render metrics summary to console friendly tabular output.

        Params
        ------
        summary : pd.DataFrame
            Dataframe containing summaries in rows.

        Kwargs
        ------
        buf : StringIO-like, optional
            Buffer to write to
        formatters : dict, optional
            Dicionary defining custom formatters for individual metrics.
            I.e `{'mota': '{:.2%}'.format}`. You can get preset formatters
            from MetricsHost.formatters
        namemap : dict, optional
            Dictionary defining new metric names for display. I.e
            `{'num_false_positives': 'FP'}`.

        Returns
        -------
        string
            Formatted string
        """

        output = self.summary.to_string(
            buf=buf,
            formatters=self.Overall_Results.formatters,
            justify="left"
        )

        return output


def run_metrics(metricObject, args):
    """ Runs metric for individual sequences

    :param MOTMetrics metricObject: metricObject that has computer_compute_metrics_per_sequence function
    :param dict args: dictionary with args for evaluation function
    :return:
    """

    metricObject.compute_metrics_per_sequence(**args)
    return metricObject


if __name__ == "__main__":
    Evaluator()
