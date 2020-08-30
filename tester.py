import os
import logging
from datetime import datetime

import numpy as np
np.seterr(all='raise')

from input import Input
from data import Data
from utilities import motmetrics_to_file, combined_motmetrics, CustomLogger, add_suffix


class Tester:
    """
    :type _params: Tester.Params
    :type _logger: logging.RootLogger | CustomLogger
    :type input: Input
    """

    class Params:
        def __init__(self):

            self.devkit = 0
            self.accumulative_eval_path = 'log/mot_metrics_accumulative.log'
            self.input = Input.Params()

    def __init__(self, params, logger):
        """
        :type params: Tester.Params
        :type logger: CustomLogger
        :rtype: None
        """

        self._params = params
        self._logger = logger

        self.input = Input(self._params.input, self._logger)

        self.annotations = None

        self._acc_dict = {}

    def initialize(self, data=None, logger=None):
        """
        :type data: Data | None
        :type logger: CustomLogger
        :rtype: bool
        """
        if logger is not None:
            self._logger = logger

        if data is not None:
            # self._logger = CustomLogger(self.__logger, names=(data.seq_name,), key='custom_header')

            """initialize input pipeline"""
            if not self.input.initialize(data, logger=self._logger):
                raise IOError('Input pipeline could not be initialized')

            if not self.input.read_annotations():
                raise IOError('Annotations could not be read')

            self.annotations = self.input.annotations

        return True

    def eval(self, load_fname, eval_path, eval_dist_type):
        """
        :type load_fname: str
        :type eval_path: str
        :type eval_dist_type: int
        :rtype: mm.MOTAccumulator | None
        """

        assert self.input.annotations is not None, "annotations have not been loaded"
        assert self.input.tracking_res is not None, "tracking results have not been loaded"

        seq_name = os.path.splitext(os.path.basename(load_fname))[0]
        if self._params.devkit:
            gtfiles = [self.input.annotations.path, ]
            tsfiles = [self.input.tracking_res.path, ]
            sequences = [seq_name, ]

            datadir = os.path.dirname(self.input.annotations.path)
            benchmark_name = os.path.basename(os.path.dirname(datadir))
            acc = (gtfiles, tsfiles, datadir, sequences, benchmark_name)

            self._logger.info('deferring evaluation for combined results')

            _eval = eval_str = None
        else:
            _eval, eval_str, acc = self.input.annotations.get_mot_metrics(self.input.tracking_res,
                                                                          seq_name, eval_dist_type)

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        if eval_str is not None:
            print('\n' + eval_str + '\n')
            if _eval is None:
                return None
            motmetrics_to_file((eval_path,), _eval, load_fname, seq_name,
                               mode='a', time_stamp=time_stamp, devkit=self._params.devkit)

        self._acc_dict[self.input.seq_name] = acc

        return acc

    def accumulative_eval(self, load_dir, eval_path, _logger):
        """

        :param str load_dir:
        :param str eval_path:
        :param CustomLogger _logger:
        :return:
        """
        accumulative_eval_path = self._params.accumulative_eval_path

        if self._params.devkit:
            accumulative_eval_path = add_suffix(accumulative_eval_path, 'devkit')
            eval_path = add_suffix(eval_path, 'devkit')

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

        if not self._acc_dict or len(self._acc_dict) == 0:
            return

        if self._params.devkit:
            from evaluation.devkit.MOT.evalMOT import MOT_evaluator
            gtfiles = []
            tsfiles = []
            sequences = []
            for _seq in self._acc_dict:
                _args = self._acc_dict[_seq]
                _gtfiles, _tsfiles, _datadir, _sequences, _benchmark_name = _args
                gtfiles.append(_gtfiles[0])
                tsfiles.append(_tsfiles[0])
                sequences.append(_sequences[0])

            datadir = os.path.dirname(self.input.annotations.path)
            benchmark_name = os.path.basename(os.path.dirname(datadir))

            eval = MOT_evaluator()
            _, _, summary, strsummary = eval.run(gtfiles, tsfiles, datadir, sequences, benchmark_name)

            for _seq in self._acc_dict:
                _args = self._acc_dict[_seq]
                _gtfiles, _tsfiles, _datadir, _sequences, _benchmark_name = _args
                motmetrics_to_file((eval_path,), summary, _tsfiles[0], _sequences[0], mode='a',
                                   time_stamp=time_stamp, verbose=0, devkit=self._params.devkit)

        else:
            summary, strsummary = combined_motmetrics(self._acc_dict, _logger)

        motmetrics_to_file((eval_path, accumulative_eval_path), summary, load_dir, 'OVERALL',
                           time_stamp=time_stamp, devkit=self._params.devkit)

    def load(self, load_path):
        """
        :type load_path: str
        :rtype: bool
        """
        if not self.input.read_tracking_results(load_path):
            self._logger.error('Tracking results could not be loaded')
            return False
        self._logger.info('Tracking results loaded successfully')
        return True

