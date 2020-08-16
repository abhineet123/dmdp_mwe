import os
import logging

from paramparse import MultiPath

from data import Data
from tester import Tester
from utilities import linux_path, CustomLogger, BaseParams


class RunParams(BaseParams):
    """
    Iterative Batch Train Parameters
    :type load_prefix: MultiPath
    :type save_prefix: MultiPath
    :type results_dir_root: MultiPath
    :type results_dir: MultiPath

    :ivar seq_set: 'Numeric ID of the data set from which the training sequences '
               'have to be taken as defined in the Data.get_sequences(); '
               'at present only sequences from a single data set can be trained on '
               'in a single run',
    0: 'MOT2015',
    1: 'MOT2017',
    2: 'KITTI',
    3: 'GRAM_ONLY',
    4: 'IDOT',
    5: 'DETRAC',
    6: 'LOST',
    7: 'ISL',
    8: 'GRAM',

    :ivar  seq: Numeric IDs of the sequences on which
    training has to be performed as defined in the Data.get_sequences()

    """

    def __init__(self):
        self.seq_set = -1
        self.seq = ()

        self.load = 0
        self.save = 1
        self.start = 0

        self.load_prefix = MultiPath()
        self.save_prefix = MultiPath()
        self.results_dir_root = MultiPath()
        self.results_dir = MultiPath()

    def _synchronize(self, src):
        """
        :type src: RunParams
        """
        if self.seq_set < 0:
            self.seq_set = src.seq_set
        if not self.seq:
            self.seq = src.seq
        if not self.results_dir_root:
            self.results_dir_root = src.results_dir_root
        if not self.results_dir:
            self.results_dir = src.results_dir


class Train:
    class Params(RunParams):
        """
        :type seq_set: int
        :type seq: (int, )

        :ivar load: 1: Load a previously trained tracker and test;
                2: Load a previously trained tracker and continue training;
                0: train from scratch,

        :ivar load_id: ID of the sequence for which to load trained target if
        load_dir is not provided;

        :ivar load_dir: directory from where to load trained target; overrides
        load_id

        :ivar start: ID of the sequence from which to start training; if load_id
        and load_dir are not provided, training is continued after loading the
        target corresponding to the sequence preceding start_id; this ID is specified
        relative to the IDs provided in seq_ids

        :ivar save: Save the trained tracker to disk so it can be loaded later;
        only matters if load is disabled

        :ivar save_prefix: Prefix in the name of the file into which the trained
        tracker is to be saved

        :ivar load_prefix: prefix in the name of the file from which the previously
        trained tracker has to be loaded for testing

        :ivar results_dir: Directory where training results files are written
        to

        """

        def __init__(self):
            RunParams.__init__(self)

            self.load_dir = ''
            self.load_id = -1

            self.seq_set = 4
            self.seq = (5,)

            self.active_pt = 1
            self.active_pt_dir = MultiPath()

            self.load_prefix = MultiPath('trained')
            self.save_prefix = MultiPath('trained')
            self.results_dir_root = MultiPath()
            self.results_dir = MultiPath('log')


class Test:
    class Params(RunParams):
        """
        :type seq_set: int
        :type seq: (int, )
        :type load: int

        :ivar load: 'Load previously saved tracking results from file for evaluation or visualization'
                ' instead of running the tracker to generate new results;'
                'load=2 will load raw results collected online for each frame instead of the post-processed '
                'ones generated at the end of each target',

        :ivar save: Save tracking results to file;only matters if load is disabled

        :ivar save_prefix: Prefix in the name of the file into which the tracking
        results are to be saved

        :ivar load_prefix: prefix in the name of the file from which the previously
        saved tracking results are to be loaded for evaluation or visualization

        :ivar results_dir: Directory where the tracking results file is saved
        in

        :ivar evaluate: 'Enable evaluation of the tracking result; '
                    'only works if the ground truth for the tested sequence is available; '
                    '1: evaluate each sequence and all combined; '
                    '2: evaluate sequences incrementally as well(i.e. seq (1,) (1,2), (1,2,3) and so on); ',
        :ivar eval_dist_type: 'Type of distance measure between tracking result and ground truth '
                          'bounding boxes to use for evaluation:'
                          '0: intersection over union (IoU) distance'
                          '1: squared Euclidean distance; '
                          'only matters if evaluate is set to 1',

        :ivar eval_dir: Name of the Directory into which a summary of the evaluation
        result will be saved; defaults to results_dir if not provided

        :ivar eval_file: Name of the file into which a summary of the evaluation
        result will be written if evaluation is enabled

        """

        def __init__(self):
            RunParams.__init__(self)

            self.seq_set_info = MultiPath()
            self.eval_dir = MultiPath()

            self.mode = 1
            self.evaluate = 1
            self.eval_dist_type = 0
            self.eval_file = 'mot_metrics.log'

            self.subseq_postfix = 1

            self._load_prefix = None
            self._save_prefix = None

        def synchronize(self, src):
            """

            :param Train.Params src:
            :return:
            """
            self._synchronize(src)

    @staticmethod
    def run(data, tester_params, test_params, logger):
        """
        test a trained target
        :type trained_target: Target
        :type data: Data
        :type tester_params: Tester.Params
        :type test_params: Test.Params
        :type logger: logging.RootLogger | logging.logger | CustomLogger
        :type logging_dir: str
        :type args_in: list
        :rtype: bool
        """

        global_logger = logger

        assert test_params.start < len(test_params.seq), f"Invalid start_id: {test_params.start} " \
            f"for {len(test_params.seq)} sequences"

        tester = Tester(tester_params, global_logger)

        success = True
        eval_path = load_dir = None

        evaluate = test_params.evaluate
        eval_dist_type = test_params.eval_dist_type

        results_dir = test_params.results_dir
        results_dir_root = test_params.results_dir_root
        if results_dir_root:
            results_dir = linux_path(results_dir_root, results_dir)

        if test_params.seq_set_info:
            results_dir = linux_path(results_dir, test_params.seq_set_info)

        save_txt = f'saving results to: {results_dir}'
        save_prefix = test_params.save_prefix
        if save_prefix:
            save_txt += f' with save_prefix: {save_prefix}'
        global_logger.info(save_txt)

        n_seq = len(test_params.seq)
        for _id, test_id in enumerate(test_params.seq[test_params.start:]):
            global_logger.info('Running tester on sequence {:d} in set {:d} ({:d} / {:d} )'.format(
                test_id, test_params.seq_set, _id + test_params.start + 1, n_seq))

            if not data.initialize(test_params.seq_set, test_id, 1, logger=global_logger):
                global_logger.error('Data module failed to initialize with sequence {:d}'.format(test_id))
                success = False
                break

            load_dir = results_dir
            load_prefix = test_params.load_prefix
            if load_prefix:
                load_dir = linux_path(load_dir, load_prefix)

            save_dir = results_dir

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_prefix = test_params.save_prefix
            if save_prefix:
                save_dir = linux_path(save_dir, save_prefix)

            eval_dir = test_params.eval_dir
            if not eval_dir:
                eval_dir = save_dir
            eval_path = linux_path(eval_dir, test_params.eval_file)

            seq_logger = CustomLogger(global_logger, names=(data.seq_name,), key='custom_header')

            if not tester.initialize(data, seq_logger):
                seq_logger.error('Tester initialization failed on sequence {:d} : {:s}'.format(
                    test_id, data.seq_name))
                success = False
                break

            if tester.annotations is None:
                seq_logger.warning('Tester annotations unavailable so disabling evaluation')
                evaluate = 0

            """load existing tracking results and optionally visualize or evaluate"""

            if test_params.subseq_postfix:
                load_fname = '{:s}_{:d}_{:d}.txt'.format(data.seq_name, data.start_frame_id + 1,
                                                         data.end_frame_id + 1)
            else:
                load_fname = '{:s}.txt'.format(data.seq_name)

            load_path = linux_path(load_dir, load_fname)
            if evaluate:
                if not tester.load(load_path):
                    seq_logger.error('Tester loading failed on sequence {:d} : {:s}'.format(
                        test_id, data.seq_name))
                    success = False
                    break

            if evaluate:
                eval_dir = test_params.eval_dir
                if not eval_dir:
                    eval_dir = load_dir
                eval_path = linux_path(eval_dir, test_params.eval_file)
                acc = tester.eval(load_path, eval_path, eval_dist_type)
                if not acc:
                    seq_logger.error('Tester evaluation failed on sequence {:d} : {:s}'.format(
                        test_id, data.seq_name))
                    success = False
                    break
                if evaluate == 2:
                    tester.accumulative_eval(load_dir, eval_path, seq_logger)

            continue

        if evaluate:
            tester.accumulative_eval(load_dir, eval_path, global_logger)

        return success
