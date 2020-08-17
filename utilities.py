import numpy as np
import os
import copy
import time
from pprint import pformat
import functools
import logging
from io import StringIO
from contextlib import contextmanager
from datetime import datetime
from colorlog import ColoredFormatter

logging.getLogger('matplotlib').setLevel(logging.ERROR)


class MDPStates:
    inactive, active, tracked, lost = range(4)
    to_str = {
        0: 'inactive',
        1: 'active',
        2: 'tracked',
        3: 'lost',
    }

class CustomLogger:
    """
    :type _backend: logging.RootLogger | logging.logger

    """

    def __init__(self, logger, names, key='custom_module'):
        """
        modify the custom module name header to append one or more names

        :param CustomLogger | logging.RootLogger logger:
        :param tuple | list names:
        """
        try:
            self._backend = logger.get_backend()
        except AttributeError:
            self._backend = logger

        self.handlers = self._backend.handlers
        self.addHandler = self._backend.addHandler
        self.removeHandler = self._backend.removeHandler

        try:
            k = logger.info.keywords['extra'][key]
        except BaseException as e:
            custom_log_header_str = '{}'.format(':'.join(names))
        else:
            custom_log_header_str = '{}:{}'.format(k, ':'.join(names))

        self.custom_log_header_tokens = custom_log_header_str.split(':')

        try:
            custom_log_header = copy.deepcopy(logger.info.keywords['extra'])
        except BaseException as e:
            custom_log_header = {}

        custom_log_header.update({key: custom_log_header_str})

        self.info = functools.partial(self._backend.info, extra=custom_log_header)
        self.warning = functools.partial(self._backend.warning, extra=custom_log_header)
        self.debug = functools.partial(self._backend.debug, extra=custom_log_header)
        self.error = functools.partial(self._backend.error, extra=custom_log_header)

    def get_backend(self):
        return self._backend

    @staticmethod
    def add_file_handler(log_dir, _prefix, logger):
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_file = linux_path(log_dir, '{}_{}.log'.format(_prefix, time_stamp))
        logging_handler = logging.FileHandler(log_file)
        logger.addHandler(logging_handler)
        logging_fmt = logging.Formatter(
            '%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s :  %(message)s',
        )

        logger.handlers[-1].setFormatter(logging_fmt)
        return log_file, logging_handler

    @staticmethod
    def add_string_handler(logger):
        log_stream = StringIO()
        logging_handler = logging.StreamHandler(log_stream)
        # logging_handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(logging_handler)
        # logger.string_stream = log_stream
        return logging_handler

    @staticmethod
    def remove_file_handler(logging_handler, logger):
        if logging_handler not in logger.handlers:
            return
        logging_handler.close()
        logger.removeHandler(logging_handler)

    @staticmethod
    def setup():
        PROFILE_LEVEL_NUM = 9

        def profile(self, message, *args, **kws):
            if self.isEnabledFor(PROFILE_LEVEL_NUM):
                self._log(PROFILE_LEVEL_NUM, message, args, **kws)

        logging.addLevelName(PROFILE_LEVEL_NUM, "PROFILE")
        logging.Logger.profile = profile
        # logging.getLogger().addHandler(ColorHandler())

        # logging_level = logging.DEBUG
        # logging_level = PROFILE_LEVEL_NUM
        # logging.basicConfig(level=logging_level, format=logging_fmt)

        colored_logging_fmt = ColoredFormatter(
            '%(header_log_color)s%(custom_header)s:%(log_color)s%(custom_module)s:%(funcName)s:%(lineno)s :  %(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'header': {
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'WARNING': 'white',
                    'ERROR': 'white',
                    'CRITICAL': 'white',
                }
            },
            style='%'
        )

        nocolor_logging_fmt = logging.Formatter(
            '%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s  :::  %(message)s',
        )
        # logging_fmt = logging.Formatter('%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s :  %(message)s')
        # logging_fmt = logging.Formatter('%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s')
        logging_level = logging.NOTSET
        logging.basicConfig(level=logging_level, format=colored_logging_fmt)
        _logger = logging.getLogger()
        CustomLogger.add_string_handler(_logger)

        _logger.setLevel(logging_level)

        _logger.handlers[0].setFormatter(colored_logging_fmt)
        _logger.handlers[1].setFormatter(nocolor_logging_fmt)

        class ContextFilter(logging.Filter):
            def filter(self, record):

                if not hasattr(record, 'custom_module'):
                    record.custom_module = record.module

                if not hasattr(record, 'custom_header'):
                    record.custom_header = record.levelname

                return True

        f = ContextFilter()
        _logger.addFilter(f)
        return _logger


@contextmanager
def profile(_id, _times, _rel_times, enable=1):
    """

    :param _id:
    :param dict _times:
    :param int enable:
    :return:
    """
    if not enable:
        yield None

    else:
        start_t = time.time()
        yield None
        end_t = time.time()
        _time = end_t - start_t

        print(f'{_id} :: {_time}')

        if _times is not None:

            _times[_id] = _time

            total_time = np.sum(list(_times.values()))

            if _rel_times is not None:

                for __id in _times:
                    rel__time = _times[__id] / total_time
                    _rel_times[__id] = rel__time

                rel_times = [(k, v) for k, v in sorted(_rel_times.items(), key=lambda item: item[1])]

                print(f'rel_times:\n {pformat(rel_times)}')

# overlaps between two sets of labeled objects, typically the annotations and the detections
class CrossOverlaps:
    """
    :type iou: list[np.ndarray]
    :type ioa_1: list[np.ndarray]
    :type ioa_2: list[np.ndarray]
    :type max_iou_1: np.ndarray
    :type max_iou_1_idx: np.ndarray
    :type max_iou_2: np.ndarray
    :type max_iou_2_idx: np.ndarray
    """

    def __init__(self):
        # intersection over union
        self.iou = None
        # intersection over area of object 1
        self.ioa_1 = None
        # intersection over area of object 2
        self.ioa_2 = None
        # max iou of each object in first set over all objects in second set from the same frame
        self.max_iou_1 = None
        # index of the object in the second set that corresponds to the maximum iou
        self.max_iou_1_idx = None
        # max iou of each object in second set over all objects in first set from the same frame
        self.max_iou_2 = None
        # index of the object in the first set that corresponds to the maximum iou
        self.max_iou_2_idx = None

    def compute(self, objects_1, objects_2, index_1, index_2, n_frames):
        """
        :type objects_1: np.ndarray
        :type objects_2: np.ndarray
        :type index_1: list[np.ndarray]
        :type index_2: list[np.ndarray]
        :type n_frames: int
        :rtype: None
        """
        # for each frame, contains a matrix that stores the overlap between each pair of
        # annotations and detections in that frame
        self.iou = [None] * n_frames
        self.ioa_1 = [None] * n_frames
        self.ioa_2 = [None] * n_frames

        self.max_iou_1 = np.zeros((objects_1.shape[0],))
        self.max_iou_2 = np.zeros((objects_2.shape[0],))

        self.max_iou_1_idx = np.full((objects_1.shape[0],), -1, dtype=np.int32)
        self.max_iou_2_idx = np.full((objects_2.shape[0],), -1, dtype=np.int32)

        for frame_id in range(n_frames):
            idx1 = index_1[frame_id]
            idx2 = index_2[frame_id]

            if idx1 is None or idx2 is None:
                continue

            boxes_1 = objects_1[idx1, :]
            n1 = boxes_1.shape[0]
            ul_1 = boxes_1[:, :2]  # n1 x 2
            size_1 = boxes_1[:, 2:]  # n1 x 2
            br_1 = ul_1 + size_1 - 1  # n1 x 2
            area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1

            boxes_2 = objects_2[idx2, :]
            n2 = boxes_2.shape[0]
            ul_2 = boxes_2[:, :2]  # n2 x 2
            size_2 = boxes_2[:, 2:]  # n2 x 2
            br_2 = ul_2 + size_2 - 1  # n2 x 2
            area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1

            ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            ul_inter = np.maximum(ul_1_rep, ul_2_rep)  # n2 x 2 x n1

            # box size is defined in terms of  no. of pixels
            br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            br_inter = np.minimum(br_1_rep, br_2_rep)  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

            size_inter = br_inter - ul_inter + 1  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            size_inter[size_inter < 0] = 0  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1]).reshape((n1, n2))

            area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_union = area_1_rep + area_2_rep - area_inter  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)

            # self.iou[frame_id] = np.divide(area_inter, area_union).reshape((n1, n2), order='F')  # n1 x n2
            # self.ioa_1[frame_id] = np.divide(area_inter, area_1_rep).reshape((n1, n2), order='F')  # n1 x n2
            # self.ioa_2[frame_id] = np.divide(area_inter, area_2_rep).reshape((n1, n2), order='F')  # n1 x n2

            self.iou[frame_id] = np.divide(area_inter, area_union)  # n1 x n2
            self.ioa_1[frame_id] = np.divide(area_inter, area_1_rep)  # n1 x n2
            self.ioa_2[frame_id] = np.divide(area_inter, area_2_rep)  # n1 x n2

            max_idx_1 = np.argmax(self.iou[frame_id], axis=1)
            max_idx_2 = np.argmax(self.iou[frame_id], axis=0).transpose()

            self.max_iou_1[idx1] = self.iou[frame_id][np.arange(n1), max_idx_1]
            self.max_iou_2[idx2] = self.iou[frame_id][max_idx_2, np.arange(n2)]

            # indices wrt the overall object arrays rather than their frame-wise subsets
            self.max_iou_1_idx[idx1] = idx2[max_idx_1]
            self.max_iou_2_idx[idx2] = idx1[max_idx_2]


# overlaps between each labeled object in a set with all other objects in that set from the same frame
class SelfOverlaps:
    """
    :type iou: np.ndarray
    :type ioa: np.ndarray
    :type max_iou: np.ndarray
    :type max_ioa: np.ndarray
    """

    def __init__(self):
        # intersection over union
        self.iou = None
        # intersection over area of object
        self.ioa = None
        # max iou of each object over all other objects from the same frame
        self.max_iou = None
        # max ioa of each object over all other objects from the same frame
        self.max_ioa = None

        self.br = None
        self.areas = None

    def compute(self, objects, index, n_frames):
        """
        :type objects: np.ndarray
        :type index: list[np.ndarray]
        :type n_frames: int
        :rtype: None
        """
        self.iou = [None] * n_frames
        self.ioa = [None] * n_frames

        self.max_ioa = np.zeros((objects.shape[0],))
        self.areas = np.zeros((objects.shape[0],))
        self.br = np.zeros((objects.shape[0], 2))

        for frame_id in range(n_frames):
            if index[frame_id] is None:
                continue

            end_id = index[frame_id]
            boxes = objects[index[frame_id], :]

            n = boxes.shape[0]

            ul = boxes[:, :2]  # n x 2
            ul_rep = np.tile(np.reshape(ul, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
            ul_2_rep = np.tile(np.reshape(ul, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
            ul_inter = np.maximum(ul_rep, ul_2_rep)  # n x 2 x n

            size = boxes[:, 2:]  # n1 x 2
            br = ul + size - 1  # n x 2

            # size_ = boxes[:, 2:]  # n x 2
            # br = ul + size_ - 1  # n x 2
            br_rep = np.tile(np.reshape(br, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
            br_2_rep = np.tile(np.reshape(br, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
            br_inter = np.minimum(br_rep, br_2_rep)  # n x 2 x n

            size_inter = br_inter - ul_inter + 1  # np(n x n x 2) -> std(n x 2 x n)
            size_inter[size_inter < 0] = 0
            # np(n x n x 1) -> std(n x 1 x n)
            area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

            area = np.multiply(size[:, 0], size[:, 1]).reshape((n, 1))  # n1 x 1
            # area = np.multiply(size_[:, :, 0], size_[:, :, 1])  # n x 1
            area_rep = np.tile(area, (1, n))  # np(n x n x 1) -> std(n x 1 x n)
            area_2_rep = np.tile(area.transpose(), (n, 1))  # np(n x n x 1) -> std(n x 1 x n)
            area_union = area_rep + area_2_rep - area_inter  # np(n x n x 1) -> std(n x 1 x n)

            # self.iou[frame_id] = np.divide(area_inter, area_union).reshape((n, n), order='F')  # n x n
            # self.ioa[frame_id] = np.divide(area_inter, area_rep).reshape((n, n), order='F')  # n x n

            self.iou[frame_id] = np.divide(area_inter, area_union)  # n x n
            self.ioa[frame_id] = np.divide(area_inter, area_rep)  # n x n

            # set box overlap with itself to 0
            idx = np.arange(n)
            self.ioa[frame_id][idx, idx] = 0
            self.iou[frame_id][idx, idx] = 0

            for i in range(n):
                invalid_idx = np.flatnonzero(np.greater(br[i, 1], br[:, 1]))
                self.ioa[frame_id][i, invalid_idx] = 0

            self.max_ioa[index[frame_id]] = np.amax(self.ioa[frame_id], axis=1)

            self.areas[index[frame_id]] = area.reshape((n,))
            self.br[index[frame_id], :] = br


def compute_overlaps_multi(iou, ioa_1, ioa_2, objects_1, objects_2, logger=None):
    """

    compute overlap between each pair of objects in two sets of objects
    can be used for computing overlap between all detections and annotations in a frame

    :type iou: np.ndarray | None
    :type ioa_1: np.ndarray | None
    :type ioa_2: np.ndarray | None
    :type object_1: np.ndarray
    :type objects_2: np.ndarray
    :type logger: logging.RootLogger | None
    :rtype: None
    """
    # handle annoying singletons
    if len(objects_1.shape) == 1:
        objects_1 = objects_1.reshape((1, 4))

    if len(objects_2.shape) == 1:
        objects_2 = objects_2.reshape((1, 4))

    n1 = objects_1.shape[0]
    n2 = objects_2.shape[0]

    ul_1 = objects_1[:, :2]  # n1 x 2
    ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    ul_2 = objects_2[:, :2]  # n2 x 2
    ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_1 = objects_1[:, 2:]  # n1 x 2
    size_2 = objects_2[:, 2:]  # n2 x 2

    # if logger is not None:
    #     logger.debug('objects_1.shape: %(1)s', {'1': objects_1.shape})
    #     logger.debug('objects_2.shape: %(1)s', {'1': objects_2.shape})
    #     logger.debug('objects_1: %(1)s', {'1': objects_1})
    #     logger.debug('objects_2: %(1)s', {'1': objects_2})
    #     logger.debug('ul_1: %(1)s', {'1': ul_1})
    #     logger.debug('ul_2: %(1)s', {'1': ul_2})
    #     logger.debug('size_1: %(1)s', {'1': size_1})
    #     logger.debug('size_2: %(1)s', {'1': size_2})

    br_1 = ul_1 + size_1 - 1  # n1 x 2
    br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    br_2 = ul_2 + size_2 - 1  # n2 x 2
    br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_inter = np.minimum(br_1_rep, br_2_rep) - np.maximum(ul_1_rep, ul_2_rep) + 1  # n2 x 2 x n1
    size_inter[size_inter < 0] = 0
    # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

    area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1
    area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1
    area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_union = area_1_rep + area_2_rep - area_inter  # n2 x 1 x n1

    if iou is not None:
        # write('iou.shape: {}\n'.format(iou.shape))
        # write('area_inter.shape: {}\n'.format(area_inter.shape))
        # write('area_union.shape: {}\n'.format(area_union.shape))
        iou[:] = np.divide(area_inter, area_union)  # n1 x n2
    if ioa_1 is not None:
        ioa_1[:] = np.divide(area_inter, area_1_rep)  # n1 x n2
    if ioa_2 is not None:
        ioa_2[:] = np.divide(area_inter, area_2_rep)  # n1 x n2

def parse_seq_IDs(ids):
    out_ids = []
    if isinstance(ids, int):
        out_ids.append(ids)
    else:
        for _id in ids:
            if isinstance(_id, list):
                if len(_id) == 1:
                    out_ids.extend(range(_id[0]))
                if len(_id) == 2:
                    out_ids.extend(range(_id[0], _id[1]))
                elif len(_id) == 3:
                    out_ids.extend(range(_id[0], _id[1], _id[2]))
            else:
                out_ids.append(_id)
    return tuple(out_ids)


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')

def combined_motmetrics(acc_dict, logger):
    # logger.info(f'Computing overall MOT metrics over {len(acc_dict)} sequences...')
    # start_t = time.time()
    try:
        import evaluation.motmetrics as mm
    except ImportError as excp:
        logger.error('MOT evaluator is not available: {}'.format(excp))
        return False
    seq_names, accs = map(list, zip(*acc_dict.items()))

    # logger.info(f'Merging accumulators...')
    accs = mm.MOTAccumulator.merge_event_dataframes(accs)

    # logger.info(f'Computing metrics...')
    mh = mm.metrics.create()
    summary = mh.compute(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        name='OVERALL',
    )
    # end_t = time.time()
    # logger.info('Time taken: {:.3f}'.format(end_t - start_t))

    summary = summary.rename(columns=mm.io.motchallenge_metric_names)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters
    )
    print(strsummary)

    return summary, strsummary

def motmetrics_to_file(eval_paths, summary, load_fname, seq_name,
                       mode='a', time_stamp='', verbose=1):
    """

    :param eval_paths:
    :param summary:
    :param load_fname:
    :param seq_name:
    :param logger:
    :param mode:
    :param combined_accuracies:
    :return:
    """
    if not time_stamp:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    for eval_path in eval_paths:
        if verbose:
            print(f'{eval_path}')

        write_header = False
        if not os.path.isfile(eval_path):
            write_header = True

        with open(eval_path, mode) as eval_fid:
            if write_header:
                eval_fid.write('{:<50}'.format('timestamp'))
                eval_fid.write('\t{:<50}'.format('file'))
                for _metric, _type in zip(summary.columns.values, summary.dtypes):
                    if _type == np.int64:
                        eval_fid.write('\t{:>6}'.format(_metric))
                    else:
                        eval_fid.write('\t{:>8}'.format(_metric))
                eval_fid.write('\t{:>10}'.format('MT(%)'))
                eval_fid.write('\t{:>10}'.format('ML(%)'))
                eval_fid.write('\t{:>10}'.format('PT(%)'))

                eval_fid.write('\n')
            eval_fid.write('{:13s}'.format(time_stamp))
            eval_fid.write('\t{:50s}'.format(load_fname))
            _values = summary.loc[seq_name].values
            # if seq_name == 'OVERALL':
            #     if verbose:
            #         print()

            for _val, _type in zip(_values, summary.dtypes):
                if _type == np.int64:
                    eval_fid.write('\t{:6d}'.format(int(_val)))
                else:
                    eval_fid.write('\t{:.6f}'.format(_val))
            try:
                _gt = float(summary['GT'][seq_name])
            except KeyError:
                pass
            else:
                mt_percent = float(summary['MT'][seq_name]) / _gt * 100.0
                ml_percent = float(summary['ML'][seq_name]) / _gt * 100.0
                pt_percent = float(summary['PT'][seq_name]) / _gt * 100.0
                eval_fid.write('\t{:3.6f}\t{:3.6f}\t{:3.6f}'.format(
                    mt_percent, ml_percent, pt_percent))

            eval_fid.write('\n')


def add_suffix(src_path, suffix):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name, src_ext = os.path.splitext(os.path.basename(src_path))
    dst_path = os.path.join(src_dir, src_name + '_' + suffix + src_ext)
    return dst_path

