import numpy as np
import os
import sys
import time
import ast

from paramparse import MultiPath

from utilities import compute_overlaps_multi, CrossOverlaps, SelfOverlaps, CustomLogger, MDPStates


class Objects:
    """
    set of labeled objects - abstracts out the common components of detections and annotations

    :ivar _params:
    :type _params: Objects.Params
    :ivar _logger:
    :type _logger: CustomLogger
    """

    class Params:
        def __init__(self, name):
            self.path = ''
            self.src_dir = MultiPath(name)
            self.fix_frame_ids = 1
            self.sort_by_frame_ids = 0
            self.ignore_ioa_thresh = 0.5
            self.allow_missing = 0

            self.help = {
                'path': 'path of the text file in MOT format from where the objects data is to be read;'
                        'if this is empty, then a default path is constructed from the sequence and dataset names',
                'fix_frame_ids': 'convert the frame IDs in the annotations and detections from 1-based '
                                 '(default MOT challenge format) to 0-based that is needed for internal'
                                 ' processing convenience',
                'sort_by_frame_ids': 'sort data by frame IDs',
                'ignored_regions': '1: read ignored_regions from annotations; '
                                   '2: discard the regions after reading'

            }

    def __init__(self, obj_type, params, logger):
        """
        :param str obj_type:
        :param Objects.Params  params:
        :param CustomLogger logger:
        :rtype: None
        """
        self._type = obj_type
        self._params = params
        self._logger = CustomLogger(logger, names=[self._type.lower(), ])
        self.path = self._params.path

        self.ignored_regions = None
        self.data = None
        self.count = 0
        self.orig_n_frames = 0
        self.start_frame_id = 0
        self.end_frame_id = 0
        self.n_frames = 0
        self.idx = None
        # iou and index of the max iuo detection with each annotation or
        # the max iuo annotation with each detection in each frame
        self.max_cross_iou = None
        self.max_cross_iou_idx = None
        # future extensions
        self.features = None

        self._resize_factor = 1

    def initialize(self, orig_n_frames, start_frame_id=-1, end_frame_id=-1):
        """
        :type orig_n_frames: int
        :type start_frame_id: int
        :type end_frame_id: int
        :rtype: None
        """
        self.orig_n_frames = orig_n_frames
        self.start_frame_id = start_frame_id if start_frame_id >= 0 else 0
        self.end_frame_id = end_frame_id if end_frame_id > 0 else self.orig_n_frames - 1
        self.n_frames = self.end_frame_id - self.start_frame_id + 1

    def _sanity_check(self):
        """
        sanity checks
        """

        """frame IDs"""
        if (self.data[:, 0] < 0).any():
            self._logger.error('Negative frame IDs found in data')
            return False
        """scores"""
        invalid_score_ids = np.where(np.logical_or(self.data[:, 6] < 0, self.data[:, 6] > 1))[0]
        if invalid_score_ids.size > 0:
            self._logger.error('Invalid scores outside the range [0, 1] found in data')
            return False
        return True

    def _remove_ignored(self, ignored_regions):
        ioa_1 = np.empty((self.data.shape[0], ignored_regions.shape[0]))
        compute_overlaps_multi(None, ioa_1, None, self.data[:, 2:6], ignored_regions)
        valid_idx = np.flatnonzero(np.apply_along_axis(
            lambda x: np.all(np.less_equal(x, self._params.ignore_ioa_thresh)),
            axis=1, arr=ioa_1))
        n_invalid = self.data.shape[0] - valid_idx.size
        if n_invalid > 0:
            self._logger.info(f'Removing {n_invalid} {self._type} having IOA > {self._params.ignore_ioa_thresh} '
                              f'with {ignored_regions.shape[0]} ignored regions')
            self.data = self.data[valid_idx, ...]

    def _curtail(self):
        if self.start_frame_id > 0 or self.end_frame_id < self.orig_n_frames - 1:
            self._logger.info('Curtailing data to frames {:d} - {:d}'.format(self.start_frame_id,
                                                                             self.end_frame_id))
            valid_idx = np.logical_and(self.data[:, 0] >= self.start_frame_id,
                                       self.data[:, 0] <= self.end_frame_id)
            self.data = self.data[valid_idx, :]
            if self.start_frame_id > 0:
                self.data[:, 0] -= self.start_frame_id

    def _process(self, resize_factor):

        self._resize_factor = resize_factor

        if self._params.sort_by_frame_ids:
            # sort by frame ID
            self._logger.info('sorting by frame IDs')
            self.data = self.data[self.data[:, 0].argsort(kind='mergesort')]

        if self._params.fix_frame_ids:
            # convert frame IDs from 1-based to 0-based
            self._logger.info('converting frame IDs to 0-based')
            self.data[:, 0] -= 1

        if resize_factor != 1:
            self._logger.info('resizing by factor: {}'.format(resize_factor))
            self.data[:, 2:6] *= resize_factor

    def _read(self):
        if not self._params.path:
            raise IOError('Data file path is not provided')

        if not os.path.isfile(self._params.path):
            msg = 'Data file does not exist: {:s}'.format(self._params.path)
            if self._params.allow_missing:
                self._logger.error(msg)
                return False
            else:
                raise IOError(msg)

        self.path = self._params.path
        self._logger.info('Reading from {:s}'.format(self._params.path))
        self.data = np.loadtxt(self._params.path, delimiter=',', ndmin=2)

        return True

    def _build_index(self):
        """
        :rtype: None
        """

        """locations where consecutive frame IDs are not equal
        these are decremented by 1 when compared to the original data vector
        since the vectors being compared are one less in size"""
        end_ids = np.flatnonzero(np.not_equal(self.data[1:, 0], self.data[:-1, 0]))

        self.idx = [None] * self.n_frames
        start_id = 0

        for i in range(end_ids.size):
            frame_id = int(self.data[start_id, 0])
            end_id = end_ids[i] + 1
            self.idx[frame_id] = np.arange(start_id, end_id)
            start_id = end_id
        frame_id = int(self.data[start_id, 0])
        self.idx[frame_id] = np.arange(start_id, self.count)

    def _build_index_slow(self):
        """
        :rtype: None
        """
        # must be stable sorting to ensure that the indices for each object are sorted by frame ID
        frame_sort_idx = np.argsort(self.data[:, 0], kind='mergesort')
        sorted_frame_ids = self.data[frame_sort_idx, 0]
        end_ids = np.flatnonzero(
            np.not_equal(sorted_frame_ids[1:], sorted_frame_ids[:-1]))
        self.idx = [None] * self.n_frames
        start_id = 0
        for i in range(end_ids.size):
            frame_id = int(self.data[frame_sort_idx[start_id], 0])
            end_id = end_ids[i] + 1
            self.idx[frame_id] = frame_sort_idx[start_id:end_id]
            start_id = end_id
        frame_id = int(self.data[frame_sort_idx[start_id], 0])
        self.idx[frame_id] = frame_sort_idx[start_id:self.count]


class Annotations(Objects):
    """
    :type _params: Annotations.Params
    :type n_traj: int
    :type traj_idx: list[np.ndarray]
    :type traj_idx_by_frame: list[dict]
    """

    class Params(Objects.Params):
        """
        :ivar read_ignored_regions: read ignored regions from annotations
        :ivar read_occlusion_status: read occlusion status from annotations
        :ivar overlap_occ: minimum IOU between two annotations for them to be considered occluded;
        only used if read_occlusion_status is 0

        """

        def __init__(self, name='Annotations'):
            Objects.Params.__init__(self, name)
            self.data_dim = 10
            self.read_ignored_regions = 1
            self.read_occlusion_status = 0

            self.overlap_occ = 0.7

            self.remove_unknown_cols = 0
            self.help.update({
                'read_ignored_regions': 'read ignored regions from annotations',
                'read_occlusion_status': 'read occlusion status from annotations',
                'overlap_occ': 'minimum IOU between two annotations for them to be considered occluded; '
                               'only used if read_occlusion_status is 0',

            })

    def __init__(self, params, logger, obj_type='Annotations'):
        """
        :type params: Annotations.Params
        :type logger: CustomLogger
        :rtype: None
        """
        Objects.__init__(self, obj_type, params, logger)
        self._params = params

        """no. of trajectories"""
        self.n_traj = 0

        """indices into data of all annotations in each trajectory"""
        self.traj_idx = None
        self.traj_idx_by_frame = None
        """dict mapping object IDs to trajectory IDs within traj_idx"""
        self.obj_to_traj = None
        """index of occurrence of each frame ID within each trajectory"""
        self.ann_idx = None

        self.max_ioa = None
        # self.occluded = None
        # self.occlusion_ratio = None
        self.occlusion_data = None
        self.scores = None
        self.area_inside_frame = None
        self.obj_sort_idx = None
        self.sorted_obj_ids = None
        self.areas = None
        """x, y coordinates of the bottom right corner of the bounding box"""
        self.br = None
        """intersection-over-union of all annotations in each frame with all other annotations in the same frame"""
        self.self_iou = None
        """intersection-over-union of all annotations in each frame with all detections in the same frame"""
        self.cross_iou = None

        self.cross_overlaps = None
        self._self_overlaps = None

    def _build_trajectory_index(self):
        # print('Annotations :: Building trajectory index...')

        # self.annotations.unique_ids, self.annotations.unique_ids_map = np.unique(
        # self.annotations.data[:, 1], return_inverse=True)

        # must be stable sorting to ensure that the indices for each object are sorted by frame ID
        self.obj_sort_idx = np.argsort(self.data[:, 1], kind='mergesort')
        self.sorted_obj_ids = self.data[self.obj_sort_idx, 1]
        end_ids = list(np.flatnonzero(
            np.not_equal(self.sorted_obj_ids[1:], self.sorted_obj_ids[:-1])))
        end_ids.append(self.count - 1)

        self.n_traj = len(end_ids)

        self.traj_idx = [None] * self.n_traj
        self.traj_idx_by_frame = [None] * self.n_traj
        self.obj_to_traj = {}
        self.traj_to_obj = {}

        start_id = 0
        for traj_id in range(self.n_traj):
            end_id = end_ids[traj_id] + 1
            traj_idx = self.obj_sort_idx[start_id:end_id]
            obj_id = int(self.data[traj_idx[0], 1])
            traj_frame_ids = self.data[traj_idx, 0].astype(np.int32)
            traj_obj_ids = self.data[traj_idx, 1].astype(np.int32)

            """sanity checks"""
            _, frame_counts = np.unique(traj_frame_ids, return_counts=True)
            unique = np.unique(traj_obj_ids, return_counts=False)

            """only one instance of each distinct object in each frame"""
            assert np.all(frame_counts == 1), f"duplicate frame IDs found for object {obj_id}:\n{traj_frame_ids}"
            """all object IDs are identical"""
            assert len(unique) == 1, f"duplicate object IDs found in trajectory {traj_id}:\n{traj_obj_ids}"
            """only one instance of each distinct object ID"""
            assert obj_id not in self.obj_to_traj, f"Duplicate object ID {obj_id} found " \
                f"for trajectories {self.obj_to_traj[obj_id]} and {traj_id}"

            # start_frame_id = int(np.amin(traj_frame_ids))
            # end_frame_id = int(np.amax(traj_frame_ids))

            self.traj_idx[traj_id] = traj_idx
            self.traj_idx_by_frame[traj_id] = {}
            for j, frame_id in enumerate(traj_frame_ids):
                frame_traj_idx = int(np.flatnonzero(np.equal(self.idx[frame_id], traj_idx[j])).item())
                assert traj_idx[j] == self.idx[frame_id][frame_traj_idx], "invalid frame_traj_idx"

                self.traj_idx_by_frame[traj_id][frame_id] = (traj_idx[j], frame_traj_idx)

            self.obj_to_traj[obj_id] = traj_id
            self.traj_to_obj[traj_id] = obj_id

            start_id = end_id

        self._logger.info('n_trajectories: {:d}'.format(self.n_traj))

    def read(self, resize_factor):
        """
        :type build_index: bool
        :type build_trajectory_index: bool
        :rtype: bool
        """
        if not self._read():
            return False

        if self._params.read_ignored_regions:
            ignored_regions_idx = np.logical_and(self.data[:, 0] == -1, self.data[:, 1] == -1)
            _ignored_regions_idx = np.flatnonzero(ignored_regions_idx)
            self.ignored_regions = self.data[_ignored_regions_idx, 2:6]
            _valid_idx = np.flatnonzero(np.logical_not(ignored_regions_idx))
            self.data = self.data[_valid_idx, ...]
            n_ignored_regions = _ignored_regions_idx.size
            if n_ignored_regions:
                self._logger.info('Found {} ignored_regions'.format(n_ignored_regions))
            # if self._params.read_ignored_regions == 2:
            #     self._logger.info('Discarding ignored_regions')
            #     self.ignored_regions = None

        self._process(resize_factor)

        if not self._sanity_check():
            return False

        if self._params.remove_unknown_cols:
            self._logger.info('Removing the amazingly annoying unexplained columns 6 and 7 from data')
            self.data = np.delete(self.data, (6, 7), 1)

        data_dim = self._params.data_dim
        assert self.data.shape[1] >= data_dim, \
            'Data file has incorrect data dimensionality: {:d}. Expected at least: {:d}'.format(
                self.data.shape[1], data_dim)

        if data_dim < 10:
            diff = 10 - data_dim
            self._logger.info(
                'Data file has only {:d} values in each line so padding it with {:d} -1s'.format(
                    self._params.data_dim, diff))
            concat_arr = [
                self.data[:, :data_dim],
                np.tile([-1] * diff, (self.data.shape[0], 1)),
                self.data[:, data_dim:]
            ]
            # if self.data.shape[1] > data_dim:
            #     concat_arr.append(self.data[:, data_dim:])
            self.data = np.concatenate(concat_arr, axis=1)

        if self._params.read_occlusion_status:
            assert self.data.shape[1] > 10, 'Occlusion column is unavailable'

            occlusion_ratio = self.data[:, 10]

            occluded = occlusion_ratio > self._params.overlap_occ
            meta_file_path = self.path.replace('.txt', '.meta')
            self._logger.info(f'Reading occlusion data from {meta_file_path}')
            with open(meta_file_path, 'r') as fid:
                self.occlusion_data = ast.literal_eval(fid.read())

            self.data = np.concatenate((
                self.data,
                occluded.reshape((-1, 1))
            ), axis=1)

        if self.ignored_regions is not None and self.ignored_regions.size > 0:
            self._remove_ignored(self.ignored_regions)

        """curtail data to subsequence"""
        self._curtail()

        self.count = self.data.shape[0]
        if self.count == 0:
            self._logger.error('No objects found')
            return False

        self._logger.info('count: {:d}'.format(self.count))

        # print('Building frame index...'.format(self.type))
        if self._params.sort_by_frame_ids:
            self._build_index()
        else:
            self._build_index_slow()

        """obtain the indices contained in each of the trajectories"""
        self._build_trajectory_index()

        return True

    def get_features(self, detections, n_frames, frame_size):
        """
        :type detections: Detections
        :type n_frames: int
        :type frame_size: tuple(int, int)
        :rtype: bool
        """

        """
        Compute self overlaps between annotations
        """
        # self.logger.info('Computing self overlaps between annotations')
        self._self_overlaps = SelfOverlaps()
        self._self_overlaps.compute(self.data[:, 2:6], self.idx, n_frames)

        self.max_ioa = self._self_overlaps.max_ioa
        self.areas = self._self_overlaps.areas
        self.br = self._self_overlaps.br
        self.self_iou = self._self_overlaps.iou

        if not self._params.read_occlusion_status:
            occluded = np.zeros((self.count,))
            occluded[self.max_ioa > self._params.overlap_occ] = 1
            occlusion_ratio = self.max_ioa

            self.data = np.concatenate((
                self.data,
                occlusion_ratio.reshape((-1, 1)),
                occluded.reshape((-1, 1))
            ), axis=1)

        """'Compute cross overlaps between detections and annotations"""
        # self.logger.info('Computing cross overlaps between detections and annotations')
        self.cross_overlaps = CrossOverlaps()
        self.cross_overlaps.compute(detections.data[:, 2:6], self.data[:, 2:6],
                                    detections.idx, self.idx, self.n_frames)

        # self.cross_iou = self.cross_overlaps.iou
        self.max_cross_iou = self.cross_overlaps.max_iou_2
        self.max_cross_iou_idx = self.cross_overlaps.max_iou_2_idx

        """annotations for which there are no corresponding detections"""
        no_det_idx = np.flatnonzero(self.max_cross_iou_idx == -1)

        self.scores = detections.data[self.max_cross_iou_idx, 6]
        self.scores[no_det_idx] = 0
        """
        compute intersection with the frame to determine the fraction of bounding box lying inside the frame extents
        """
        # n = self.count
        max_iou_det_data = detections.data[self.max_cross_iou_idx, :]
        ul_inter = np.maximum(np.array((1, 1)), max_iou_det_data[:, 2:4])  # n x 2
        br = max_iou_det_data[:, 2:4] + max_iou_det_data[:, 4:6] - 1
        br_inter = np.minimum(np.array(frame_size), br)  # n x 2
        size_inter = br_inter - ul_inter + 1  # n x 2
        size_inter[size_inter < 0] = 0
        area_inter = np.multiply(size_inter[:, 0], size_inter[:, 1])  # n x 1
        areas = np.multiply(max_iou_det_data[:, 4], max_iou_det_data[:, 5])
        self.area_inside_frame = np.divide(area_inter, areas)  # n x 1

        self.area_inside_frame[no_det_idx] = 0

        return True

    def get_mot_metrics(self, track_res, seq_name, dist_type=0):
        """
        :param TrackingResults track_res: tracking result
        :type dist_type: int
        :rtype: pandas.DataFrame, str
        """

        summary = strsummary = None
        # try:
        import evaluation.motmetrics as mm
        # except ImportError as excp:
        #     raise ImportError(excp)
        # self._logger.error(excp)
        # return None, 'MOT evaluator is not available'
        assert self.n_frames == track_res.n_frames, 'MOT data to be compared must have the same number of frames'

        self._logger.info('Accumulating MOT data...')
        start_t = time.time()
        acc = mm.MOTAccumulator(auto_id=True)

        if dist_type == -1:
            return summary, strsummary, acc
        elif dist_type == 0:
            dist_func = mm.distances.iou_matrix
            self._logger.info('Using intersection over union (IoU) distance')
        else:
            dist_func = mm.distances.norm2squared_matrix
            self._logger.info('Using squared Euclidean distance')

        print_diff = int(self.n_frames / 10)

        for frame_id in range(self.n_frames):
            idx1 = self.idx[frame_id]
            idx2 = track_res.idx[frame_id]
            if idx1 is not None:
                bbs_1 = self.data[idx1, 2:6]
                ids_1 = self.data[idx1, 1]
            else:
                bbs_1 = []
                ids_1 = []

            if idx2 is not None:
                bbs_2 = track_res.data[idx2, 2:6]
                ids_2 = track_res.data[idx2, 1]
            else:
                bbs_2 = []
                ids_2 = []

            dist = dist_func(bbs_1, bbs_2)
            acc.update(ids_1, ids_2, dist)
            if print_diff > 0 and (frame_id + 1) % print_diff == 0:
                # print('Done {:d}/{:d} frames'.format(frame_id + 1, self.n_frames))
                # sys.stdout.write("\033[F")
                sys.stdout.write('\rProcessed {:d}/{:d} frames'.format(
                    frame_id + 1, self.n_frames))
                sys.stdout.flush()
        sys.stdout.write('\rProcessed {:d}/{:d} frames\n'.format(self.n_frames, self.n_frames))
        sys.stdout.flush()
        end_t = time.time()
        fps = self.n_frames / (end_t - start_t)
        self._logger.info('FPS: {:.3f}'.format(fps))

        self._logger.info('Computing MOT metrics...')
        start_t = time.time()
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=seq_name)
        end_t = time.time()
        fps = self.n_frames / (end_t - start_t)
        self._logger.info('FPS: {:.3f}'.format(fps))

        summary = summary.rename(columns=mm.io.motchallenge_metric_names)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters
        )
        return summary, strsummary, acc



class TrackingResults(Annotations):
    """
    :type _params: TrackingResults.Params
    """

    class Params(Objects.Params):
        def __init__(self):
            Objects.Params.__init__(self, 'TrackingResults')
            self.allow_debug = 2
            self.help.update({
                'allow_debug': 'read debugging data from tracking results including MDP state and trajectory validity'
            })

    def __init__(self, params, logger, obj_type='TrackingResults'):
        Annotations.__init__(self, params, logger, obj_type)
        """
        :type params: TrackingResults.Params
        :type logger: CustomLogger
        :rtype: None
        """
        self._params = params

    def read(self, resize_factor):
        """
        :type build_index: bool
        :type build_trajectory_index: bool
        :rtype: bool
        """
        if not self._read():
            return False

        self._process(resize_factor)

        if not self._sanity_check():
            return False

        n_data_cols = self.data.shape[1]

        if n_data_cols != 10:
            self._logger.error('Data file has incorrect data dimensionality: {:d}'.format(
                self.data.shape[1]))
            return False

        """curtail data to subsequence"""
        self._curtail()

        self.count = self.data.shape[0]
        if self.count == 0:
            self._logger.error('No objects found')
            return False

        self._logger.info('count: {:d}'.format(self.count))

        if self._params.sort_by_frame_ids:
            self._build_index()
        else:
            self._build_index_slow()

        self._build_trajectory_index()

        return True
