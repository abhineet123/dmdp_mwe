import copy

from data import Data
from objects import Annotations, TrackingResults
from utilities import linux_path, CustomLogger


class Input:
    class Params:

        def __init__(self):
            self.path = ''
            self.frame_ids = (-1, -1)

            self.db_root_path = 'data'
            self.source_type = 0

            self.annotations = Annotations.Params()
            self.tracking_res = TrackingResults.Params()

    def __init__(self, params, logger):
        """
        :type params: Input.Params
        :type logger: CustomLogger
        :rtype: None
        """

        self.params = copy.deepcopy(params)
        self._logger = logger

        self.source_path = None

        self.annotations = None
        self.tracking_res = None

        self.start_frame_id, self.end_frame_id = self.params.frame_ids
        self.seq_name = None
        self.seq_set = None
        self.seq_n_frames = 0
        self.n_frames = 0

        self.is_initialized = False

    def initialize(self, data, logger=None):
        """
        :type data: Data
        :type read_img: bool | int
        :type logger: CustomLogger | logging.RootLogger
        :rtype: bool
        """

        if logger is not None:
            self._logger = logger

        self.annotations = None
        self.tracking_res = None

        self.start_frame_id, self.end_frame_id = self.params.frame_ids

        if not data.is_initialized:
            self._logger.error('Source path must be provided with uninitialized data module')
            return False
        self.seq_name, self.seq_set, self.seq_n_frames = data.seq_name, data.seq_set, data.seq_n_frames

        if self.start_frame_id < 0:
            self.start_frame_id = data.start_frame_id
        if self.end_frame_id < 0:
            self.end_frame_id = data.end_frame_id
        self.n_frames = self.end_frame_id - self.start_frame_id + 1

        self._logger.info('seq_set: {:s}'.format(self.seq_set))
        self._logger.info('seq_name: {:s}'.format(self.seq_name))
        self._logger.info('seq_n_frames: {:d}'.format(self.seq_n_frames))
        self._logger.info('start_frame_id: {:d}'.format(self.start_frame_id))
        self._logger.info('end_frame_id: {:d}'.format(self.end_frame_id))
        self._logger.info('n_frames: {:d}'.format(self.n_frames))

        return True

    def read_annotations(self):
        """
        :rtype: bool
        """
        annotations_params = copy.deepcopy(self.params.annotations)
        if not annotations_params.path:
            annotations_params.path = linux_path(self.params.db_root_path, self.seq_set,
                                                 annotations_params.src_dir, self.seq_name + '.txt')

        self.annotations = Annotations(annotations_params, self._logger)
        self.annotations.initialize(self.seq_n_frames, self.start_frame_id, self.end_frame_id)

        if not self.annotations.read(1):
            self._logger.error('Failed to read annotations')
            self.annotations = None
            return False

        return True

    def read_tracking_results(self, res_path):
        """
        :type res_path: str
        :rtype: bool
        """
        self.params.tracking_res.path = res_path
        self.tracking_res = TrackingResults(self.params.tracking_res, self._logger)
        self.tracking_res.initialize(self.seq_n_frames,
                                     self.start_frame_id, self.end_frame_id)
        if not self.tracking_res.read(1):
            self._logger.error('Failed to read tracking results')
            self.tracking_res = None
            return False
        return True
