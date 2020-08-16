import paramparse
from utilities import CustomLogger


class Data:
    """
    :type params: Data.Params
    """

    class Params:
        """
        :ivar ratios: 'two element tuple to indicate fraction of frames in each sequence on which'
                  ' (training, testing) is to be performed; '
                  'negative values mean that frames are taken from the end of the sequence; '
                  'zero for the second entry means that all frames not used for training are used for '
                  'testing in each sequence; '
                  'if either entry is > 1, it is set to the corresponding value for the sequence set being used',
        :ivar offsets: 'two element tuple to indicate offsets in the start frame ID with respect to the sub sequence'
                   ' obtained from the (train, test) ratios on which (training, testing) is to be performed;'
                   'ratios and offsets together specify the subsequences, if any, on which the two components'
                   ' of the program are to be run',
        :ivar ratios_gram: 'train and test ratios for sequences in the GRAM dataset',
        :ivar ratios_idot: 'train and test ratios for sequences in the IDOT dataset',
        :ivar ratios_detrac: 'train and test ratios for sequences in the DETRAC dataset',
        :ivar ratios_lost: 'train and test ratios for sequences in the LOST dataset',
        :ivar ratios_isl: 'train and test ratios for sequences in the ISL dataset',
        :ivar ratios_mot2015: 'train and test ratios for sequences in the MOT2015 dataset',
        :ivar ratios_kitti: 'train and test ratios for sequences in the KITTI dataset',

        """

        def __init__(self):
            self.ratios = [1., 1.]
            self.offsets = [0., 0.]

            self.ratios_gram = [1., 0.]
            self.ratios_idot = [1., 0.]
            self.ratios_detrac = [1., 0.]
            self.ratios_lost = [1., 0.]
            self.ratios_isl = [1., 1.]
            self.ratios_mot2015 = [1., 0.]
            self.ratios_mot2017 = [1., 0.]
            self.ratios_mot2017_sdp = [1., 0.]
            self.ratios_mot2017_dpm = [1., 0.]
            self.ratios_kitti = [1., 0.]
            self.ratios_mnist_mot = [1., 1.]

        def synchronize(self, _id=0):
            attrs = paramparse.get_valid_members(self)
            for attr in attrs:
                if attr == 'tee_log':
                    continue
                attr_val = list(getattr(self, attr))
                attr_val[1 - _id] = attr_val[_id]
                setattr(self, attr, attr_val)

    def __init__(self, params, logger):
        """
        :type params: Data.Params
        :type logger: logging.RootLogger | CustomLogger
        :rtype: None
        """
        self.params = params
        self._logger = logger
        self.__logger = logger

        self.sets, self.sequences, self.ratios = self.get_sequences()

        self.seq_set = None
        self.seq_name = None
        self.seq_n_frames = 0
        # self.seq_ratios = list(self.params.ratios)

        self.start_frame_id = 0
        self.end_frame_id = 0

        self.is_initialized = False

    def initialize(self, seq_set_id, seq_id, seq_type_id, logger=None):
        """
        :type seq_set_id: int
        :type seq_id: int
        :type seq_type_id: int
        :type logger: CustomLogger
        :rtype: bool
        """

        if logger is not None:
            self.__logger = logger

        if seq_set_id < 0 or seq_id < 0:
            self._logger.info('Using external sequence')
            return

        self.seq_set = self.sets[seq_set_id]
        self.seq_name = self.sequences[self.seq_set][seq_id][0]
        self.seq_n_frames = self.sequences[self.seq_set][seq_id][1]

        self._logger = CustomLogger(self.__logger, names=(self.seq_name,), key='custom_header')

        seq_ratios = self.ratios[self.seq_set][seq_id]

        start_offset = self.params.offsets[seq_type_id]

        if seq_type_id == 0:
            seq_type = 'training'
            seq_ratio = seq_ratios[0]
        else:
            seq_type = 'testing'
            if seq_ratios[1] == 0:
                """
                test on all non-training frames
                """
                if seq_ratios[0] < 0:
                    """training samples from end"""
                    seq_ratio = seq_ratios[0] + 1
                else:
                    seq_ratio = seq_ratios[0] - 1
            else:
                seq_ratio = seq_ratios[1]

        self.start_frame_id, self.end_frame_id = self.get_sub_seq_idx(
            seq_ratio, start_offset, self.seq_n_frames)

        if self.seq_n_frames <= self.start_frame_id or \
                self.start_frame_id < 0 or \
                self.seq_n_frames <= self.end_frame_id or \
                self.end_frame_id <= 0 or \
                self.end_frame_id < self.start_frame_id:
            self._logger.error('Invalid {:s} ratio: {:.2f} or start frame id: {:d} '
                               'specified'.format(seq_type, seq_ratio, start_offset,
                                                  self.seq_n_frames))
            return False

        self._logger.info('seq_ratios: {}'.format(seq_ratios))
        self._logger.info('seq_ratio: {:f}'.format(seq_ratio))
        self._logger.info('start_offset: {:d}'.format(start_offset))
        self.is_initialized = True

        return True

    def get_sub_seq_idx(self, seq_ratio, start_offset, n_frames):
        if seq_ratio < 0:
            """
            sample from end
            """
            start_idx = int(n_frames * (1 + seq_ratio)) - start_offset
            end_idx = int(round(n_frames - start_offset - 1))
        else:
            start_idx = int(start_offset)
            end_idx = int(round(n_frames * seq_ratio)) + start_offset - 1
        if start_idx < 0:
            start_idx = 0
        if end_idx >= n_frames:
            end_idx = n_frames - 1
        return start_idx, end_idx

    def get_inv_sub_seq_idx(self, sub_seq_ratio, start_offset, n_frames):
        if sub_seq_ratio < 0:
            inv_sub_seq_ratio = sub_seq_ratio + 1
        else:
            inv_sub_seq_ratio = sub_seq_ratio - 1
        return self.get_sub_seq_idx(inv_sub_seq_ratio, start_offset, n_frames)

    def combine_sequences(self, seq_lists):
        combined_sequences = {}
        offset = 0
        for sequences in seq_lists:
            for key in sequences:
                combined_sequences[key + offset] = sequences[key]
            offset += len(sequences)
        return combined_sequences

    def get_sequences(self):
        # names of sequences and the no. of frames in each
        sequences_mot2015 = {
            # train
            0: ('ADL-Rundle-6', 525),
            1: ('ADL-Rundle-8', 654),
            2: ('ETH-Bahnhof', 1000),
            3: ('ETH-Pedcross2', 837),
            4: ('ETH-Sunnyday', 354),
            5: ('KITTI-13', 340),
            6: ('KITTI-17', 145),
            7: ('PETS09-S2L1', 795),
            8: ('TUD-Campus', 71),
            9: ('TUD-Stadtmitte', 179),
            10: ('Venice-2', 600),
            # test
            11: ('ADL-Rundle-1', 500),
            12: ('ADL-Rundle-3', 625),
            13: ('AVG-TownCentre', 450),
            14: ('ETH-Crossing', 219),
            15: ('ETH-Jelmoli', 440),
            16: ('ETH-Linthescher', 1194),
            17: ('KITTI-16', 209),
            18: ('KITTI-19', 1059),
            19: ('PETS09-S2L2', 436),
            20: ('TUD-Crossing', 201),
            21: ('Venice-1', 450),
        }
        sequences_mot2017 = {
            # train
            0: ('MOT17-02-FRCNN', 600),
            1: ('MOT17-04-FRCNN', 1050),
            2: ('MOT17-05-FRCNN', 837),
            3: ('MOT17-09-FRCNN', 525),
            4: ('MOT17-10-FRCNN', 654),
            5: ('MOT17-11-FRCNN', 900),
            6: ('MOT17-13-FRCNN', 750),
            # test
            7: ('MOT17-01-FRCNN', 450),
            8: ('MOT17-03-FRCNN', 1500),
            9: ('MOT17-06-FRCNN', 1194),
            10: ('MOT17-07-FRCNN', 500),
            11: ('MOT17-08-FRCNN', 625),
            12: ('MOT17-12-FRCNN', 900),
            13: ('MOT17-14-FRCNN', 750),
        }
        sequences_mot2017_sdp = {
            # train
            0: ('MOT17-02-SDP', 600),
            1: ('MOT17-04-SDP', 1050),
            2: ('MOT17-05-SDP', 837),
            3: ('MOT17-09-SDP', 525),
            4: ('MOT17-10-SDP', 654),
            5: ('MOT17-11-SDP', 900),
            6: ('MOT17-13-SDP', 750),
            # test
            7: ('MOT17-01-SDP', 450),
            8: ('MOT17-03-SDP', 1500),
            9: ('MOT17-06-SDP', 1194),
            10: ('MOT17-07-SDP', 500),
            11: ('MOT17-08-SDP', 625),
            12: ('MOT17-12-SDP', 900),
            13: ('MOT17-14-SDP', 750),
        }

        sequences_mot2017_dpm = {
            # train
            0: ('MOT17-02-DPM', 600),
            1: ('MOT17-04-DPM', 1050),
            2: ('MOT17-05-DPM', 837),
            3: ('MOT17-09-DPM', 525),
            4: ('MOT17-10-DPM', 654),
            5: ('MOT17-11-DPM', 900),
            6: ('MOT17-13-DPM', 750),
            # test
            7: ('MOT17-01-DPM', 450),
            8: ('MOT17-03-DPM', 1500),
            9: ('MOT17-06-DPM', 1194),
            10: ('MOT17-07-DPM', 500),
            11: ('MOT17-08-DPM', 625),
            12: ('MOT17-12-DPM', 900),
            13: ('MOT17-14-DPM', 750),
        }

        sequences_kitti = {
            0: ('train_0000', 154),
            1: ('train_0001', 447),
            2: ('train_0002', 233),
            3: ('train_0003', 144),
            4: ('train_0004', 314),
            5: ('train_0005', 297),
            6: ('train_0006', 270),
            7: ('train_0007', 800),
            8: ('train_0008', 390),
            9: ('train_0009', 803),
            10: ('train_0010', 294),
            11: ('train_0011', 373),
            12: ('train_0012', 78),
            13: ('train_0013', 340),
            14: ('train_0014', 106),
            15: ('train_0015', 376),
            16: ('train_0016', 209),
            17: ('train_0017', 145),
            18: ('train_0018', 339),
            19: ('train_0019', 1059),
            20: ('train_0020', 837),
            21: ('test_0000', 465),
            22: ('test_0001', 147),
            23: ('test_0002', 243),
            24: ('test_0003', 257),
            25: ('test_0004', 421),
            26: ('test_0005', 809),
            27: ('test_0006', 114),
            28: ('test_0007', 215),
            29: ('test_0008', 165),
            30: ('test_0009', 349),
            31: ('test_0010', 1176),
            32: ('test_0011', 774),
            33: ('test_0012', 694),
            34: ('test_0013', 152),
            35: ('test_0014', 850),
            36: ('test_0015', 701),
            37: ('test_0016', 510),
            38: ('test_0017', 305),
            39: ('test_0018', 180),
            40: ('test_0019', 404),
            41: ('test_0020', 173),
            42: ('test_0021', 203),
            43: ('test_0022', 436),
            44: ('test_0023', 430),
            45: ('test_0024', 316),
            46: ('test_0025', 176),
            47: ('test_0026', 170),
            48: ('test_0027', 85),
            49: ('test_0028', 175)
        }
        sequences_gram = {
            0: ('M-30', 7520),
            1: ('M-30-HD', 9390),
            2: ('Urban1', 23435),
            # 3: ('M-30-Large', 7520),
            # 4: ('M-30-HD-Small', 9390)
        }
        sequences_idot = {
            0: ('idot_1_intersection_city_day', 8991),
            1: ('idot_2_intersection_suburbs', 8990),
            2: ('idot_3_highway', 8981),
            3: ('idot_4_intersection_city_day', 8866),
            4: ('idot_5_intersection_suburbs', 8851),
            5: ('idot_6_highway', 8791),
            6: ('idot_7_intersection_city_night', 8964),
            7: ('idot_8_intersection_city_night', 8962),
            8: ('idot_9_intersection_city_day', 8966),
            9: ('idot_10_city_road', 7500),
            10: ('idot_11_highway', 7500),
            11: ('idot_12_city_road', 7500),
            12: ('idot_13_intersection_city_day', 8851),
            # 13: ('idot_1_intersection_city_day_short', 84)
        }
        sequences_lost = {
            0: ('009_2011-03-23_07-00-00', 3027),
            1: ('009_2011-03-24_07-00-00', 5000)
        }
        sequences_isl = {
            0: ('isl_1_20170620-055940', 10162),
            1: ('isl_2_20170620-060941', 10191),
            2: ('isl_3_20170620-061942', 10081),
            3: ('isl_4_20170620-062943', 10089),
            4: ('isl_5_20170620-063943', 10177),
            5: ('isl_6_20170620-064944', 10195),
            6: ('isl_7_20170620-065944', 10167),
            7: ('isl_8_20170620-070945', 10183),
            8: ('isl_9_20170620-071946', 10174),
            9: ('isl_10_20170620-072946', 10127),
            10: ('isl_11_20170620-073947', 9738),
            11: ('isl_12_20170620-074947', 10087),
            12: ('isl_13_20170620-075949', 8614),
            13: ('ISL16F8J_TMC_SCU2DJ_2016-10-05_0700', 1188000),
            14: ('DJI_0020_masked_2000', 2000),
            15: ('debug_with_colors', 10)
        }
        sequences_detrac = {
            # train
            0: ('detrac_1_MVI_20011', 664),
            1: ('detrac_2_MVI_20012', 936),
            2: ('detrac_3_MVI_20032', 437),
            3: ('detrac_4_MVI_20033', 784),
            4: ('detrac_5_MVI_20034', 800),
            5: ('detrac_6_MVI_20035', 800),
            6: ('detrac_7_MVI_20051', 906),
            7: ('detrac_8_MVI_20052', 694),
            8: ('detrac_9_MVI_20061', 800),
            9: ('detrac_10_MVI_20062', 800),
            10: ('detrac_11_MVI_20063', 800),
            11: ('detrac_12_MVI_20064', 800),
            12: ('detrac_13_MVI_20065', 1200),
            13: ('detrac_14_MVI_39761', 1660),
            14: ('detrac_15_MVI_39771', 570),
            15: ('detrac_16_MVI_39781', 1865),
            16: ('detrac_17_MVI_39801', 885),
            17: ('detrac_18_MVI_39811', 1070),
            18: ('detrac_19_MVI_39821', 880),
            19: ('detrac_20_MVI_39851', 1420),
            20: ('detrac_21_MVI_39861', 745),
            21: ('detrac_22_MVI_39931', 1270),
            22: ('detrac_23_MVI_40131', 1645),
            23: ('detrac_24_MVI_40141', 1600),
            24: ('detrac_25_MVI_40152', 1750),
            25: ('detrac_26_MVI_40161', 1490),
            26: ('detrac_27_MVI_40162', 1765),
            27: ('detrac_28_MVI_40171', 1150),
            28: ('detrac_29_MVI_40172', 2635),
            29: ('detrac_30_MVI_40181', 1700),
            30: ('detrac_31_MVI_40191', 2495),
            31: ('detrac_32_MVI_40192', 2195),
            32: ('detrac_33_MVI_40201', 925),
            33: ('detrac_34_MVI_40204', 1225),
            34: ('detrac_35_MVI_40211', 1950),
            35: ('detrac_36_MVI_40212', 1690),
            36: ('detrac_37_MVI_40213', 1790),
            37: ('detrac_38_MVI_40241', 2320),
            38: ('detrac_39_MVI_40243', 1265),
            39: ('detrac_40_MVI_40244', 1345),
            40: ('detrac_41_MVI_40732', 2120),
            41: ('detrac_42_MVI_40751', 1145),
            42: ('detrac_43_MVI_40752', 2025),
            43: ('detrac_44_MVI_40871', 1720),
            44: ('detrac_45_MVI_40962', 1875),
            45: ('detrac_46_MVI_40963', 1820),
            46: ('detrac_47_MVI_40981', 1995),
            47: ('detrac_48_MVI_40991', 1820),
            48: ('detrac_49_MVI_40992', 2160),
            49: ('detrac_50_MVI_41063', 1505),
            50: ('detrac_51_MVI_41073', 1825),
            51: ('detrac_52_MVI_63521', 2055),
            52: ('detrac_53_MVI_63525', 985),
            53: ('detrac_54_MVI_63544', 1160),
            54: ('detrac_55_MVI_63552', 1150),
            55: ('detrac_56_MVI_63553', 1405),
            56: ('detrac_57_MVI_63554', 1445),
            57: ('detrac_58_MVI_63561', 1285),
            58: ('detrac_59_MVI_63562', 1185),
            59: ('detrac_60_MVI_63563', 1390),
            # test
            60: ('detrac_61_MVI_39031', 1470),
            61: ('detrac_62_MVI_39051', 1120),
            62: ('detrac_63_MVI_39211', 1660),
            63: ('detrac_64_MVI_39271', 1570),
            64: ('detrac_65_MVI_39311', 1505),
            65: ('detrac_66_MVI_39361', 2030),
            66: ('detrac_67_MVI_39371', 1390),
            67: ('detrac_68_MVI_39401', 1385),
            68: ('detrac_69_MVI_39501', 540),
            69: ('detrac_70_MVI_39511', 380),
            70: ('detrac_71_MVI_40701', 1130),
            71: ('detrac_72_MVI_40711', 1030),
            72: ('detrac_73_MVI_40712', 2400),
            73: ('detrac_74_MVI_40714', 1180),
            74: ('detrac_75_MVI_40742', 1655),
            75: ('detrac_76_MVI_40743', 1630),
            76: ('detrac_77_MVI_40761', 2030),
            77: ('detrac_78_MVI_40762', 1825),
            78: ('detrac_79_MVI_40763', 1745),
            79: ('detrac_80_MVI_40771', 1720),
            80: ('detrac_81_MVI_40772', 1200),
            81: ('detrac_82_MVI_40773', 985),
            82: ('detrac_83_MVI_40774', 950),
            83: ('detrac_84_MVI_40775', 975),
            84: ('detrac_85_MVI_40792', 1810),
            85: ('detrac_86_MVI_40793', 1960),
            86: ('detrac_87_MVI_40851', 1140),
            87: ('detrac_88_MVI_40852', 1150),
            88: ('detrac_89_MVI_40853', 1590),
            89: ('detrac_90_MVI_40854', 1195),
            90: ('detrac_91_MVI_40855', 1090),
            91: ('detrac_92_MVI_40863', 1670),
            92: ('detrac_93_MVI_40864', 1515),
            93: ('detrac_94_MVI_40891', 1545),
            94: ('detrac_95_MVI_40892', 1790),
            95: ('detrac_96_MVI_40901', 1335),
            96: ('detrac_97_MVI_40902', 1005),
            97: ('detrac_98_MVI_40903', 1060),
            98: ('detrac_99_MVI_40904', 1270),
            99: ('detrac_100_MVI_40905', 1710),
        }

        sequences_mnist_mot = {
            # train
            0: ('train_0', 1968),
            1: ('train_1', 1989),
            2: ('train_2', 1994),
            3: ('train_3', 1958),
            4: ('train_4', 1895),
            5: ('train_5', 1962),
            6: ('train_6', 1959),
            7: ('train_7', 1928),
            8: ('train_8', 1991),
            9: ('train_9', 1946),
            10: ('train_10', 1994),
            11: ('train_11', 1982),
            12: ('train_12', 1957),
            13: ('train_13', 1999),
            14: ('train_14', 1964),
            15: ('train_15', 1976),
            16: ('train_16', 1904),
            17: ('train_17', 1913),
            18: ('train_18', 1942),
            19: ('train_19', 1929),
            20: ('train_20', 1982),
            21: ('train_21', 1913),
            22: ('train_22', 1988),
            23: ('train_23', 1890),
            24: ('train_24', 1984),
            # test
            25: ('test_0', 1965),
            26: ('test_1', 1952),
            27: ('test_2', 1938),
            28: ('test_3', 1941),
            29: ('test_4', 1981),
            30: ('test_5', 1941),
            31: ('test_6', 1969),
            32: ('test_7', 1981),
            33: ('test_8', 1959),
            34: ('test_9', 1974),
            35: ('test_10', 1929),
            36: ('test_11', 1999),
            37: ('test_12', 1957),
            38: ('test_13', 1928),
            39: ('test_14', 1976),
            40: ('test_15', 1968),
            41: ('test_16', 2000),
            42: ('test_17', 1998),
            43: ('test_18', 1998),
            44: ('test_19', 1977),
            45: ('test_20', 1923),
            46: ('test_21', 1971),
            47: ('test_22', 1973),
            48: ('test_23', 1992),
            49: ('test_24', 1980),
        }

        # combined GRAM, IDOT, LOST and ISL for convenience of inter-dataset training and testing
        sequences_combined = self.combine_sequences((
            sequences_gram,  # 0 - 2
            sequences_idot,  # 3 - 15
            # sequences_detrac,  # 17 - 116
            # sequences_lost,  # 117 - 118
            # sequences_isl  # 120 - 135
        ))
        sets = {
            0: 'MOT2015',
            1: 'MOT2017',
            2: 'MOT2017_SDP',
            3: 'MOT2017_DPM',
            4: 'KITTI',
            5: 'GRAM_ONLY',
            6: 'IDOT',
            7: 'DETRAC',
            8: 'LOST',
            9: 'ISL',
            10: 'GRAM',  # combined sequence set; named GRAM for convenience
            11: 'MNIST_MOT',
        }
        sequences = dict(zip([sets[i] for i in range(len(sets))],
                             [sequences_mot2015,
                              sequences_mot2017,
                              sequences_mot2017_sdp,
                              sequences_mot2017_dpm,
                              sequences_kitti,
                              sequences_gram,
                              sequences_idot,
                              sequences_detrac,
                              sequences_lost,
                              sequences_isl,
                              sequences_combined,
                              sequences_mnist_mot,
                              ]))

        ratios_mot2015 = dict(zip(range(len(sequences_mot2015)),
                                  [self.params.ratios_mot2015] * len(sequences_mot2015)))
        ratios_mot2017 = dict(zip(range(len(sequences_mot2017)),
                                  [self.params.ratios_mot2017] * len(sequences_mot2017)))
        ratios_mot2017_sdp = dict(zip(range(len(sequences_mot2017_sdp)),
                                      [self.params.ratios_mot2017_sdp] * len(sequences_mot2017_sdp)))
        ratios_mot2017_dpm = dict(zip(range(len(sequences_mot2017_dpm)),
                                      [self.params.ratios_mot2017_dpm] * len(sequences_mot2017_dpm)))

        ratios_kitti = dict(zip(range(len(sequences_kitti)),
                                [self.params.ratios_kitti] * len(sequences_kitti)))
        ratios_gram = dict(zip(range(len(sequences_gram)),
                               [self.params.ratios_gram] * len(sequences_gram)))
        ratios_idot = dict(zip(range(len(sequences_idot)),
                               [self.params.ratios_idot] * len(sequences_idot)))
        ratios_detrac = dict(zip(range(len(sequences_detrac)),
                                 [self.params.ratios_detrac] * len(sequences_detrac)))
        ratios_lost = dict(zip(range(len(sequences_lost)),
                               [self.params.ratios_lost] * len(sequences_lost)))
        ratios_isl = dict(zip(range(len(sequences_isl)),
                              [self.params.ratios_isl] * len(sequences_isl)))
        ratios_mnist_mot = dict(zip(range(len(sequences_mnist_mot)),
                                    [self.params.ratios_mnist_mot] * len(sequences_mnist_mot)))

        ratios_combined = self.combine_sequences((
            ratios_gram,  # 0 - 2
            ratios_idot,  # 3 - 15
            # ratios_detrac,  # 19 - 78
            # ratios_lost,  # 79 - 80
            # ratios_isl  # 81 - 95
        ))
        ratios = dict(zip([sets[i] for i in range(len(sets))],
                          [ratios_mot2015,
                           ratios_mot2017,
                           ratios_mot2017_sdp,
                           ratios_mot2017_dpm,
                           ratios_kitti,
                           ratios_gram,
                           ratios_idot,
                           ratios_detrac,
                           ratios_lost,
                           ratios_isl,
                           ratios_combined,
                           ratios_mnist_mot,
                           ]))

        return sets, sequences, ratios
