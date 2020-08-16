from data import Data
from run import Test, Train
from tester import Tester

from utilities import parse_seq_IDs


class Params:
    def __init__(self):
        self.gpu = ""

        self.cfg_root = 'cfg'
        self.cfg_ext = 'cfg'
        self.cfg = ('params',)

        self.log_dir = ''

        self.data = Data.Params()
        self.train = Train.Params()
        self.test = Test.Params()
        self.tester = Tester.Params()

    def process(self):
        self.train.seq = parse_seq_IDs(self.train.seq)
        self.test.seq = parse_seq_IDs(self.test.seq)

        self.test.synchronize(self.train)
