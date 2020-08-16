try:
    """has to be imported before any other package to avoid annoying 
    bad_alloc issues due to obscure conflicts with one or more other 
    packages including pytorch"""
    import matlab.engine
except ImportError:
    pass

import paramparse

from utilities import CustomLogger

from params import Params
from data import Data

from run import Test


def main():
    """get parameters"""
    params = Params()
    paramparse.process(params, allow_unknown=1)
    params.process()

    """setup logger"""
    _logger = CustomLogger.setup()

    _data = Data(params.data, _logger)
    test_logger = CustomLogger(_logger, names=('test',), key='custom_header')
    Test.run(_data, params.tester, params.test, test_logger)


if __name__ == '__main__':
    main()
