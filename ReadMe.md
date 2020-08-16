<!-- MarkdownTOC -->

- [setup development kit](#optional)
    - [matlab_python_engine       @ setup_development_kit_](#optional)
    - [compile       @ setup_development_kit_](#optional)
- [install](#install_)
- [run](#run_)
    - [mot15       @ run](#mot15___ru_n_)
        - [motmetrics       @ mot15/run](#motmetrics___mot15_ru_n_)
        - [devkit       @ mot15/run](#devkit___mot15_ru_n_)
    - [mot17       @ run](#mot17___ru_n_)
        - [motmetrics       @ mot17/run](#motmetrics___mot17_ru_n_)
        - [devkit       @ mot17/run](#devkit___mot17_ru_n_)

<!-- /MarkdownTOC -->

<a id="optional"></a>
# setup development kit [optional]

<a id="optional"></a>
## matlab_python_engine       @ setup_development_kit_[optional]

install matlab and adjust version as needed:

```
python3 /usr/local/MATLAB/R2020a/extern/engines/python/setup.py install
```
<a id="optional"></a>
## compile       @ setup_development_kit_[optional]
```
cd evaluation/devkit/matlab_devkit
matlab
compile
```

<a id="install_"></a>
# install

```
python3 -m pip install -r requirements.txt
```

<a id="run_"></a>
# run

<a id="mot15___ru_n_"></a>
## mot15       @ run

<a id="motmetrics___mot15_ru_n_"></a>
### motmetrics       @ mot15/run

```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester eval_with_devkit=0
```
<a id="devkit___mot15_ru_n_"></a>
### devkit       @ mot15/run
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester eval_with_devkit=1
```
<a id="mot17___ru_n_"></a>
## mot17       @ run

<a id="motmetrics___mot17_ru_n_"></a>
### motmetrics       @ mot17/run
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester eval_with_devkit=0
```
<a id="devkit___mot17_ru_n_"></a>
### devkit       @ mot17/run
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester eval_with_devkit=1
```