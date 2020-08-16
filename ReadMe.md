<!-- MarkdownTOC -->

- [setting up developmet kit](#setting_up_developmet_ki_t_)
    - [matlab_python_engine       @ setting_up_developmet_kit](#matlab_python_engine___setting_up_developmet_ki_t_)
    - [compile       @ setting_up_developmet_kit](#compile___setting_up_developmet_ki_t_)
- [running](#running_)
    - [mot15       @ running](#mot15___runnin_g_)
        - [motmetrics       @ mot15/running](#motmetrics___mot15_runnin_g_)
        - [devkit       @ mot15/running](#devkit___mot15_runnin_g_)
    - [mot17       @ running](#mot17___runnin_g_)
        - [motmetrics       @ mot17/running](#motmetrics___mot17_runnin_g_)
        - [devkit       @ mot17/running](#devkit___mot17_runnin_g_)

<!-- /MarkdownTOC -->

<a id="setting_up_developmet_ki_t_"></a>
# setting up developmet kit

<a id="matlab_python_engine___setting_up_developmet_ki_t_"></a>
## matlab_python_engine       @ setting_up_developmet_kit

install matlab and adjust version as needed

```
python3 /usr/local/MATLAB/R2020a/extern/engines/python/setup.py install
```
<a id="compile___setting_up_developmet_ki_t_"></a>
## compile       @ setting_up_developmet_kit
```
cd evaluation/devkit/matlab_devkit
matlab
compile
```
<a id="running_"></a>
# running

<a id="mot15___runnin_g_"></a>
## mot15       @ running

<a id="motmetrics___mot15_runnin_g_"></a>
### motmetrics       @ mot15/running

```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester eval_with_devkit=0
```
<a id="devkit___mot15_runnin_g_"></a>
### devkit       @ mot15/running
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester eval_with_devkit=1
```
<a id="mot17___runnin_g_"></a>
## mot17       @ running

<a id="motmetrics___mot17_runnin_g_"></a>
### motmetrics       @ mot17/running
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester eval_with_devkit=0
```
<a id="devkit___mot17_runnin_g_"></a>
### devkit       @ mot17/running
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester eval_with_devkit=1
```