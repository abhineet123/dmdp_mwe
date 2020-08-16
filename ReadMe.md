<!-- No Heading Fix -->

<!-- MarkdownTOC -->

- [setup development kit](#optional)
    - [matlab_python_engine](#matlab_python_engin_e_)
    - [compile](#compile_)
- [install](#install_)
- [run](#run_)
    - [mot15](#mot15_)
        - [motmetrics](#motmetric_s_)
        - [devkit](#devki_t_)
    - [mot17](#mot17_)
        - [motmetrics](#motmetric_s__1)
        - [devkit](#devki_t__1)

<!-- /MarkdownTOC -->

<a id="optional"></a>
# setup development kit [optional]

<a id="matlab_python_engin_e_"></a>
## matlab_python_engine 

install matlab and adjust version as needed:

```
python3 /usr/local/MATLAB/R2020a/extern/engines/python/setup.py install
```
<a id="compile_"></a>
## compile 
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

<a id="mot15_"></a>
## mot15 

<a id="motmetric_s_"></a>
### motmetrics 

```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester eval_with_devkit=0
```
<a id="devki_t_"></a>
### devkit 
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester eval_with_devkit=1
```
<a id="mot17_"></a>
## mot17 

<a id="motmetric_s__1"></a>
### motmetrics
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester eval_with_devkit=0
```
<a id="devki_t__1"></a>
### devkit
```
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester eval_with_devkit=1
```