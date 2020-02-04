source ~/nduginec_evn3/bin/activate

GPU=1
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 500 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 500 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 100 --test_left 101 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_only_classif --pre_train 100 --epochs 100 --train_left 0 --segments 2 --train_right 100 --test_left 101 --test_right 2592 --gpu $GPU

