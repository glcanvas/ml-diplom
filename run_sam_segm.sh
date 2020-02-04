source ~/nduginec_evn3/bin/activate

GPU=3
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 250 --train_right 500 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 250 --train_right 500 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 100 --train_right 100 --test_left 101 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 15 --epochs 150 --train_left 0 --segments 100 --train_right 100 --test_left 101 --test_right 2592 --gpu $GPU
