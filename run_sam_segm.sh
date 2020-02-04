source ~/nduginec_evn3/bin/activate

GPU=0
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 10 --change_lr 10 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 10 --change_lr 10 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 10 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 10 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 5 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 5 --epochs 150 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU

~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 10 --change_lr 10 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 10 --change_lr 10 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 10 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 10 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 5 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_sam.py --description sam_cl_seg --pre_train 30 --change_lr 5 --epochs 150 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
