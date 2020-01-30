source ~/nduginec_evn3/bin/activate

GPU=1

~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 1 --train_left 0 --gpu $GPU --segments 100 --train_right 200 --test_left 2001 --test_right 2592 --epochs   150
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --epochs 150  --am_loss=True
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 500 --train_right 501 --test_left 501 --test_right 2592 --epochs 150
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 500 --train_right 501 --test_left 501 --test_right 2592 --epochs 150  --am_loss=True
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 250 --train_right 500 --test_left 501 --test_right 2592 --epochs 150
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 250 --train_right 500 --test_left 501 --test_right 2592 --epochs 150  --am_loss=True
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --epochs 150
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --epochs 150  --am_loss=True
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 99 --train_right 100 --test_left 101 --test_right 2592 --epochs 150
~/nduginec_evn3/bin/python ~/ml-diplom/main_cbam.py --description bam_upd_all --pre_train 25 --train_left 0 --gpu $GPU --segments 99 --train_right 100 --test_left 101 --test_right 2592 --epochs 150  --am_loss=True
