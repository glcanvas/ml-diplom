source ~/nduginec_evn3/bin/activate

GPU=3

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu $GPU --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu $GPU --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 250 --train_right 500 --test_left 501 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 250 --train_right 500 --test_left 501 --test_right 2592 --gpu $GPU --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 99 --train_right 100 --test_left 101 --test_right 2592 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 99 --train_right 100 --test_left 101 --test_right 2592 --gpu $GPU --am_loss=True


~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.28
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.28 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.27
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.27 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.26
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.26 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.25
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.25 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.24
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description JUST_GAIN --pre_train 25 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu $GPU --from_gradient_layer True --gradient_layer_name features.24 --am_loss=True