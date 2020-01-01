source ~/nduginec_evn3/bin/activate

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description splitted --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description splitted --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description splitted --train_left 0 --segments 250 --train_right 500 --test_left 501 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description splitted --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description splitted --train_left 0 --segments 99 --train_right 100 --test_left 101 --test_right 2592 --gpu 3

#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 500 --test_left 501 --test_right 2592 --description --gpu 4 --segments 500
#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 500 --test_left 501 --test_right 2592 --description  --gpu 4 --segments 500
#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 1000 --test_left 1001 --test_right 2592 --description  --gpu 4 --segments 500
#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 1000 --test_left 1001 --test_right 2592 --description  --gpu 4 --segments 250
#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 1000 --test_left 1001 --test_right 2592 --description  --gpu 4 --segments 250
#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 1000 --test_left 1001 --test_right 2592 --description  --gpu 4 --segments 1000
#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 100 --test_left 101 --test_right 2592 --description  --gpu 4 --segments 100
#~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --train_left 0 --train_right 100 --test_left 101 --test_right 2592 --description  --gpu 4 --segments 100
