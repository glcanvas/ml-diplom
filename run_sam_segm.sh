source ~/nduginec_evn3/bin/activate

GPU=0
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.5 --segments 0.4 --test 0.1  --pre_train 10 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.5 --segments 0.4 --test 0.1  --pre_train 10 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.1 --segments 0.8 --test 0.1  --pre_train 30 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.1 --segments 0.8 --test 0.1  --pre_train 30 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.1 --segments 0.5 --test 0.4  --pre_train 30 --change_lr 5 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.1 --segments 0.5 --test 0.4  --pre_train 30 --change_lr 5 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.25 --segments 0.25 --test 0.5 --pre_train 10 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.25 --segments 0.25 --test 0.5 --pre_train 10 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.1--segments 0.6 --test 0.3   --pre_train 30 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.1--segments 0.6 --test 0.3   --pre_train 30 --change_lr 10 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.5 --segments 0.4 --test 0.1  --pre_train 30 --change_lr 5 --epochs 150 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_lr_l1_cbam_honest --classifier 0.5 --segments 0.4 --test 0.1  --pre_train 30 --change_lr 5 --epochs 150 --gpu $GPU
