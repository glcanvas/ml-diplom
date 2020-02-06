source ~/nduginec_evn3/bin/activate

GPU=1
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.5 --segments 0.4 --test 0.1  --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.5 --segments 0.4 --test 0.1  --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.1 --segments 0.8 --test 0.1  --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.1 --segments 0.8 --test 0.1  --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.1 --segments 0.5 --test 0.4  --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.1 --segments 0.5 --test 0.4  --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.25 --segments 0.25 --test 0.5 --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.25 --segments 0.25 --test 0.5 --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.1--segments 0.6 --test 0.3   --pre_train 100 --epochs 100 --gpu $GPU
~/nduginec_evn3/bin/python ~/ml2/ml-diplom/main_sam.py --description sam_only_classif_honest --classifier 0.1--segments 0.6 --test 0.3   --pre_train 100 --epochs 100 --gpu $GPU

