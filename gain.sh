source ~/nduginec_evn3/bin/activate

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 1000 --train_right 2000 --test_left 2001 --test_right 2592 --gpu 3 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 500 --train_right 501 --test_left 501 --test_right 2592 --gpu 3 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 250 --train_right 500 --test_left 501 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 250 --train_right 500 --test_left 501 --test_right 2592 --gpu 3 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 500 --train_right 1000 --test_left 1001 --test_right 2592 --gpu 3 --am_loss=True

~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 99 --train_right 100 --test_left 101 --test_right 2592 --gpu 3
~/nduginec_evn3/bin/python ~/ml-diplom/main_gain.py --description ngxbac --pre_train 3 --train_left 0 --segments 99 --train_right 100 --test_left 101 --test_right 2592 --gpu 3 --am_loss=True

egister forward hook !
Register backward hook !
EXCEPTION Mismatch in shape: grad_output[0] has a shape of torch.Size([10, 5]) and output[0] has a shape of torch.Size([]).
<class 'RuntimeError'>
Traceback (most recent call last):
  File "/home/nduginec/ml-diplom/main_gain.py", line 50, in <module>
    raise e
  File "/home/nduginec/ml-diplom/main_gain.py", line 45, in <module>
    gain.train_model(train_segments_set, train_classifier_set, test_set, pre_train_epoch=pre_train)
  File "/home/nduginec/ml-diplom/gain.py", line 124, in train_model
Traceback (most recent call last):
  File "/home/nduginec/ml-diplom/main_gain.py", line 50, in <module>
    raise e
  File "/home/nduginec/ml-diplom/main_gain.py", line 45, in <module>
    gain.train_model(train_segments_set, train_classifier_set, test_set, pre_train_epoch=pre_train)
  File "/home/nduginec/ml-diplom/gain.py", line 124, in train_model
    optimizer)
  File "/home/nduginec/ml-diplom/gain.py", line 172, in __gain_branch
    logits, logits_am, heatmap = self.forward(images, labels, illness)
  File "/home/nduginec/ml-diplom/gain.py", line 246, in forward
    grad_logits.backward(gradient=gradient, retain_graph=True)
  File "/home/nduginec/nduginec_evn3/lib/python3.6/site-packages/torch/tensor.py", line 166, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/home/nduginec/nduginec_evn3/lib/python3.6/site-packages/torch/autograd/__init__.py", line 93, in backward
    grad_tensors = _make_grads(tensors, grad_tensors)
  File "/home/nduginec/nduginec_evn3/lib/python3.6/site-packages/torch/autograd/__init__.py", line 29, in _make_grads
    + str(out.shape) + ".")
RuntimeError: Mismatch in shape: grad_output[0] has a shape of torch.Size([10, 5]) and output[0] has a shape of torch.Size([]).
/home/nduginec/nduginec_evn3/lib/python3.6/site-packages/torchvision/transforms/transforms.py:220: UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead.
  "please use transforms.Resize instead.")
nduginec@turing:~/ml-diplom$ Register forward hook !
Register backward hook !










