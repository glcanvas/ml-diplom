Loss_CL
Loss_M
Loss_Total
Loss_L1
Accuracy_CL

f_1_0
f_1_1
f_1_2
f_1_3
f_1_4
f_1_global

recall_0
recall_1
recall_2
recall_3
recall_4
recall_global

precision_0
precision_1
precision_2
precision_3
precision_4
precision_global



python on_server_executor.py '0;vgg16+100-50;False;softf1' '0;vgg16+AM;False;softf1' '0;vgg16+1-1;False;softf1' 

df -h

ssh -i key.pem ubuntu@176.99.130.202