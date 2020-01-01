"""draw plot of accuracy"""
import matplotlib.pyplot as plt
import os


def train_parse(train_line):
    values = train_line.split()
    loss_cl = float(values[4][:-1])
    loss_am = float(values[6][:-1])
    loss_e = float(values[9][:-1])
    loss_total = float(values[12][:-1])
    accuracy = float(values[14][:-1])
    print(loss_cl, loss_am, loss_e, loss_total, accuracy)


def draw_gain_plot(file_path: str):
    train = []
    test = []
    with open(file_path) as f:
        for l in f.readlines():
            if "TRAIN" in l:
                train.append(l)
            elif "TEST" in l:
                test.append(l)
        file_name = os.path.basename(f.name)

    plt.suptitle(file_name)
    plt.show()

"""
TRAIN Epoch 96, Loss_CL: 2917.251599, Loss_AM: 0.000000, Loss E: 856033376329497.125000, Loss Total: 85603336746468464.000000, Accuracy_CL: 1.600000%
TEST Loss_CL: 3284.033308, Accuracy_CL: 21.476190%
TRAIN Epoch 97, Loss_CL: 3067.886648, Loss_AM: 0.000000, Loss E: 856033376329497.125000, Loss Total: 85603336746468464.000000, Accuracy_CL: 1.600000%
"""

train_parse("TRAIN Epoch 97, Loss_CL: 3067.886648, Loss_AM: 0.000000, Loss E: 856033376329497.125000, Loss Total: 85603336746468464.000000, Accuracy_CL: 1.600000%")