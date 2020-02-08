"""draw plot of accuracy"""
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

plot_images_path = "/home/nikita/PycharmProjects/ml-diplom/sam_plots"
logs_path = "/home/nikita/PycharmProjects/ml-data/logs"


def __load_statistics(dct: dict) -> tuple:
    f_measures = []
    recall = []
    precision = []
    for i in range(0, 5):
        key = 'f_1_{}'.format(i)
        f_measures.append((key, float(dct[key]) if key in dct else -1))
    f_measures.append(('f_1_global', float(dct['f_1_global']) if 'f_1_global' in dct else -1))

    for i in range(0, 5):
        key = 'recall_{}'.format(i)
        recall.append((key, float(dct[key]) if key in dct else -1))
    recall.append(('recall_global', float(dct['recall_global']) if 'recall_global' in dct else -1))

    for i in range(0, 5):
        key = 'precision_{}'.format(i)
        precision.append((key, float(dct[key]) if key in dct else -1))
    precision.append(('precision_global', float(dct['precision_global']) if 'precision_global' in dct else -1))
    return f_measures, recall, precision


def __to_dict(input: str) -> dict:
    dct = dict()
    for key_value in input.split():
        idx = key_value.find("=")
        if idx == -1:
            continue
        key = key_value[:idx]
        value = key_value[idx + 1:]
        dct[key] = value
    return dct


def sam_parse_train(input: str):
    dct = __to_dict(input)
    Loss_CL = float(dct['Loss_CL'])
    Loss_M = float(dct['Loss_M']) if 'Loss_M' in dct else -1
    Loss_L1 = float(dct['Loss_L1']) if 'Loss_L1' in dct else -1
    Loss_Total = float(dct['Loss_Total']) if 'Loss_Total' in dct else -1
    Accuracy_CL = dct['Accuracy_CL']
    if Accuracy_CL[0] == "=":
        Accuracy_CL = float(Accuracy_CL[1:])
    else:
        Accuracy_CL = float(Accuracy_CL)
    f_measures, recall, precision = __load_statistics(dct)
    return Loss_CL, Loss_M, Loss_L1, Loss_Total, Accuracy_CL, f_measures, recall, precision


def __load_file(file_path: str, train_parse, test_parse):
    train = []
    test = []
    train_index = 1
    test_index = 1
    train_indexes = []
    test_indexes = []
    with open(file_path) as f:
        for l in f.readlines():
            if "TRAIN" in l:
                train.append(train_parse(l))
                train_indexes.append(train_index)
                train_index += 1
            elif "TEST" in l:
                test.append(test_parse(l))
                test_indexes.append(test_index)
                test_index += 1
        file_name = os.path.basename(f.name)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty data")
    return train, test, train_indexes, test_indexes, file_name


# OK
def draw_gain_metric_plot(file_path: str):
    train, test, train_indexes, test_indexes, file_name = __load_file(file_path, sam_parse_train,
                                                                      sam_parse_train)

    fig, axes = plt.subplots(8, 2, figsize=(15, 50))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=2)

    def __set_train_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][0].set_title(label_text)
        axes[axes_index][0].plot(train_indexes, list(map(lambda x: x[item_index], train)))

    def __set_test_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][1].set_title(label_text)
        axes[axes_index][1].plot(test_indexes, list(map(lambda x: x[item_index], test)))

    def __set_train_measures(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][0].set_title(label_text)
        for i in range(0, 6):
            legend = train[0][item_index][i][0]
            axes[axes_index][0].plot(train_indexes, list(map(lambda x: x[item_index][i][1], train)), label=legend)
        axes[axes_index][0].legend(loc='upper left')

    def __set_test_measures(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][1].set_title(label_text)
        for i in range(0, 6):
            legend = test[0][item_index][i][0]
            axes[axes_index][1].plot(test_indexes, list(map(lambda x: x[item_index][i][1], test)), label=legend)
        axes[axes_index][1].legend(loc='upper left')

    __set_train_axes(0, "train classification loss", 0)
    __set_train_axes(1, "train M loss", 1)
    __set_train_axes(2, "train l1 loss", 2)
    __set_train_axes(3, "train total loss", 3)
    __set_train_axes(4, "train accuracy", 4)

    __set_train_measures(5, "f_1_train", 5)
    __set_train_measures(6, "recall_train", 6)
    __set_train_measures(7, "precision_train", 7)

    __set_test_axes(0, "test classification loss", 0)
    __set_test_axes(1, "test M loss", 1)
    __set_test_axes(2, "test l1 loss", 2)
    __set_test_axes(3, "test total loss", 3)
    __set_test_axes(4, "test accuracy", 4)

    __set_test_measures(5, "f_1_test", 5)
    __set_test_measures(6, "recall_test", 6)
    __set_test_measures(7, "precision_test", 7)

    for ax in axes[0:7][:].flat:
        ax.set(xlabel='epoch', ylabel='value')

    plt.suptitle(file_name, y=1.99)
    plt.savefig(os.path.join(plot_images_path, file_name.replace(".", "_")[:-4]), bbox_inches='tight')
    # plt.show()


import traceback


def visualize_all():
    for r, d, f in os.walk(logs_path):
        for file in f:
            try:
                if "cl_5" in file:
                    print("begin", file)
                    draw_gain_metric_plot(os.path.join(r, file))
                    print("end", file)
                elif "__sam" in file:
                    print("begin", file)
                    draw_gain_metric_plot(os.path.join(r, file))
                    print("end", file)
            except ValueError as e:
                print("error file: {}".format(file), e)
                traceback.print_exc()


if __name__ == "__main__":
    visualize_all()
