"""draw plot of accuracy"""
import matplotlib.pyplot as plt
import os

plot_images_path = "/home/nikita/PycharmProjects/ml-diplom/stat_plot"
logs_path = "/home/nikita/PycharmProjects/ml-data/logs"


def gain_train_parse(train_line):
    values = train_line.split()
    loss_cl = float(values[4][:-1])
    loss_am = float(values[6][:-1])
    loss_e = float(values[9][:-1])
    loss_total = float(values[12][:-1])
    accuracy = float(values[14][:-1])
    return loss_cl, loss_am, loss_e, loss_total, accuracy


def gain_test_parse(test_line):
    values = test_line.split()
    loss_cl = float(values[2][:-1])
    accuracy = float(values[4][:-1])
    return loss_cl, accuracy


def classifier_train_parse(train_line):
    values = train_line.split()
    loss_cl = float(values[7][:-1])
    accuracy = float(values[9][:-1])
    return loss_cl, accuracy


def classifier_test_parse(test_line):
    values = test_line.split()
    loss_cl = float(values[2][:-1])
    accuracy = float(values[4][:-1])
    return loss_cl, accuracy


def draw_gain_plot(file_path: str):
    train = []
    test = []
    train_index = 1
    test_index = 4
    train_indexes = []
    test_indexes = []
    with open(file_path) as f:
        for l in f.readlines():
            if "TRAIN" in l:
                train.append(gain_train_parse(l))
                train_indexes.append(train_index)
                train_index += 1
            elif "TEST" in l:
                test.append(gain_test_parse(l))
                test_indexes.append(test_index)
                test_index += 4
        file_name = os.path.basename(f.name)
    fig, axes = plt.subplots(7, figsize=(9, 11))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=2)

    def __set_train_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index].set_title(label_text)
        axes[axes_index].plot(train_indexes, list(map(lambda x: x[item_index], train)))

    def __set_test_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index].set_title(label_text)
        axes[axes_index].plot(test_indexes, list(map(lambda x: x[item_index], test)))

    __set_train_axes(0, "train classification loss", 0)
    __set_train_axes(1, "train am loss", 1)
    __set_train_axes(2, "train e loss", 2)
    __set_train_axes(3, "train total loss", 3)
    __set_train_axes(4, "train accuracy", 4)

    __set_test_axes(5, "test classification loss", 0)
    __set_test_axes(6, "test accuracy", 1)

    for ax in axes.flat:
        ax.set(xlabel='epoch', ylabel='value')
    plt.suptitle(file_name, y=1.99)
    plt.savefig(os.path.join(plot_images_path, file_name[:-4]), bbox_inches='tight')


def draw_classification_plot(file_path: str):
    train = []
    test = []
    train_index = 1
    test_index = 4
    train_indexes = []
    test_indexes = []
    with open(file_path) as f:
        for l in f.readlines():
            if "TRAIN" in l:
                train.append(classifier_train_parse(l))
                train_indexes.append(train_index)
                train_index += 1
            elif "TEST" in l:
                test.append(classifier_test_parse(l))
                test_indexes.append(test_index)
                test_index += 4
        file_name = os.path.basename(f.name)
    fig, axes = plt.subplots(4, figsize=(9, 11))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=2)

    def __set_train_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index].set_title(label_text)
        axes[axes_index].plot(train_indexes, list(map(lambda x: x[item_index], train)))

    def __set_test_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index].set_title(label_text)
        axes[axes_index].plot(test_indexes, list(map(lambda x: x[item_index], test)))

    __set_train_axes(0, "train classification loss", 0)
    __set_train_axes(1, "train accuracy", 1)

    __set_test_axes(2, "test classification loss", 0)
    __set_test_axes(3, "test accuracy", 1)

    for ax in axes.flat:
        ax.set(xlabel='epoch', ylabel='value')
    plt.suptitle(file_name, y=1.99)
    plt.savefig(os.path.join(plot_images_path, file_name[:-4]), bbox_inches='tight')


def visualize_all():
    for r, d, f in os.walk(logs_path):
        for file in f:
            if "classifier" in file:
                draw_classification_plot(os.path.join(r, file))

            elif "gain" in file:
                draw_gain_plot(os.path.join(r, file))


visualize_all()

if __name__ == "__main__":
    visualize_all()
