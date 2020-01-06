"""draw plot of accuracy"""
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

plot_images_path = "/home/nikita/PycharmProjects/ml-diplom/stat_plot"
logs_path = "/home/nikita/PycharmProjects/logs"


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


def __load_statistics(dct: dict) -> tuple:
    f_measures = []
    recall = []
    precision = []
    for i in range(0, 5):
        key = 'f_1_{}'.format(i)
        f_measures.append((key, dct[key]))
    f_measures.append(('f_1_global', dct['f_1_global']))

    for i in range(0, 5):
        key = 'recall_{}'.format(i)
        recall.append((key, dct[key]))
    recall.append(('recall_global', dct['recall_global']))

    for i in range(0, 5):
        key = 'precision_{}'.format(i)
        precision.append((key, dct[key]))
    precision.append(('precision_global', dct['precision_global']))
    return f_measures, recall, precision


def __parse_auc(auc_value: str) -> tuple:
    splited = re.split(r'trust_[0-5]{1}=|prob_[0-5]{1}=', auc_value)
    splited = list(filter(lambda x: len(x) > 0, splited))
    trust = []
    prob = []
    for i in range(0, 5):
        trust.append(('trust_{}'.format(i), list(map(lambda x: int(float(x)), splited[i].split(",")))))
    for i in range(0, 5):
        prob.append(('prob_{}'.format(i), list(map(float, splited[5 + i].split(",")))))
    return trust, prob


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


def gain_parse_test_and_classifier_all(input: str):
    dct = __to_dict(input)
    loss_cl = float(dct['Loss_CL'])
    accuracy_cl = float(dct['Accuracy_CL_Percent']) if 'Accuracy_CL_Percent' in dct else float(dct['Accuracy_CL'])
    f_measures, recall, precision = __load_statistics(dct)
    if not 'auc_roc' in dct:
        return loss_cl, accuracy_cl, f_measures, recall, precision
    else:
        trust, prob = __parse_auc(dct['auc_roc'])
        return loss_cl, accuracy_cl, f_measures, recall, precision, trust, prob


def gain_parse_train(input: str):
    dct = __to_dict(input)
    Loss_CL = float(dct['Loss_CL'])
    Loss_AM = float(dct['Loss_AM'])
    Loss_E = float(dct['Loss_E'])
    Loss_Total = float(dct['Loss_Total'])
    Accuracy_CL = dct['Accuracy_CL']
    if Accuracy_CL[0] == "=":
        Accuracy_CL = float(Accuracy_CL[1:])
    else:
        Accuracy_CL = float(Accuracy_CL)
    f_measures, recall, precision = __load_statistics(dct)
    if not 'auc_roc' in dct:
        return Loss_CL, Loss_AM, Loss_E, Loss_Total, Accuracy_CL, f_measures, recall, precision
    else:
        trust, prob = __parse_auc(dct['auc_roc'])
        return Loss_CL, Loss_AM, Loss_E, Loss_Total, Accuracy_CL, f_measures, recall, precision, trust, prob


def __load_file(file_path: str, train_parse, test_parse):
    train = []
    test = []
    train_index = 1
    test_index = 4
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
                test_index += 4
        file_name = os.path.basename(f.name)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty data")
    return train, test, train_indexes, test_indexes, file_name


def draw_metrics_classification_plot(file_path: str):
    train, test, train_indexes, test_indexes, file_name = __load_file(file_path, gain_parse_test_and_classifier_all,
                                                                      gain_parse_test_and_classifier_all)
    train_auc = 0
    test_auc = 0
    for i in train:
        if len(i) > 5:
            train_auc += 1
    for i in test:
        if len(i) > 5:
            test_auc += 1

    fig, axes = plt.subplots(5 + max(train_auc, test_auc) * 2, 2, figsize=(15, 50))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=2)

    def __set_train_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][0].set_title(label_text)
        axes[axes_index][0].plot(train_indexes, list(map(lambda x: x[item_index], train)))

    def __set_test_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][0].set_title(label_text)
        axes[axes_index][0].plot(test_indexes, list(map(lambda x: x[item_index], test)))

    def __set_train_measures(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][1].set_title(label_text)
        for i in range(0, 6):
            legend = train[0][item_index][i][0]
            axes[axes_index][1].plot(train_indexes, list(map(lambda x: x[item_index][i][1], train)), label=legend)
        axes[axes_index][1].legend(loc='upper left')

    def __set_test_measures(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][1].set_title(label_text)
        for i in range(0, 6):
            legend = test[0][item_index][i][0]
            axes[axes_index][1].plot(test_indexes, list(map(lambda x: x[item_index][i][1], test)), label=legend)
        axes[axes_index][1].legend(loc='upper left')

    __set_train_axes(0, "train classification loss", 0)
    __set_train_axes(1, "train accuracy", 1)
    __set_train_measures(0, "f_1_train", 2)
    __set_train_measures(1, "recall_train", 3)
    __set_train_measures(2, "precision_train", 4)

    __set_test_axes(2, "test classification loss", 0)
    __set_test_axes(3, "test accuracy", 1)
    __set_test_measures(3, "f_1_test", 2)
    __set_test_measures(4, "recall_test", 3)
    __set_test_measures(5, "precision_test", 4)

    train_idx = 0
    for idx, values in enumerate(train):
        if len(values) <= 5:
            continue
        axes[5 + train_idx * 2][0].set_title("auc plot for train={}".format(idx + 1))
        axes[5 + train_idx * 2 + 1][0].set_title("auc plot for train={}".format(idx + 1))
        for i in range(0, 5):
            trust_name, trust_value = values[5][i]
            prob_name, prob_value = values[6][i]
            a, b, _ = roc_curve(trust_value, prob_value)
            c = auc(a, b)
            axes[5 + train_idx * 2][0].plot(a, b, lw=2, label='ROC curve {} (area = {:.2f})'.format(prob_name[4:], c))
            a1, b1, _ = precision_recall_curve(trust_value, prob_value)
            c1 = average_precision_score(trust_value, prob_value)
            axes[5 + train_idx * 2 + 1][0].plot(a1, b1, lw=2,
                                                label='PR curve {} (area = {:.2f})'.format(prob_name[4:], c1))
        axes[5 + train_idx * 2][0].legend(loc='upper left')
        axes[5 + train_idx * 2 + 1][0].legend(loc='upper left')
        train_idx += 1

    test_idx = 0
    for idx, values in enumerate(test):
        if len(values) <= 5:
            continue
        axes[5 + test_idx * 2][1].set_title("auc plot for test={}".format((idx + 1) * 4))
        axes[5 + test_idx * 2 + 1][1].set_title("auc plot for test={}".format((idx + 1) * 4))
        for i in range(0, 5):
            trust_name, trust_value = values[5][i]
            prob_name, prob_value = values[6][i]
            a, b, _ = roc_curve(trust_value, prob_value)
            c = auc(a, b)
            axes[5 + test_idx * 2][1].plot(a, b, lw=2, label='ROC curve {} (area = {:.2f})'.format(prob_name[4:], c))
            a1, b1, _ = precision_recall_curve(trust_value, prob_value)
            c1 = average_precision_score(trust_value, prob_value)
            axes[5 + test_idx * 2 + 1][1].plot(a1, b1, lw=2,
                                               label='PR curve {} (area = {:.2f})'.format(prob_name[4:], c1))
        axes[5 + test_idx * 2][1].legend(loc='upper left')
        axes[5 + test_idx * 2 + 1][1].legend(loc='upper left')
        test_idx += 1

    for ax in axes[0:5][:].flat:
        ax.set(xlabel='epoch', ylabel='value')

    for ax in axes[6:][:].flat:
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

    plt.suptitle(file_name, y=1.99)
    plt.savefig(os.path.join(plot_images_path, file_name[:-4]), bbox_inches='tight')
    plt.show()


def draw_gain_metric_plot(file_path: str):
    train, test, train_indexes, test_indexes, file_name = __load_file(file_path, gain_parse_train,
                                                                      gain_parse_test_and_classifier_all)

    train_auc = 0
    test_auc = 0
    for i in train:
        if len(i) > 8:
            train_auc += 1
    for i in test:
        if len(i) > 5:
            test_auc += 1

    fig, axes = plt.subplots(7 + max(train_auc, test_auc) * 2, 2, figsize=(15, 50))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=2)

    def __set_train_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][0].set_title(label_text)
        axes[axes_index][0].plot(train_indexes, list(map(lambda x: x[item_index], train)))

    def __set_test_axes(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][0].set_title(label_text)
        axes[axes_index][0].plot(test_indexes, list(map(lambda x: x[item_index], test)))

    def __set_train_measures(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][1].set_title(label_text)
        for i in range(0, 6):
            legend = train[0][item_index][i][0]
            axes[axes_index][1].plot(train_indexes, list(map(lambda x: x[item_index][i][1], train)), label=legend)
        axes[axes_index][1].legend(loc='upper left')

    def __set_test_measures(axes_index: int, label_text: str, item_index: int):
        axes[axes_index][1].set_title(label_text)
        for i in range(0, 6):
            legend = test[0][item_index][i][0]
            axes[axes_index][1].plot(test_indexes, list(map(lambda x: x[item_index][i][1], test)), label=legend)
        axes[axes_index][1].legend(loc='upper left')

    __set_train_axes(0, "train classification loss", 0)
    __set_train_axes(1, "train am loss", 1)
    __set_train_axes(2, "train e loss", 2)
    __set_train_axes(3, "train total loss", 3)
    __set_train_axes(4, "train accuracy", 4)
    __set_train_measures(0, "f_1_train", 5)
    __set_train_measures(1, "recall_train", 6)
    __set_train_measures(2, "precision_train", 7)

    __set_test_axes(5, "test classification loss", 0)
    __set_test_axes(6, "test accuracy", 1)
    __set_test_measures(3, "f_1_test", 2)
    __set_test_measures(4, "recall_test", 3)
    __set_test_measures(5, "precision_test", 4)

    train_idx = 0
    for idx, values in enumerate(train):
        if len(values) <= 8:
            continue
        axes[7 + train_idx * 2][0].set_title("auc plot for train={}".format(idx + 1))
        axes[7 + train_idx * 2 + 1][0].set_title("auc plot for train={}".format(idx + 1))
        for i in range(0, 5):
            trust_name, trust_value = values[8][i]
            prob_name, prob_value = values[9][i]
            a, b, _ = roc_curve(trust_value, prob_value)
            c = auc(a, b)
            axes[7 + train_idx * 2][0].plot(a, b, lw=2, label='ROC curve {} (area = {:.2f})'.format(prob_name[4:], c))
            a1, b1, _ = precision_recall_curve(trust_value, prob_value)
            c1 = average_precision_score(trust_value, prob_value)
            axes[7 + train_idx * 2 + 1][0].plot(a1, b1, lw=2,
                                                label='PR curve {} (area = {:.2f})'.format(prob_name[4:], c1))
        axes[7 + train_idx * 2][0].legend(loc='upper left')
        axes[7 + train_idx * 2 + 1][0].legend(loc='upper left')
        train_idx += 1

    test_idx = 0
    for idx, values in enumerate(test):
        if len(values) <= 5:
            continue
        axes[7 + test_idx * 2][1].set_title("auc plot for test={}".format((idx + 1) * 4))
        axes[7 + test_idx * 2 + 1][1].set_title("auc plot for test={}".format((idx + 1) * 4))
        for i in range(0, 5):
            trust_name, trust_value = values[5][i]
            prob_name, prob_value = values[6][i]
            a, b, _ = roc_curve(trust_value, prob_value)
            c = auc(a, b)
            axes[7 + test_idx * 2][1].plot(a, b, lw=2, label='ROC curve {} (area = {:.2f})'.format(prob_name[4:], c))
            a1, b1, _ = precision_recall_curve(trust_value, prob_value)
            c1 = average_precision_score(trust_value, prob_value)
            axes[7 + test_idx * 2 + 1][1].plot(a1, b1, lw=2,
                                               label='PR curve {} (area = {:.2f})'.format(prob_name[4:], c1))
        axes[7 + test_idx * 2][1].legend(loc='upper left')
        axes[7 + test_idx * 2 + 1][1].legend(loc='upper left')
        test_idx += 1

    for ax in axes[0:7][:].flat:
        ax.set(xlabel='epoch', ylabel='value')

    for ax in axes[8:][:].flat:
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

    plt.suptitle(file_name, y=1.99)
    plt.savefig(os.path.join(plot_images_path, file_name[:-4]), bbox_inches='tight')
    plt.show()


def draw_gain_plot(file_path: str):
    train, test, train_indexes, test_indexes, file_name = __load_file(file_path, gain_train_parse, gain_test_parse)
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
    train, test, train_indexes, test_indexes, file_name = __load_file(file_path, classifier_train_parse,
                                                                      classifier_test_parse)
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
            try:
                if "metric_classifier" in file:
                    print("begin", file)
                    draw_metrics_classification_plot(os.path.join(r, file))
                    print("end", file)
                elif "classifier" in file:
                    print("begin", file)
                    draw_classification_plot(os.path.join(r, file))
                    print("end", file)
                elif "metric_gain" in file:
                    print("begin", file)
                    draw_gain_metric_plot(os.path.join(r, file))
                    print("end", file)
                elif "gain" in file:
                    print("begin", file)
                    draw_gain_plot(os.path.join(r, file))
                    print("end", file)
            except ValueError as e:
                print("empty file: {}".format(file))


if __name__ == "__main__":
    visualize_all()
