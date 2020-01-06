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


# visualize_all()

# if __name__ == "__main__":
#    visualize_all()
from sklearn import metrics
import numpy as np

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print(fpr, tpr, thresholds)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                         random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

trust_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prob_4 = [0.67969, 0.63364, 0.42932, 0.63385, 0.64212, 0.36453, 0.54794, 0.44479, 0.48420, 0.53724, 0.49006, 0.65804,
          0.57631, 0.49099, 0.47839, 0.61472, 0.30450, 0.41202, 0.49871, 0.37486, 0.39097, 0.55319, 0.32127, 0.60204,
          0.64038, 0.59119, 0.23944, 0.63146, 0.64349, 0.68464, 0.48382, 0.34811, 0.54935, 0.63747, 0.59903, 0.55663,
          0.53202, 0.50537, 0.55123, 0.31934, 0.54358, 0.47011, 0.56575, 0.72461, 0.62780, 0.39381, 0.42562, 0.60204,
          0.54649, 0.54091]

a, b, _ = roc_curve(trust_0, prob_4)
c = auc(a, b)

plt.figure()
lw = 2
plt.plot(a, b, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % c)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
