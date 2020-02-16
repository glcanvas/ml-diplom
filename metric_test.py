import sklearn.metrics as metrics


def __to_global(a, b):
    aa = []
    bb = []
    for index, i in enumerate(a):
        aa.extend(list(map(lambda x: x * (index + 1), i)))
    for index, i in enumerate(b):
        bb.extend(list(map(lambda x: x * (index + 1), i)))
    return aa, bb


if __name__ == "__main__":
    a_true = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    a_pred = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    a, b = __to_global([a_true], [a_pred])
    print(a, b)
    print(metrics.f1_score(a_true, a_pred, pos_label=1))

    print(metrics.f1_score(a_true, a_pred, average='micro'))

    print(metrics.f1_score(a_true, a_pred, average='macro'))
