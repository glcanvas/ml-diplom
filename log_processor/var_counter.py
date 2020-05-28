import numpy as np

#white_list = "f_1_0 f_1_1 f_1_2 f_1_3 f_1_4 f_1_global".split()
#mapper = ["F-measure_заболевание_1", "F-measure_заболевание_2", "F-measure_заболевание_3",
#          "F-measure_заболевание_4", "F-measure_заболевание_5", "F-measure_общая"]

white_list = "f_1_global".split()
mapper = ["F-measure_общая"]


def _map(x):
    for idx, i in enumerate(white_list):
        if i in x:
            return mapper[idx]


def process_values(output_plot_logs_dir: str):
    with open(output_plot_logs_dir, 'r') as f:
        output_plot_logs = "\n".join(f.readlines())

    dct = dict()
    max_len = 1
    algos = set()
    for l in output_plot_logs.split("\n"):
        if len(l.split()) != 3:
            continue
        v, alg, metric = l.split()
        if '_|' not in alg:
            algos.add(alg)
            continue
        dct.setdefault(alg, dict())
        v = "{:.3f}".format(float(v))
        found = False
        for m in white_list:
            if m in metric:
                found = True
        if not found:
            continue
        metric = _map(metric)
        dct[alg][metric] = v
        max_len = max([len(v), len(alg), len(metric), max_len])


    var_arary = np.zeros((len(dct.keys()), 1))

    for i, k in enumerate(sorted(dct.keys())):
        #print(k, " " * (max_len - len(k)), end=" ")
        for j, m in enumerate(sorted(dct[k].keys())):
            #print(dct[k][m] + " " * (max_len - len(dct[k][m])), end=" ")
            var_arary[i][j] = float(dct[k][m])
        #print()
    print()
    print("Дисперсия:")
    algos = sorted(algos)

    dct_keys = np.array(sorted(dct.keys()))
    for algo in algos:
        indexes = list(map(lambda x: x[0], filter(lambda x: algo in x[1], zip(range(0, 1000), dct_keys))))
        filtered_var = var_arary[indexes]
        print(algo)
        for i in sorted(mapper):
            print(i, end=" ")
        print()
        for i in np.var(filtered_var, axis=0):
            print(i, end=" ")
        print()
