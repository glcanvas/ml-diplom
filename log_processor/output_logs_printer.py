#white_list = "Accuracy_CL f_1_0 f_1_1 f_1_2 f_1_3 f_1_4 f_1_global".split()
#mapper = ["Точность", "F-measure_заболевание_1", "F-measure_заболевание_2", "F-measure_заболевание_3",
#          "F-measure_заболевание_4", "F-measure_заболевание_5", "F-measure_общая"]


white_list = "f_1_global".split()
mapper = ["F-measure_общая"]


def map(x):
    for idx, i in enumerate(white_list):
        if i in x:
            return mapper[idx]


def process_values(output_plot_logs_dir: str):
    with open(output_plot_logs_dir, 'r') as f:
        output_plot_logs = "\n".join(f.readlines())

    dct = dict()
    max_len = 1
    for l in output_plot_logs.split("\n"):
        if len(l.split()) != 3:
            continue
        v, alg, metric = l.split()
        if '_|' in alg:
            continue
        dct.setdefault(alg, dict())
        v = "{:.3f}".format(float(v))
        found = False
        for m in white_list:
            if m in metric:
                found = True
        if not found:
            continue
        metric = map(metric)
        dct[alg][metric] = v
        max_len = max([len(v), len(alg), len(metric), max_len])

    for k in dct.keys():
        print("g" * max_len, end=" ")
        for m in sorted(dct[k].keys()):
            print(m + " " * (max_len - len(m)), end=" ")
        print()
        break

    for k in dct.keys():
        print(k, " " * (max_len - len(k)), end=" ")
        for m in sorted(dct[k].keys()):
            print(dct[k][m] + " " * (max_len - len(dct[k][m])), end=" ")
        print()
