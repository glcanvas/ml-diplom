"""draw plot of accuracy"""
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import traceback

"""stupid_flag = False
base_data_dir = "/home/nikita/PycharmProjects"
if os.path.exists("/media/disk1/nduginec"):
    base_data_dir = "/media/disk1/nduginec"
elif os.path.exists("/media/disk2/nduginec"):
    base_data_dir = "/media/disk2/nduginec"
    stupid_flag = True

base_data_dir += "/ml-data" if stupid_flag else ""
"""
base_data_dir = "D://diplom-base-dir"
logs_path = base_data_dir + "/trust_logs"
plot_images_path = base_data_dir + "/images"
os.makedirs(plot_images_path, exist_ok=True)


def __load_statistics(dct: dict) -> tuple:
    f_measures = {}
    recall = {}
    precision = {}
    for i in range(0, 5):
        key = 'f_1_{}'.format(i)
        f_measures[key] = float(dct[key]) if key in dct else 0
    f_measures['f_1_global'] = float(dct['f_1_global']) if 'f_1_global' in dct else 0

    for i in range(0, 5):
        key = 'recall_{}'.format(i)
        recall[key] = float(dct[key]) if key in dct else 0
    recall['recall_global'] = float(dct['recall_global']) if 'recall_global' in dct else 0

    for i in range(0, 5):
        key = 'precision_{}'.format(i)
        precision[key] = float(dct[key]) if key in dct else 0
    precision['precision_global'] = float(dct['precision_global']) if 'precision_global' in dct else 0
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
    Loss_CL = float(dct['Loss_CL']) if 'Loss_CL' in dct else 0
    Loss_M = float(dct['Loss_M']) if 'Loss_M' in dct else 0
    Loss_L1 = float(dct['Loss_L1']) if 'Loss_L1' in dct else 0
    Loss_Total = float(dct['Loss_Total']) if 'Loss_Total' in dct else 0
    Accuracy_CL = float(dct['Accuracy_CL']) if 'Accuracy_CL' in dct else 0
    f_measures, recall, precision = __load_statistics(dct)
    return {'Loss_CL': Loss_CL, 'Loss_M': Loss_M, 'Loss_L1': Loss_L1,
            'Loss_Total': Loss_Total, 'Accuracy_CL': Accuracy_CL, 'f_measures': f_measures, 'recall': recall,
            'precision': precision}


def __load_file(file_path: str, train_parse, test_parse):
    train = []
    test = []
    train_index = 1
    test_index = 1
    with open(file_path) as f:
        for l in f.readlines():
            if "TRAIN" in l:
                train.append(train_parse(l))
                train_index += 1
            elif "TEST" in l:
                test.append(test_parse(l))
                test_index += 1
        file_name = os.path.basename(f.name)
    return train, test, file_name


def draw_plot_return_avg(ax, title, algo_name, algorithm_list: list, execute_measure_function, plot_color,
                         mull_index=1):
    ax.set_title(title)

    use_legend = True
    max_value = []
    for idx, algo in enumerate(algorithm_list):
        legend = algo_name
        datas = execute_measure_function(algo)

        if len(datas) > 0:
            max_value.append((max(datas), algo_name + "_" + str(idx), title.replace(" ", "_")))

        indexes = [i * mull_index for i, _ in enumerate(datas)]
        if use_legend:
            ax.plot(indexes, datas, label=legend, color=plot_color)
            use_legend = False
        else:
            ax.plot(indexes, datas, color=plot_color)
    lists = []
    for _, algo in enumerate(algorithm_list):
        lists.append(execute_measure_function(algo))
    result = []
    for epoch in range(0, 151):
        cnt = 0
        sm = 0.0
        for idx in range(len(lists)):
            if len(lists[idx]) == 0 or len(lists[idx]) <= epoch:
                continue
            sm += lists[idx][epoch]
            cnt += 1
        if cnt == 0:
            result.append(0)
        else:
            result.append(sm / cnt)
    return result, max_value


def draw_plot_avg(ax, title: str, algo_name, algorithm_list: list, color: list, mull_index=1):
    ax.set_title(title)
    legend = algo_name
    algorithm_list = [] if len(algorithm_list) == 0 else [algorithm_list[0]] + list(
        filter(lambda x: x > 0.0, algorithm_list[1:]))
    indexes = [i * mull_index for i, _ in enumerate(algorithm_list)]

    max_values = [max(algorithm_list) for _ in algorithm_list]
    ax.plot(indexes, max_values, label=legend + " max", linestyle=':',
            linewidth=2,
            color=(abs(color[0] - 0.4), abs(color[1] - 0.4), abs(color[2] - 0.4)))

    ax.plot(indexes, algorithm_list, label=legend, color=color)
    return max_values[0], legend, title.replace(" ", "_")


def get_simple_measure_by_name(name):
    def inner(lst):
        res = []
        for i in lst:
            if name in i:
                res.append(i[name])
        return res

    return inner


def get_hard_measure_by_name(name1, name2):
    def inner(lst):
        res = []
        for i in lst:
            if name1 in i and name2 in i[name1]:
                res.append(i[name1][name2])
        return res

    return inner


def to_color_array(color_idx) -> list:
    color = [0, 0, 0]
    for i in range(0, 3):
        if (1 << i) & color_idx:
            color[i] = 1
    return color


def draw_simple_metrics(axes, algorithms: dict):
    for algo_name, color_idx in zip(algorithms, [0, 1, 2, 3, 4, 5, 6]):
        metrics = [
            'Loss_CL',
            'Loss_M',
            'Loss_Total',
            'Loss_L1',
            'Accuracy_CL'
        ]
        res = []
        for m_idx, m in enumerate(metrics):
            trains = algorithms[algo_name]['train']
            loss_cl_avg, _ = draw_plot_return_avg(axes[m_idx][0], m + ' Train', algo_name, trains,
                                                  get_simple_measure_by_name(m),
                                                  to_color_array(color_idx))
            # color = [0, 0, 0]
            # color[color_idx] = 1
            draw_plot_avg(axes[m_idx][1], m + ' AVG', algo_name, loss_cl_avg, to_color_array(color_idx))

            tests = algorithms[algo_name]['test']
            loss_cl_avg, x = draw_plot_return_avg(axes[m_idx][2], m + ' Test', algo_name, tests,
                                                  get_simple_measure_by_name(m),
                                                  to_color_array(color_idx),
                                                  mull_index=4)
            # color = [0, 0, 0]
            # color[color_idx] = 1
            res.extend(x)
            res.append(draw_plot_avg(axes[m_idx][3], m + ' AVG', algo_name, loss_cl_avg, to_color_array(color_idx),
                                     mull_index=4))
        for i, j, k in res:
            print(i, j, k)


def draw_hard_metrics(axes, algorithms: dict):
    for algo_name, color_idx in zip(algorithms, [0, 1, 2, 3, 4, 5, 6, 7]):
        metrics = {
            'f_measures': [
                'f_1_0',
                'f_1_1',
                'f_1_2',
                'f_1_3',
                'f_1_4',
                'f_1_global'
            ],
            'recall': [
                'recall_0',
                'recall_1',
                'recall_2',
                'recall_3',
                'recall_4',
                'recall_global',
            ],
            'precision': [
                'precision_0',
                'precision_1',
                'precision_2',
                'precision_3',
                'precision_4',
                'precision_global'
            ]
        }
        res = []
        for m_idx, m in enumerate(metrics):
            m_idx *= 6
            for sub_metrics_idx, sub_metrics in enumerate(metrics[m]):
                m_idx_1 = m_idx + sub_metrics_idx
                trains = algorithms[algo_name]['train']
                loss_cl_avg, _ = draw_plot_return_avg(axes[m_idx_1 + 5][0], sub_metrics + ' Train', algo_name, trains,
                                                      get_hard_measure_by_name(m, sub_metrics),
                                                      to_color_array(color_idx))
                # color = [0, 0, 0]
                # color[color_idx] = 1
                draw_plot_avg(axes[m_idx_1 + 5][1], sub_metrics + ' AVG', algo_name, loss_cl_avg,
                              to_color_array(color_idx))

                tests = algorithms[algo_name]['test']
                loss_cl_avg, x = draw_plot_return_avg(axes[m_idx_1 + 5][2], sub_metrics + ' Test', algo_name, tests,
                                                      get_hard_measure_by_name(m, sub_metrics),
                                                      to_color_array(color_idx),
                                                      mull_index=4)
                # color = [0, 0, 0]
                # color[color_idx] = 1
                res.extend(x)
                res.append(draw_plot_avg(axes[m_idx_1 + 5][3], sub_metrics + ' AVG', algo_name, loss_cl_avg,
                                         to_color_array(color_idx), mull_index=4))
        for i, j, k in res:
            print(i, j, k)


def visualize_algorithms(algorithms: dict, run_name: str):
    fig, axes = plt.subplots(23, 4, figsize=(50, 100))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=2)

    draw_simple_metrics(axes, algorithms)
    draw_hard_metrics(axes, algorithms)
    for ax in axes.flat:
        ax.set(xlabel='epoch', ylabel='value')
        ax.legend(loc='upper right')

    for ax in axes.flat[12:]:
        ax.set_ylim((0, 1))
    plt.savefig(os.path.join(plot_images_path, run_name.replace(".", "_")), bbox_inches='tight')
    plt.show()


def parse_run(run_number="run_01", use_print: bool = False):
    algorithms = {}
    for current_path, folder, _ in os.walk(os.path.join(logs_path, run_number)):
        # print(folder)
        for algorithm in folder:
            algorithm_list_test = []
            algorithm_list_train = []
            algorithms[algorithm] = {'test': algorithm_list_test, 'train': algorithm_list_train}

            for executed_algorithm_path, _, executed_algorithm_list in os.walk(os.path.join(current_path, algorithm)):
                for executed_algorithm in executed_algorithm_list:
                    # executed_algorithm_path = os.path.join(executed_algorithm_path, executed_algorithm)
                    log_path = os.path.join(current_path, algorithm, executed_algorithm)
                    if '.txt' not in log_path:
                        continue
                    if use_print:
                        print('BEGIN READ ' + log_path)
                    train, test, _ = __load_file(log_path, sam_parse_train,
                                                 sam_parse_train)
                    if use_print:
                        print('END READ ' + log_path)
                    algorithm_list_test.append(test)
                    algorithm_list_train.append(train)
        break
    return algorithms, run_number


def reduce_stat(algorithms: dict, run_number: str):
    print("=" * 50)
    print("=" * 50)
    for name, dct in algorithms.items():
        print("-" * 50)

        s = run_number[run_number.index("CLR-") + 4:]
        print("id = {}".format(run_number[4:7]))
        print("lr = {}".format(s[:s.index("_")]))
        print("algorithm = {}".format(name))
        more_then_140 = 0
        for algo_index, epochs in enumerate(dct['train']):
            if len(epochs) >= 140:
                more_then_140 += 1
        print("count = {}".format(more_then_140))
        print("-" * 50)
    print("=" * 50)
    print("=" * 50)


if __name__ == "__main__":
    runs = [
        # "RUN_500_no_pretrian",
        #"RUN_500_pretained_default",
        #"RUN_500_pretrain_sum",
        # "RUN_501_no_pretrian",
        #"RUN_501_pretained_default",
        # "RUN_501_pretrain_sum",
        # "RUN_502_pretained_default",
        # "RUN_503_pretained_default"
        # "vgg_vs_resnet50"
        #"RUN_1000",
        #"RUN_1001",
        #"RUN_1002",
    ]
    for i in runs:
        a, r_n = parse_run(i)
        visualize_algorithms(a, r_n)
        # reduce_stat(a, r_n)
