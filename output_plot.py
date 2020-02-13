"""draw plot of accuracy"""
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import traceback

# plot_images_path = "/home/nikita/PycharmProjects/ml-diplom/sam_plots"
# logs_path = "/home/nikita/PycharmProjects/ml-data/logs"

stupid_flag = False
base_data_dir = "/home/nikita/PycharmProjects"
if os.path.exists("/media/disk1/nduginec"):
    base_data_dir = "/media/disk1/nduginec"
elif os.path.exists("/media/disk2/nduginec"):
    base_data_dir = "/media/disk2/nduginec"
    stupid_flag = True

data_inputs_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input"
data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3"

cache_data_inputs_path = base_data_dir + "/ISIC2018_Task1-2_Training_Input/cached"
cache_data_labels_path = base_data_dir + "/ISIC2018_Task2_Training_GroundTruth_v3/cached"

base_data_dir += "/ml-data" if stupid_flag else ""

logs_path = base_data_dir + "/logs"
plot_images_path = base_data_dir + "/images"
os.makedirs(plot_images_path, exist_ok=True)


def __load_statistics(dct: dict) -> tuple:
    f_measures = {}
    recall = {}
    precision = {}
    for i in range(0, 5):
        key = 'f_1_{}'.format(i)
        f_measures[key] = float(dct[key]) if key in dct else -1
    f_measures['f_1_global'] = float(dct['f_1_global']) if 'f_1_global' in dct else -1

    for i in range(0, 5):
        key = 'recall_{}'.format(i)
        recall[key] = float(dct[key]) if key in dct else -1
    recall['recall_global'] = float(dct['recall_global']) if 'recall_global' in dct else -1

    for i in range(0, 5):
        key = 'precision_{}'.format(i)
        precision[key] = float(dct[key]) if key in dct else -1
    precision['precision_global'] = float(dct['precision_global']) if 'precision_global' in dct else -1
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


def draw_plot_return_avg(ax, title, algo_name, algorithm_list: list, execute_measure_function, color_idx):
    ax.set_title(title)
    for idx, algo in enumerate(algorithm_list):
        legend = "{}_{}".format(algo_name, idx)
        datas = execute_measure_function(algo)
        indexes = [i for i, _ in enumerate(datas)]
        plot_color = [0, 0, 0]
        plot_color[color_idx] = 1
        plot_color[(color_idx + 1) % 3] = idx * 1 / len(algorithm_list)
        ax.plot(indexes, datas, label=legend, color=plot_color)

    lists = []
    for _, algo in enumerate(algorithm_list):
        lists.append(execute_measure_function(algo))
    result = []
    for epoch in range(0, 200):
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
    return result


def draw_plot_avg(ax, title, algo_name, algorithm_list: list, color: list):
    ax.set_title(title)
    legend = "{}_avg".format(algo_name)
    indexes = [i for i, _ in enumerate(algorithm_list)]
    ax.plot(indexes, algorithm_list, label=legend, color=color)


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


def draw_simple_metrics(axes, algorithms: dict):
    for algo_name, color_idx in zip(algorithms, [0, 1, 2]):
        metrics = [
            'Loss_CL',
            'Loss_M',
            'Loss_Total',
            'Loss_L1',
            'Accuracy_CL'
        ]
        for m_idx, m in enumerate(metrics):
            trains = algorithms[algo_name]['train']
            loss_cl_avg = draw_plot_return_avg(axes[m_idx][0], m + ' Train', algo_name, trains,
                                               get_simple_measure_by_name(m),
                                               color_idx)
            color = [0, 0, 0]
            color[color_idx] = 1
            draw_plot_avg(axes[m_idx][1], m + ' AVG', algo_name, loss_cl_avg, color)

            tests = algorithms[algo_name]['test']
            loss_cl_avg = draw_plot_return_avg(axes[m_idx][2], m + ' Test', algo_name, tests,
                                               get_simple_measure_by_name(m),
                                               color_idx)
            color = [0, 0, 0]
            color[color_idx] = 1
            draw_plot_avg(axes[m_idx][3], m + ' AVG', algo_name, loss_cl_avg, color)


def draw_hard_metrics(axes, algorithms: dict):
    for algo_name, color_idx in zip(algorithms, [0, 1, 2]):
        metrics = {
            'f_measures': [
                'f_1_0',
                'f_1_1',
                'f_1_2',
                'f_1_3',
                'f_1_4,',
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
        for m_idx, m in enumerate(metrics):
            m_idx *= 6
            for sub_metrics_idx, sub_metrics in enumerate(metrics[m]):
                m_idx_1 = m_idx + sub_metrics_idx
                trains = algorithms[algo_name]['train']
                loss_cl_avg = draw_plot_return_avg(axes[m_idx_1 + 5][0], sub_metrics + ' Train', algo_name, trains,
                                                   get_hard_measure_by_name(m, sub_metrics),
                                                   color_idx)
                color = [0, 0, 0]
                color[color_idx] = 1
                draw_plot_avg(axes[m_idx_1 + 5][1], m + ' AVG', algo_name, loss_cl_avg, color)

                tests = algorithms[algo_name]['test']
                loss_cl_avg = draw_plot_return_avg(axes[m_idx_1 + 5][2], sub_metrics + ' Test', algo_name, tests,
                                                   get_hard_measure_by_name(m, sub_metrics),
                                                   color_idx)
                color = [0, 0, 0]
                color[color_idx] = 1
                draw_plot_avg(axes[m_idx_1 + 5][3], m + ' AVG', algo_name, loss_cl_avg, color)


def visualize_algorithms(algorithms: dict, run_name: str):
    fig, axes = plt.subplots(23, 4, figsize=(50, 100))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.5, top=2)

    draw_simple_metrics(axes, algorithms)
    draw_hard_metrics(axes, algorithms)
    for ax in axes.flat:
        ax.set(xlabel='epoch', ylabel='value')
        ax.legend(loc='upper left')

    plt.savefig(os.path.join(plot_images_path, run_name.replace(".", "_")), bbox_inches='tight')
    plt.show()


def parse_run(run_number="run_01"):
    algorithms = {}
    for current_path, folder, file_name in os.walk(os.path.join(logs_path, run_number)):
        for algorithm in folder:
            algorithm_list_test = []
            algorithm_list_train = []
            algorithms[algorithm] = {'test': algorithm_list_test, 'train': algorithm_list_train}

            for executed_algorithm_path, _, executed_algorithm_list in os.walk(os.path.join(current_path, algorithm)):
                for executed_algorithm in executed_algorithm_list:
                    # executed_algorithm_path = os.path.join(executed_algorithm_path, executed_algorithm)
                    log_path = os.path.join(current_path, algorithm, executed_algorithm)
                    print('BEGIN READ ' + log_path)
                    train, test, _ = __load_file(log_path, sam_parse_train,
                                                 sam_parse_train)
                    print('END READ ' + log_path)
                    algorithm_list_test.append(test)
                    algorithm_list_train.append(train)
    visualize_algorithms(algorithms, run_number)


if __name__ == "__main__":
    runs = [
        "RUN_01",
        "RUN_02",
        "RUN_03",
        "RUN_04",
        "RUN_05",
        "RUN_06",
        "RUN_07"
    ]
    for i in runs:
        parse_run(i)
