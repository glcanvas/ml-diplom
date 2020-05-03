output_plot_logs = """0.4329128571428571 RESNET_50 Loss_CL_AVG
0.0 RESNET_50 Loss_M_AVG
0.0 RESNET_50 Loss_Total_AVG
0.0 RESNET_50 Loss_L1_AVG
0.8184362500000001 RESNET_50 Accuracy_CL_AVG
1.9888500000000002 VGG16 Loss_CL_AVG
0.0 VGG16 Loss_M_AVG
0.0 VGG16 Loss_Total_AVG
0.0 VGG16 Loss_L1_AVG
0.8153422727272729 VGG16 Accuracy_CL_AVG
0.0 RESNET_50 f_1_0_AVG
0.0 RESNET_50 f_1_1_AVG
0.0 RESNET_50 f_1_2_AVG
0.027609999999999996 RESNET_50 f_1_3_AVG
0.744076 RESNET_50 f_1_4_AVG
0.148815 RESNET_50 f_1_global_AVG
0.0 RESNET_50 recall_0_AVG
0.0 RESNET_50 recall_1_AVG
0.0 RESNET_50 recall_2_AVG
0.021276666666666666 RESNET_50 recall_3_AVG
0.79178 RESNET_50 recall_4_AVG
0.15835714285714286 RESNET_50 recall_global_AVG
0.0 RESNET_50 precision_0_AVG
0.0 RESNET_50 precision_1_AVG
0.0 RESNET_50 precision_2_AVG
0.18545099999999998 RESNET_50 precision_3_AVG
0.7193820000000001 RESNET_50 precision_4_AVG
0.18096799999999996 RESNET_50 precision_global_AVG
0.22995470588235295 VGG16 f_1_0_AVG
0.118133 VGG16 f_1_1_AVG
0.32816999999999996 VGG16 f_1_2_AVG
0.40730500000000003 VGG16 f_1_3_AVG
0.7525400000000001 VGG16 f_1_4_AVG
0.352036 VGG16 f_1_global_AVG
0.1913770588235294 VGG16 recall_0_AVG
0.1047585 VGG16 recall_1_AVG
0.312298 VGG16 recall_2_AVG
0.3989595 VGG16 recall_3_AVG
0.7959768181818181 VGG16 recall_4_AVG
0.3396405 VGG16 recall_global_AVG
0.38545500000000005 VGG16 precision_0_AVG
0.180999 VGG16 precision_1_AVG
0.42043949999999997 VGG16 precision_2_AVG
0.47006666666666663 VGG16 precision_3_AVG
0.7517065 VGG16 precision_4_AVG
0.41720949999999996 VGG16 precision_global_AVG"""

white_list = "Accuracy_CL_AVG f_1_0_AVG f_1_1_AVG f_1_2_AVG f_1_3_AVG f_1_4_AVG f_1_global_AVG".split()
mapper = ["Точность", "F-measure_заболевание_1", "F-measure_заболевание_2", "F-measure_заболевание_3",
          "F-measure_заболевание_4", "F-measure_заболевание_5", "F-measure_общая"]


def map(x):
    for idx, i in enumerate(white_list):
        if i == x:
            return mapper[idx]


dct = dict()
max_len = 1
for l in output_plot_logs.split("\n"):
    v, alg, metric = l.split()
    dct.setdefault(alg, dict())
    v = "{:.3f}".format(float(v))
    if metric not in white_list:
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

print(" dd" * -2)
if __name__ == "__main__":
    pass
