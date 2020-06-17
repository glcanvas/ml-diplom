import matplotlib.pyplot as plt
import random

EXPERIMENT_SET = [

    # baselines
    {
        'symbol': '.',
        'label': 'resnet18, 1e-3, bceloss',
        'values': [
            ("resnet18+baseline", 0.327),
        ]
    },

    {
        'symbol': 'o',
        'label': 'resnet18, 1e-3, soft-f1',
        'values': [
            ("resnet18+baseline", 0.399),
        ]
    },

    {
        'symbol': '^',
        'label': 'resnet18, 1e-4, bceloss',
        'values': [
            ("resnet18+baseline", 0.385),
        ]
    },

    {
        'symbol': 'v',
        'label': 'resnet18, 1e-4, soft-f1',
        'values': [
            ("resnet18+baseline", 0.265),
        ]
    },

    {
        'symbol': '<',
        'label': 'resnet34, 1e-3, bceloss',
        'values': [
            ("resnet34+baseline", 0.276),
        ]
    },

    {
        'symbol': '>',
        'label': 'resnet34, 1e-3, focal-loss;0.5;0',
        'values': [
            ("resnet34+baseline", 0.312),
        ]
    },

    # strategies
    {
        'symbol': '1',
        'label': 'resnet18, 1e-3, bceloss, bceloss, product',
        'values': [
            ('resnet_18+100-50', 0.379),
            ("resnet18+am", 0.362),
            ("resnet18_1-1", 0.320)
        ]
    },

    {
        'symbol': '2',
        'label': 'resnet18, 1e-3, bceloss, bceloss, sum',
        'values': [
            ('resnet_18+100-50', 0.394),
            ("resnet18+am", 0.321),
            ("resnet18_1-1", 0.326)
        ]
    },

    {
        'symbol': '3',
        'label': 'resnet18, 1e-4, bceloss, bceloss, product',
        'values': [
            ("resnet18+1-1", 0.425),
            ("resnet18+am", 0.389),
            ("resnet18_1-1", 0.377)
        ]
    },

    {
        'symbol': '4',
        'label': 'resnet18, 1e-4, bceloss, bceloss, sum',
        'values': [
            ("resnet18+1-1", 0.425),
            ("resnet18+100-50", 0.414),
            ("resnet18+", 0.398),
            ("resnet18_1-1", 0.364),
        ]
    },

    {
        'symbol': 's',
        'label': 'resnet18, 1e-3, bceloss, bceloss, conv sum',
        'values': [
            ("resnet18+100-50", 0.325),
            ("resnet18+am", 0.314),
            ("resnet18*1-1", 0.358),
        ]
    },

    {
        'symbol': 'p',
        'label': 'resnet18, 1e-3, soft-f1, soft-f1, conv sum',
        'values': [
            ("resnet18+100-50", 0.384),
            ("resnet18*1-1", 0.408),
            ("resnet18+am", 0.399),
        ]
    },

    {
        'symbol': 'P',
        'label': 'resnet18, 1e-4, bceloss, bceloss, conv product',
        'values': [
            ("resnet18+am", 0.340),
            ("resnet18+100-50", 0.393),
        ]
    },

    {
        'symbol': '*',
        'label': 'resnet18, 1e-4, soft-f1, soft-f1, conv sum',
        'values': [
            ("resnet18+100-50", 0.408),
            ("resnet18*1-1", 0.414),
            ("resnet18+am", 0.406),
        ]
    },

    {
        'symbol': '+',
        'label': 'resnet18, 1e-4, soft-f1, soft-f1, conv product',
        'values': [
            ("resnet18+am", 0.424),
            ("resnet18*1-1", 0.398),
        ]
    },

    {
        'symbol': 'x',
        'label': 'resnet34, 1e-3, bceloss, bceloss, product',
        'values': [
            ("resnet34+100-50", 0.334),
            ("resnet34+am", 0.321),
            ("resnet34_1-1_product", 0.320)
        ]
    },

    {
        'symbol': 'X',
        'label': 'resnet34, 1e-3, bceloss, bceloss, sum',
        'values': [
            ("resnet34+100-50", 0.378),
            ("resnet34+am", 0.316),
            ("resnet34_1-1", 0.316),
        ]
    },

    {
        'symbol': 'D',
        'label': 'resnet34, 1e-3, bceloss, bceloss, conv product',
        'values': [
            ("resnet34+100-50", 0.378),
            ("resnet34+am", 0.316),
            ("resnet34_1-1", 0.316),
        ]
    },

    {
        'symbol': 'd',
        'label': 'resnet34, 1e-3, bceloss, bceloss, conv product',
        'values': [
            ("resnet34+am", 0.319),
            ("resnet34+100-50", 0.361),
            ("resnet34*1-1", 0.317),
        ]
    },

    {
        'symbol': r'$ @ $',
        'label': 'resnet34, 1e-3, bceloss, bceloss, conv sum',
        'values': [
            ("resnet34+am", 0.325),
            ("resnet34*1-1", 0.303),
            ("resnet34+100-50", 0.335),
        ]
    },

    {
        'symbol': r'$ f $',
        'label': 'resnet18, 1e-3, bceloss, bceloss, cbam',
        'values': [
            ("resnet18+am", 0.386),
            ("resnet18*1-1", 0.387),
        ]
    },

    {
        'symbol': r'$ y $',
        'label': 'resnet18, 1e-3, bceloss, focal-loss;all;0, cbam',
        'values': [
            ("resnet18+am", 0.305),
            ("resnet18*1-1", 0.381),
        ]
    },

    {
        'symbol': r'$ T $',
        'label': 'resnet18, 1e-4, soft-f1, bceloss, cbam',
        'values': [
            ("resnet18+am", 0.260),
            ("resnet18*1-1", 0.273),
        ]
    },

    {
        'symbol': r'$ & $',
        'label': 'resnet18, 1e-4, bceloss, bceloss, cbam',
        'values': [
            ("resnet18+am", 0.386),
            ("resnet18*1-1", 0.387),
        ]
    },
]

CONSTANTS = [
    ("BASELINE", "x"),
    ("SIMULTANEOUS_DOUBLE", "<"),
    ("SIMULTANEOUS_SINGLE", ">"),
    ("SEQUENTIAL", "v"),
    ("AM", "o"),
]

VGG_SET = [

    {
        'label': '1e-3, cllf=BCEloss',
        'values': [
            ("VGG16_BASELINE", 0.143),
        ]
    },

    {
        'label': '1e-4, cllf=BCEloss',
        'values': [
            ("VGG16_BASELINE", 0.352),
        ]
    },

    {
        'label': '1e-3, cllf=BCEloss, amlf=BCEloss, cb=декартово произведение с умножением',
        'values': [
            ("VGG16_SIMULTANEOUS_DOUBLE", 0.200),
            ("VGG16_SEQUENTIAL", 0.269),
            ("VGG16_AM", 0.149),
        ]
    },
    {
        'label': '1e-4, cllf=BCEloss, amlf=BCEloss, cb=декартово произведение с умножением',
        'values': [
            ("VGG16_SIMULTANEOUS_DOUBLE", 0.327),
            ("VGG16_SEQUENTIAL", 0.387),
            ("VGG16_AM", 0.296),

        ]
    },
    {
        'label': '1e-3, cllf=BCEloss, amlf=BCEloss, cb=декартово произведение со сложением',
        'values': [
            ("VGG16_SIMULTANEOUS_DOUBLE", 0.275),
            ("VGG16_SEQUENTIAL", 0.239),
            ("VGG16_AM", 0.174),

        ]
    },

    {
        'label': '1e-4, cllf=BCEloss, amlf=BCEloss, cb=декартово произведение со сложением',
        'values': [
            ("VGG16_SIMULTANEOUS_DOUBLE", 0.245),
            ("VGG16_SEQUENTIAL", 0.396),
            ("VGG16_AM", 0.207),

        ]
    }
]

RESNET18_SET = [
    {
        'label': "clr=1e-3, cllf=BCELoss",
        'values': [
            ("RESNET18_BASELINE", 0.327),
        ]
    },

    {
        'label': 'clr=1e-3, cllf=BCELoss, amlf=BCELoss, cb=декартово произведение с умножением',
        'values': [
            ("RESNET18_SEQUENTIAL", 0.379),
            ("RESNET18_AM", 0.362),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.320),

        ]
    },

    {
        'label': "clr=1e-3, cllf=BCELoss, amlf=BCELoss, cb=декартово произведение со сложением",
        'values': [
            ("RESNET18_SEQUENTIAL", 0.394),
            ("RESNET18_AM", 0.321),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.326),

        ]
    },
    {
        'label': "clr=1e-3, cllf=BCELoss, amlf=BCELoss, cb=простой блок со сложением",
        'values': [
            ("RESNET18_SEQUENTIAL", 0.325),
            ("RESNET18_AM", 0.314),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.358),

        ]
    },

    {
        'label': "clr=1e-3, cllf=SoftF1Loss",
        'values': [
            ("RESNET18_BASELINE", 0.327),
        ]
    },

    {
        'label': "clr=1e-3, cllf=SoftF1Loss, amlf=SoftF1Loss, cb=простой блок со сложением",
        'values': [
            ("RESNET18_SEQUENTIAL", 0.384),
            ("RESNET18_AM", 0.399),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.408),

        ]
    },

    {
        'label': "clr=1e-4, cllf=BCELoss",
        'values': [
            ("RESNET18_BASELINE", 0.288),
        ]
    },

    {
        'label': "clr=1e-4, cllf=BCELoss, amlf=BCELoss, cb=простой блок с умножением",
        'values': [
            ("RESNET18_AM", 0.340),
            ("RESNET18_SEQUENTIAL", 0.393),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.390),

        ]
    },

    {
        'label': "clr=1e-4, cllf=SoftF1Loss",
        'values': [
            ("RESNET18_BASELINE", 0.265),
        ]
    },

    {
        'label': "clr=1e-4, cllf=SoftF1Loss, amlf=SoftF1Loss, cb=простой блок со сложением",
        'values': [
            ("RESNET18_SEQUENTIAL", 0.408),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.414),
            ("RESNET18_AM", 0.406),
        ]
    },

    {
        'label': "clr=1e-4, cllf=SoftF1Loss, amlf=SoftF1Loss, cb=простой блок с умножением",
        'values': [
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.409),
            ("RESNET18_SEQUENTIAL", 0.375),
            ("RESNET18_AM", 0.400),
        ]
    }

]
# random.Random().shuffle(RESNET18_SET)

RESNET34_SET = [
    {
        'label': "clr=1e-3, cllf=BCELoss",
        'values': [
            ("RESNET34_BASELINE", 0.303),
        ]
    },

    {
        'label': "clr=1e-3, cllf=BCELoss, amlf=BCELoss, cb=простой блок с умножением",
        'values': [
            ("RESNET34_AM", 0.319),
            ("RESNET34_SEQUENTIAL", 0.361),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.317),
        ]
    },

    {
        'label': "clr=1e-3, cllf=BCELoss, amlf=BCELoss, cb=простой блок со сложением",
        'values': [
            ("RESNET34_AM", 0.325),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.276),
            ("RESNET34_SEQUENTIAL", 0.335),

        ]
    },

    {
        'label': "clr=1e-3, cllf=BCELoss, amlf=BCELoss, cb=декартово произведение с умножением",
        'values': [
            ("RESNET34_SEQUENTIAL", 0.334),
            ("RESNET34_AM", 0.321),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.320),

        ]
    },

    {
        'label': "clr=1e-3, cllf=BCELoss, amlf=BCELoss, cb=декартово произведение со сложением",
        'values': [
            ("RESNET34_SEQUENTIAL", 0.378),
            ("RESNET34_AM", 0.316),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.316),
        ]

    },

    {
        'label': "clr=1e-3, cllf=FocalLoss, α=1/2, γ=0",
        'values': [
            ("RESNET34_BASELINE", 0.324),
        ]
    },

    {
        'label': "clr=1e-3, cllf=FocalLoss, α=1/2, γ=0, amlf=SoftF1Loss, cb=декартово произведение с умножением",
        'values': [
            ("RESNET34_SEQUENTIAL", 0.348),
            ("RESNET34_AM", 0.319),
        ]
    },

    {
        'label': "clr=1e-3, cllf=SoftF1Loss",
        'values': [
            ("RESNET34_BASELINE", 0.378),
        ]
    },

    {
        'label': "clr=1e-3, cllf=SoftF1Loss, amlf=SoftF1Loss, cb=простой блок с умножением",
        'values': [
            ("RESNET34_SEQUENTIAL", 0.384),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.386),
            ("RESNET34_AM", 0.384),
        ]
    },

    {
        'label': "clr=1e-3, cllf=SoftF1Loss, amlf=SoftF1Loss, cb=простой блок со сложением",
        'values': [
            ("RESNET34_AM", 0.378),
            ("RESNET34_SEQUENTIAL", 0.385),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.380),
        ]
    },

    {
        'label': "clr=1e-4, cllf=BCELoss",
        'values': [
            ("RESNET34_BASELINE", 0.397),
        ]
    },

    {
        'label': "clr=1e-4, cllf=BCELoss, amlf=BCELoss, cb=простой блок с умножением",
        'values': [
            ("RESNET34_AM", 0.361),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.340),
            ("RESNET34_SEQUENTIAL", 0.363),
        ]
    },

    {
        'label': "clr=1e-4, cllf=BCELoss, amlf=BCELoss, cb=простой блок со сложением",
        'values': [
            ("RESNET34_AM", 0.379),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.361),
            ("RESNET34_SEQUENTIAL", 0.304),
        ]
    },

    {
        'label': "clr=1e-4, amlr=1e-4, cllf=BCELoss, amlf=BCELoss, cb=простой блок со сложением",
        'values': [
            ("RESNET34_AM", 0.379),
            ("RESNET34_SIMULTANEOUS_DOUBLE", 0.379),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.381),

        ]
    },


    {
        'label': "clr=1e-4, amlr=1e-4, cllf=BCELoss, amlf=BCELoss, cb=простой блок с умножением",
        'values': [
            ("RESNET34_AM", 0.361),
            ("RESNET34_SEQUENTIAL", 0.379),
            ("RESNET34_SIMULTANEOUS_DOUBLE", 0.385),
        ]
    },

    {
        'label': "clr=1e-4, amlr=1e-4, cllf=BCELoss, amlf=BCELoss, cb=декартово произведение со сложением",
        'values': [
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.442),
            ("RESNET34_AM", 0.387),
            ("RESNET34_SEQUENTIAL", 0.383),

        ]
    },

    {
        'label': "clr=1e-4, amlr=1e-4, cllf=BCELoss, amlf=BCELoss, cb=декартово произведение с умножением",
        'values': [
            ("RESNET34_SEQUENTIAL", 0.403),
            ("RESNET34_SIMULTANEOUS_DOUBLE", 0.439),
            ("RESNET34_AM", 0.405),
        ]
    },

    {
        'label': "clr=1e-4, amlr=1e-4, cllf=SoftF1Loss",
        'values': [
            ("RESNET34_BASELINE", 0.449),
        ]
    },

    {
        'label': "clr=1e-4, amlr=1e-4, cllf=SoftF1Loss, amlf=SoftF1Loss, cb=декартово произведение со сложением",
        'values': [
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.425),
            ("RESNET34_SIMULTANEOUS_DOUBLE", 0.467),
            ("RESNET34_AM", 0.433),
            ("RESNET34_SEQUENTIAL", 0.385),
        ]
    },

    {
        'label': "clr=1e-4, amlr=1e-4, cllf=SoftF1Loss, amlf=SoftF1Loss, cb=декартово произведение с умножением",
        'values': [
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.411),
            ("RESNET34_SIMULTANEOUS_DOUBLE", 0.456),
            ("RESNET34_AM", 0.401),
            ("RESNET34_SEQUENTIAL", 0.320),
        ]
    },


]

RESNET_CBAM = [
    {
        'label': "RESNET18, clr=1e-3, cllf=BCELoss",
        'values': [
            ("RESNET18_BASELINE", 0.327),
        ]
    },

    {
        'label': "RESNET18, clr=1e-3, cllf=BCELoss, amlf=BCELoss",
        'values': [
            ("RESNET18_AM", 0.386),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.387),
        ]
    },

    {
        'label': "RESNET18, clr=1e-4, cllf=SoftF1Loss",
        'values': [
            ("RESNET18_BASELINE", 0.265),
        ]
    },

    {
        'label': "RESNET18, clr=1e-4, cllf=SoftF1Loss, amlf=BCELoss",
        'values': [
            ("RESNET18_AM", 0.388),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.394),
        ]
    },

    {
        'label': "RESNET18, clr=1e-3, cllf=BCELoss",
        'values': [
            ("RESNET18_BASELINE", 0.327),
        ]
    },

    {
        'label': "RESNET18, clr=1e-3, cllf=BCELoss, amlf=FocalLoss, α=обратная классовая частота, γ=0",
        'values': [
            ("RESNET18_AM", 0.305),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.381),
        ]
    },

    {
        'label': "RESNET34, clr=1e-3, cllf=SoftF1Loss",
        'values': [
            ("RESNET34_BASELINE", 0.378),
        ]
    },

    {
        'label': "RESNET34, clr=1e-3, cllf=SoftF1Loss, amlf=SoftF1Loss",
        'values': [
            ("RESNET34_AM", 0.413),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.415),

        ]
    },

    {
        'label': "RESNET18, clr=1e-3, cllf=SoftF1Loss",
        'values': [
            ("RESNET18_BASELINE", 0.399),
        ]
    },

    {
        'label': "RESNET18, clr=1e-3, cllf=SoftF1Loss, amlf=SoftF1Loss",
        'values': [
            ("RESNET18_AM", 0.401),
            ("RESNET18_SIMULTANEOUS_SINGLE", 0.424),
        ]
    },

    {
        'label': "RESNET34, clr=1e-3, cllf=BCELoss",
        'values': [
            ("RESNET34_BASELINE", 0.303),
        ]
    },

    {
        'label': "RESNET34, clr=1e-3, cllf=BCELoss, amlf=SoftF1Loss",
        'values': [
            ("RESNET34_AM", 0.326),
            ("RESNET34_SIMULTANEOUS_SINGLE", 0.339),
        ]
    },

]


def found_marker(full: str) -> str:
    for (name, marker) in CONSTANTS:
        if name.lower() in full.lower():
            return marker
    raise Exception("AAAA")


def sort_by_name(lst):
    for i in lst:
        i['values'] = sorted(i['values'], key=lambda x: x[1])


if __name__ == "__main__":
    sort_by_name(RESNET18_SET)
    sort_by_name(RESNET34_SET)
    sort_by_name(VGG_SET)
    sort_by_name(RESNET_CBAM)
    for (name, marker) in CONSTANTS:
        plt.plot([], [], marker=marker, label=name, color='black')

    # plt.rcParams["figure.figsize"] = (30, 50)
    global_index = 1
    x_names = []
    y_names = []
    line = []

    MY_BLOCKS = VGG_SET #RESNET18_SET + RESNET34_SET + VGG_SET + RESNET_CBAM

    max_in_set = list(map(lambda x: max(map(lambda y: y[1], x['values'])), MY_BLOCKS))
    value_before = list(reversed(sorted(max_in_set)))
    value_before = value_before[8] if len(value_before) > 8 else value_before[-1]

    sets_labels = [True if i >= value_before else False for i in max_in_set]
    # sets_labels = [True for i in max_in_set]

    for idx, experiment in enumerate(MY_BLOCKS):
        label = experiment['label']
        elements = experiment['values']
        n, _ = elements[0]

        ###############
        """if "RESNET" in label and len(elements) > 1:
            label = label + ", CBAM"
        elif "RESNET" in label:
            pass
        else:
            label = n.split("_")[0] + ", " + label
        """
        ###############
        if sets_labels[idx]:
            color = plt.plot([], [], label=label)[0].get_color()
        else:
            color = plt.plot([], [])[0].get_color()

        # color = plt.plot([], [], label=label)[0].get_color()
        for (name, value) in elements:
            marker = found_marker(name)
            full_index = name + " " + label

            x_names.append(full_index)
            y_names.append(value)
            value = value

            label = full_index
            xx = plt.plot(global_index, value, marker=marker, color=color)
            line.append((xx[0].get_color(), value))
            global_index += 1

    diff = global_index // 8  # 8 -- vgg
    for pos, (color, value) in enumerate(line):
        plt.plot([i for i in range(pos - diff, pos + diff * 2 + 1)],  # 3 + 1 -- vgg
                 [value for i in range(pos - diff, pos + diff * 2 + 1)], color=color)

    plt.legend(numpoints=1, bbox_to_anchor=(1.0, -0.01), fontsize=8.5)  # 1.1 # prop={'size': 15}

    plt.xlim(0, global_index + 1)
    # plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45)

    plt.gca().set_xticklabels([])
    # plt.gca().set_xticklabels(x_names)
    # plt.gca().set_yticklabels(y_names)
    plt.savefig("greate_image_vgg", bbox_inches='tight')
