prefix = 'ISIC_'
attribute = '_attribute_'
image_size = 224

input_attribute = 'input'
cached_extension = '.torch'

data_inputs_path = "/home/nikita/PycharmProjects/ISIC2018_Task1-2_Training_Input"
data_labels_path = "/home/nikita/PycharmProjects/ISIC2018_Task2_Training_GroundTruth_v3"

cache_data_inputs_path = "/home/nikita/PycharmProjects/ISIC2018_Task1-2_Training_Input/cached"
cache_data_labels_path = "/home/nikita/PycharmProjects/ISIC2018_Task2_Training_GroundTruth_v3/cached"

labels_attributes = [
    # 'streaks',
    # 'negative_network',
    # 'milia_like_cyst',
    # 'globules',
    'pigment_network'
]

labels_attributes_weight = [
    2,
    4,
    8,
    16,
    32
]


