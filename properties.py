data_inputs_path = "/home/nikita/PycharmProjects/ISIC2018_Task1-2_Training_Input"
data_labels_path = "/home/nikita/PycharmProjects/ISIC2018_Task2_Training_GroundTruth_v3"

tensor_train_path = "/home/nikita/PycharmProjects/tensors/test"
tensor_validate_path = "/home/nikita/PycharmProjects/tensors/validate"

# for image_loader
classes = [
    'pigment_network',
    'streaks',
    'globules',
    'milia_like_cyst',
    'negative_network'
]
field_id = 'id'
field_train = 'train'
image_size = 224
num_workers = 5

labels_number = len(classes) # 5
train_size = 200
test_size = 100
batch_size = 8
