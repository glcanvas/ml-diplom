import image_loader as il
from torch.utils.data import DataLoader
import property as P
import sys
import torchvision.models as m
import traceback
import classifier as cl

classes = 5
class_number = None

if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    parsed_description = parsed.description
    train_set_size = int(parsed.train_set)
    epochs = int(parsed.epochs)
    run_name = parsed.run_name
    algorithm_name = parsed.algorithm_name
    use_class_number = int(parsed.use_class_number)
    if use_class_number != -1:
        classes = 1
        class_number = use_class_number

    description = "description-{},train_set-{},epochs-{},classes-{},class_number-{}".format(
        parsed_description,
        train_set_size,
        epochs,
        classes,
        class_number
    )

    P.initialize_log_name(run_name, algorithm_name, description)

    P.write_to_log("description={}".format(description))
    P.write_to_log("classes={}".format(classes))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    try:
        segments_set, test_set = il.load_data_2(train_set_size)

        train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=5, shuffle=True)
        print("ok")
        test_set = DataLoader(il.ImageDataset(test_set), batch_size=5)
        print("ok")

        model = m.vgg16(pretrained=True)
        P.write_to_log(model)

        classifier = cl.Classifier(description=description, classes=classes, gpu=True, gpu_device=gpu, model=model)
        classifier.train(epochs,
                         test_each_epochs=4,
                         save_test_roc_each_epochs=-1,
                         save_train_roc_each_epochs=-1,
                         train_data_set=train_segments_set,
                         test_data_set=test_set,
                         class_number=class_number)

    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
