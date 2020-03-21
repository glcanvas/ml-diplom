import voc_loader as vl
from torch.utils.data import DataLoader
import property as P
import sys
import torchvision.models as m
import traceback
import classifier_vgg16_train as cl
import os

if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu = 0
    parsed_description = parsed.description
    train_set_size = int(parsed.train_set)
    test_set_size = int(parsed.test_set)
    epochs = int(parsed.epochs)
    run_name = parsed.run_name
    algorithm_name = parsed.algorithm_name
    voc_items = list(filter(lambda x: len(x) > 0, parsed.voc_items.split(",")))

    description = "description-{},train_set-{},test_set-{},epochs-{},voc-{}".format(
        parsed_description,
        train_set_size,
        test_set_size,
        epochs,
        ",".join(voc_items)
    )

    classes = len(voc_items)

    P.initialize_log_name(run_name, algorithm_name, description)

    P.write_to_log("description={}".format(description))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    try:
        v = vl.VocDataLoader(P.voc_data_path, voc_items, P.voc_list_to_indexes(voc_items))
        train_data_set = DataLoader(vl.VocDataset(v.train_data[0:train_set_size]), batch_size=5, shuffle=True)
        test_data_set = DataLoader(vl.VocDataset(v.test_data[0:test_set_size]), batch_size=5)

        model = m.vgg16(pretrained=True)
        P.write_to_log(model)

        classifier = cl.Classifier(model,
                                   train_data_set,
                                   test_data_set,
                                   classes=classes,
                                   test_each_epoch=4,
                                   gpu_device=gpu,
                                   train_epochs=epochs,
                                   description=run_name + "_" + description)
        classifier.train()

    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
