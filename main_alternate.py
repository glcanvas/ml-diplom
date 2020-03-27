import image_loader as il
from torch.utils.data import DataLoader
import property as P
import sys
import traceback
import am_model as ss
import alternate_attention_module_train as st
import os

if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu = 0
    parsed_description = parsed.description
    pre_train = int(parsed.pre_train)
    train_set_size = int(parsed.train_set)
    epochs = int(parsed.epochs)

    run_name = parsed.run_name
    algorithm_name = parsed.algorithm_name
    left_class_number = int(parsed.left_class_number)
    right_class_number = int(parsed.right_class_number)
    freeze_list = parsed.freeze_list
    classifier_learning_rate = float(parsed.classifier_learning_rate)
    attention_module_learning_rate = float(parsed.attention_module_learning_rate)

    classes = right_class_number - left_class_number

    description = "description-{},train_set-{},epochs-{},l-{},r-{},clr-{},amlr-{},freeze-{}".format(
        parsed_description,
        train_set_size,
        epochs,
        left_class_number,
        right_class_number,
        classifier_learning_rate,
        attention_module_learning_rate,
        freeze_list
    )
    P.initialize_log_name(run_name, algorithm_name, description)

    P.write_to_log("description={}".format(description))
    P.write_to_log("classes={}".format(classes))
    P.write_to_log("run=" + run_name)
    P.write_to_log("algorithm_name=" + algorithm_name)

    log_name, log_dir = os.path.basename(P.log)[:-4], os.path.dirname(P.log)

    snapshots_path = os.path.join(log_dir, log_name)
    os.makedirs(snapshots_path, exist_ok=True)

    try:
        segments_set, test_set = il.load_data(train_set_size)

        train_segments_set = DataLoader(il.ImageDataset(segments_set), batch_size=5, shuffle=True)
        print("ok")
        test_set = DataLoader(il.ImageDataset(test_set), batch_size=5)
        print("ok")

        am_model = ss.build_attention_module_model(classes)

        P.write_to_log(am_model)
        sam_train = st.AlternateModuleTrain(am_model, train_segments_set, test_set,
                                            classes=classes,
                                            pre_train_epochs=pre_train,
                                            gpu_device=gpu,
                                            train_epochs=epochs,
                                            save_train_logs_epochs=4,
                                            test_each_epoch=4,
                                            left_class_number=left_class_number,
                                            right_class_number=right_class_number,
                                            description=run_name + "_" + description,
                                            snapshot_elements_count=20,
                                            snapshot_dir=snapshots_path,
                                            classifier_learning_rate=classifier_learning_rate,
                                            attention_module_learning_rate=attention_module_learning_rate)

        # если только и делаем что делаем претрейн, то надо обучать ВСЕ-ВСЕ веса
        is_used_only_one_loss = pre_train >= epochs
        sam_train.train(is_used_only_one_loss)

    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
