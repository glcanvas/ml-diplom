import voc_loader as vl
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

    log_name, log_dir = os.path.basename(P.log)[:-4], os.path.dirname(P.log)

    snapshots_path = os.path.join(log_dir, log_name)
    os.makedirs(snapshots_path, exist_ok=True)

    try:
        v = vl.VocDataLoader(P.voc_data_path, voc_items, P.voc_list_to_indexes(voc_items))
        train_data_set = DataLoader(vl.VocDataset(v.train_data[0:train_set_size]), batch_size=5, shuffle=True)
        test_data_set = DataLoader(vl.VocDataset(v.test_data[0:test_set_size]), batch_size=5)

        am_model = ss.build_attention_module_model(classes)

        P.write_to_log(am_model)
        sam_train = st.AlternateModuleTrain(am_model, train_data_set, test_data_set,
                                            classes=classes,
                                            pre_train_epochs=pre_train,
                                            gpu_device=gpu,
                                            train_epochs=epochs,
                                            save_train_logs_epochs=4,
                                            test_each_epoch=4,
                                            description=run_name + "_" + description,
                                            snapshot_elements_count=20,
                                            snapshot_dir=snapshots_path)
        sam_train.train()

    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        traceback.print_stack()

        raise e
