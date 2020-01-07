import image_loader as il
from torch.utils.data import DataLoader
import gain
import property as P
import sys

if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = int(parsed.gpu)
    parsed_description = parsed.description
    train_left = int(parsed.train_left)
    train_right = int(parsed.train_right)
    test_left = int(parsed.test_left)
    test_right = int(parsed.test_right)
    train_segments_count = int(parsed.segments)
    use_am_loss = parsed.am_loss.lower() == "true"
    pre_train = int(parsed.pre_train)

    description = "{}_train_left-{},train_segments-{},train_right-{},test_left-{},test_right-{},am_loss-{}," \
                  "pre_train-{}" \
        .format(parsed_description,
                train_left,
                train_segments_count,
                train_right,
                test_left,
                test_right,
                use_am_loss,
                pre_train
                )

    P.initialize_log_name("gain_ngxbac_" + description)

    try:
        gain = gain.GAIN(description, gpu=True, device=gpu, use_am=use_am_loss)

        loader = il.DatasetLoader.initial()
        train_segments = loader.load_tensors(train_left, train_segments_count, train_segments_count)
        train_classifier = loader.load_tensors(train_segments_count, train_right, 0)
        test = loader.load_tensors(test_left, test_right)

        train_segments_set = DataLoader(il.ImageDataset(train_segments), batch_size=10, shuffle=True)
        train_classifier_set = DataLoader(il.ImageDataset(train_classifier), batch_size=10, shuffle=True)
        test_set = DataLoader(il.ImageDataset(test), batch_size=10)

        gain.train_model(train_segments_set, train_classifier_set, test_set, pre_train_epoch=pre_train)
    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        raise e
