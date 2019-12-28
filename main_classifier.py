import image_loader as il
from torch.utils.data import DataLoader
import classifier
import property as P
import sys


if __name__ == "__main__":
    parsed = P.parse_input_commands().parse_args(sys.argv[1:])

    description = parsed.description
    train_left = int(parsed.train_left)
    train_right = int(parsed.train_right)
    test_left = int(parsed.test_left)
    test_right = int(parsed.test_right)

    try:
        clf = classifier.Classifier(description, 5, gpu=True)

        loader = il.DatasetLoader.initial()
        train = loader.load_tensors(train_left, train_right)
        test = loader.load_tensors(test_left, test_right)

        train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
        test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

        clf.train(4, train_set, test_set, 100)
    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        raise e
