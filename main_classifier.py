import image_loader as il
from torch.utils.data import DataLoader
import classifier
import property as P

if __name__ == "__main__":
    g = None
    try:
        clf = classifier.Classifier(5, gpu=True)

        loader = il.DatasetLoader.initial()
        train = loader.load_tensors(0, 2000)
        test = loader.load_tensors(2000, 2592)

        train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
        test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

        clf.train(4, train_set, test_set, 100)
    except BaseException as e:
        g = e
        print("AAAAA", e)
        print(type(e))
        P.write_to_log("AAAAAAAA!!!!", e, type(e))
    finally:
        print("QQQQQQ", g)
        P.write_to_log("QQQQQQQQ ex =", g)
