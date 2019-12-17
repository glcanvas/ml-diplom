import image_loader as il
from torch.utils.data import DataLoader
import gain
import property as P

if __name__ == "__main__":
    try:
        gain = gain.AttentionGAIN("all-segments-exists", 5, gpu=True)

        loader = il.DatasetLoader.initial()
        train = loader.load_tensors(0, 2000, 2000)
        test = loader.load_tensors(2000, 2592)

        train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=False, num_workers=0)
        test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

        gain.train({'train': train_set, 'test': test_set}, 100, 4)
    except BaseException as e:
        g = e
        print("AAAAA", e, type(e))
        P.write_to_log("AAAAAAAA!!!!", e, type(e))
    finally:
        print("QQQQQQ")
        P.write_to_log("QQQQQQQQ")
