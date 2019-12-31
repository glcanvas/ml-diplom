import image_loader as il
from torch.utils.data import DataLoader
import gain
import property as P
import sys
import torch

if __name__ == "__main__":
    # parsed = P.parse_input_commands().parse_args(sys.argv[1:])
    gpu = 0  # int(parsed.gpu)
    description = ""  # parsed.description
    train_left = 0  # int(parsed.train_left)
    train_right = 100  # int(parsed.train_right)
    test_left = 101  # int(parsed.test_left)
    test_right = 2000  # int(parsed.test_right)
    train_segments_count = 1000  # int(parsed.segments)

    if train_segments_count % 10 != 0:
        raise ValueError("train_segments_count must be multiple of 10")

    P.initialize_log_name("gain_" + description)

    try:
        gain = gain.AttentionGAIN(description, 5, gpu=True, device=gpu, usage_am_loss=False)

        loader = il.DatasetLoader.initial()
        train = loader.load_tensors(train_left, train_right, train_segments_count)
        test = loader.load_tensors(test_left, test_right)

        # здесь важно что train_set не мешается, так как сначала идут картинки с сегментами затем просто картинки
        # так же train_segments_count должно быть кратно 10 !
        train_set = DataLoader(il.ImageDataset(train), batch_size=10)
        test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True)

        gain.train({'train': train_set, 'test': test_set}, 100, 4)
    except BaseException as e:
        print("EXCEPTION", e)
        print(type(e))
        P.write_to_log("EXCEPTION", e, type(e))
        raise e
