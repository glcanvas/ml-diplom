import gain
import image_loader as il
from torch.utils.data import DataLoader

gain_model = gain.AttentionGAIN(gpu=True)

loader = il.DatasetLoader.initial()
train = loader.load_tensors(0, 1)
test = loader.load_tensors(0, 1)

train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

rds = {'train': train_set, 'test': train_set}

if __name__ == "__main__":
    gain_model.train(rds, 50)
    gain_model.test(rds)
    """
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    a = [a] * 3
    a = [a] * 4
    a = torch.tensor(a)

    b = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    b = [b]
    b = [b] * 4
    b = torch.tensor(b)
    print(a.shape)
    print(b.shape)
    print(a - b)
    """
