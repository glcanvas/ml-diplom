import gain
import image_loader as il
from torch.utils.data import DataLoader

gain_model = gain.AttentionGAIN()

loader = il.DatasetLoader.initial()
train = loader.load_tensors(0, 1)
test = loader.load_tensors(0, 1)

train_set = DataLoader(il.ImageDataset(train), batch_size=1, shuffle=True, num_workers=0)
test_set = DataLoader(il.ImageDataset(test), batch_size=1, shuffle=True, num_workers=0)

rds = {'train': train_set, 'test': train_set}

if __name__ == "__main__":
    gain_model.train(rds, 50)
    gain_model.test(rds)
