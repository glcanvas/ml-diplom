import image_loader as il
from torch.utils.data import DataLoader
import classifier

# all elements: 2594
clf = classifier.Classifier(5, gpu=True)

loader = il.DatasetLoader.initial()
train = loader.load_tensors(0, 2000)
test = loader.load_tensors(2000, 2592)

train_set = DataLoader(il.ImageDataset(train), batch_size=10, shuffle=True, num_workers=0)
test_set = DataLoader(il.ImageDataset(test), batch_size=10, shuffle=True, num_workers=0)

if __name__ == "__main__":
    clf.train(2, train_set, test_set, 4, 10, 1)