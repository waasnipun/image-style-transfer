from torch.utils.data import Dataset
from PIL import Image

class StyleTransferDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target, label = self.data[idx]
        img = Image.open(img).convert('RGB')
        target = Image.open(target).convert('RGB')

        if self.transform:
            img = self.transform(img)
            target = self.transform(target)

        return img, target, label
    