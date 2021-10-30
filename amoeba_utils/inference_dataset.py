

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



def inference_transforms():
    ttransforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return ttransforms



### we probably will have a lot of variations on this but for now let it be
class InferenceDataset(Dataset):
    def __init__(self, images):
        self.transform = inference_transforms()
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        labels = 0  ### we can literally bs this
        if self.transform:
            image = self.transform(image)
        return image, labels