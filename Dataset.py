import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import time
import math

path = 'images'


class ChestXRay(Dataset):

    def __init__(self, image_file, transform=None):

        self.image_names = []
        self.image_labels = []
        self.transform = transform

        with open(image_file, 'r') as file:
            for line in file:
                image_name = line.split()[0]
                image_label = line.split()[1:]

                image_path = path + '\\' + image_name
                self.image_names.append(image_path)
                self.image_labels.append(image_label)

    def __getitem__(self, index):

        image = Image.open(self.image_names[index]).convert('RGB')
        # image = np.array(image)

        label = self.image_labels[index]
        label = [int(i) for i in label]

        if self.transform:
            image = self.transform(image)

        label = torch.FloatTensor(label)

        return image, label

    def __len__(self):

        return len(self.image_names)


"""
since = time.time()

test_set = ChestXRay('test_list.txt', transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                    transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor()]))
   
train_set = ChestXRay('train_list.txt', transform=transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                                                    transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor()]))
to = time.time()

print("Transformation of Images done!")

print("Total time: {} seconds".format(to-since))
print("Total Train Images: {}".format(train_set.__len__()))
#print("Total Test Images: {}".format(test_set.__len__()))

print("Total Train Batches: {}".format(math.ceil(train_set.__len__()/32)))

start = time.time()

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
  

imgs, labels = next(iter(train_loader))
print("size of imgs: {}".format(imgs.size()))  
t = time.time()-start

batches = math.ceil(train_set.__len__()/32)
print("Load time for 1 batch: {} seconds".format(time.time()-start))
print("Load time for {} batches: {}".format(batches, t*batches))

print("Batch Shape: {}".format(imgs.shape))
print("Batch Labels: {}".format(labels.shape))
"""
