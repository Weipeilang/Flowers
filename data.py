#encoding = utf-8
#读取数据集

import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datatxt,transform=None):
        super(Dataset, self).__init__()
        images = []
        with open(datatxt,'r') as f:
            for line in f:
                image_and_label = line.rstrip().split()  #读取图片路径和标签，存在一起,0表示路径，1表示标签
                images.append(image_and_label)
        self.images = images
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(
            mean=(0.46645354523231014, 0.42501653146608886, 0.3039079953194502),
            std=(0.2988135400695513, 0.26933252694782384, 0.2928475491205623))])

    def __getitem__(self, index):
        image_and_label = self.images[index]
        #print(image_and_label[0])
        # image = cv2.imread(image_and_label[0])
        # if image.ndim==2:
        #     image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image,(224,224))  #有些图片无法读取
        image = Image.open(image_and_label[0],'r')
        image = image.resize((224,224))
        if transforms:
            image = self.transforms(image)
        label = torch.from_numpy(np.asarray(image_and_label[1], dtype=np.float32)).long()
        return image.reshape(3,224,224), label

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.images)

if __name__ == '__main__':
    train_data = Dataset(datatxt='train.txt',transform=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True, drop_last=True)
    for data, label in train_loader:
        print(data.shape)
        print(label)
