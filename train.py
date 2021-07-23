import time
import torch
from data import Dataset
from torchvision.models import vgg16_bn,resnet18
from matplotlib import pyplot as plt
import torchvision
import numpy as np

Batch_size = 8
Learning_rate = 1e-4
Epoch = 20

def imshow(img):
    img = img / 2 + 0.5  # 将正则化后图片恢复
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    train_data = Dataset(datatxt='train.txt')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True,drop_last = True)

    test_data = Dataset(datatxt='test.txt')
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=Batch_size, shuffle=True, drop_last=True)

    # 准备数据集并预处理
    # transform_train = transforms.Compose([
    #     transforms.Resize((224,224)),       # 把图像变换成224大小
    #     transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train)  # 训练数据集
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=Batch_size, shuffle=True,num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
    #
    # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=Batch_size, shuffle=False, num_workers=2)

    # 随取抽取训练集中的图片
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # # 显示图片
    # imshow(torchvision.utils.make_grid(images[0]))

    model = resnet18(num_classes=5)

    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(1,Epoch+1):
        model.train()
        for index, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('{} epoch: {}  idx:{}  average loss: {}'.format(
                time.asctime(time.localtime(time.time())),epoch ,index+1, loss.sum().item()))

        model.eval()
        test_loss = 0
        correct = 0
        for (images,labels) in test_loader:
            outputs = model(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        print('Average loss {:.4f},accuracy: {}% '.format(test_loss/len(test_loader) , 100.0 * correct / len(test_data)))
        torch.save(model.state_dict(), 'model_{:.2f}.pkl'.format(int(100.0 * correct / len(test_data))))
