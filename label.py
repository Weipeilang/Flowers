#encoding=utf-8
#华为云花卉识别，数据集下载地址：https://modelarts-cnnorth1-market-dataset.obs.cn-north-1.myhuaweicloud.com/dataset-market/Flowers-Data-Set/archiver/Flowers-Data-Set.zip
#数据集格式是一张图片对应一个txt文件，txt文件里面是标签

import os
from PIL import Image
import numpy as np

data_path = r"..\Flowers-Data-Set"
images_and_txt = [i for i in os.listdir(data_path)]
train = open("train.txt", 'w')
test = open("test.txt", 'w')
labels = ['daisy', 'dandelion', 'sunflowers', 'roses', 'tulips']
sum = 0
dataset = []
for image in images_and_txt:
    if image.endswith(".jpg"):
        image_path = os.path.join(data_path,image)
        label_path = os.path.join(data_path,image[:-4] + '.txt')
        with open(label_path) as f:
            label = f.readline()
        image_and_label = image_path + " " + str(labels.index(label)) + '\n'
        if sum % 10 < 7 : #划分训练集和测试集
            train.write(image_and_label)
        else:
            test.write(image_and_label)
        sum += 1

        #计算方差和均值
        image_data = Image.open(image_path,'r')
        image_data = image_data.resize((224,224))
        image_data = np.array(image_data).reshape(224,224,3)/255.
        dataset.append(image_data)
        print(sum," ",image)
train.close()
test.close()

dataset = np.array(dataset)
print(dataset.shape)
data_r = np.dstack([dataset[i][:, :, 0] for i in range(len(dataset))])
data_g = np.dstack([dataset[i][:, :, 1] for i in range(len(dataset))])
data_b = np.dstack([dataset[i][:, :, 2] for i in range(len(dataset))])
mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
std = np.std(data_r), np.std(data_g), np.std(data_b)
print(mean,std)
