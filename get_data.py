# Author Jack Shen
# _*_ coding:utf-8 _*_
import torch
import torch.utils.data
from libtiff import TIFF
import scipy.io as scio
import numpy as np

data_path_lab = './label.mat'
data_path_ms = './ms4.tif'
data_path_pan = './pan.tif'
#读取数据和标签
labels = scio.loadmat(data_path_lab)

ms = TIFF.open(data_path_ms,mode='r')
pan = TIFF.open(data_path_pan,mode='r')

image_ms = ms.read_image()
image_pan = pan.read_image()
image_ms = np.array(image_ms).transpose((2,0,1))#调整通道

#data preprosessing
MS_CUT_SIZE = 16
PAN_CUT_SIZE = 64
TRAIN_RATE = 0.3
np.random.seed(1)
torch.manual_seed(1)
'''# a = np.array(image_ms[:,784:800,814:830])
# print(a)
# print(a.shape)
# new=np.zeros(a.shape)
# h=a.shape[1]
# l=a.shape[2]
# for i in range(h):
#       new[:,h-1-i,:]=a[:,i,:]
#
# c = np.concatenate((a,new),axis=1)
# print(c,c.shape)
# print(new)'''

def mirror(image,cut_size):
    '''镜像函数，根据裁剪尺寸对数据做镜像处理'''
    image = np.array(image)
    type = image.dtype
    shape = image.shape
    if len(shape)>2:
        #垂直镜像/行镜像
        add_h = np.array(image[:,image.shape[1]-cut_size+1:,:])
        mir_h = np.zeros(add_h.shape,dtype=type)#与原始数据类型保持一致
        h = add_h.shape[1]
        for i in range(h):
            mir_h[:,h-1-i,:]=add_h[:,i,:]
        a1 = np.concatenate((image,mir_h),axis=1)#a1 shape(:,h+cut_size-1,l)
        #水平镜像/列镜像
        add_l = np.array(a1[:,:,image.shape[2]-cut_size+1:])
        mir_l = np.zeros(add_l.shape,dtype=type)
        l = add_l.shape[2]
        for j in range(l):
            mir_l[:,:,l-1-j]=add_l[:,:,j]

        mir = np.concatenate((a1,mir_l),axis=2)#a2 shape(:,h+cut_size-1,l+cut_size-1)
    else:
        # 垂直镜像/行镜像
        add_h = np.array(image[image.shape[0] - cut_size + 1:, :])
        mir_h = np.zeros(add_h.shape, dtype=type)  # 与原始数据类型保持一致
        h = add_h.shape[0]
        for i in range(h):
            mir_h[ h - 1 - i, :] = add_h[ i, :]
        a1 = np.concatenate((image, mir_h), axis=0)  # a1 shape(h+cut_size-1,l)
        # 垂直镜像/行镜像
        add_l = np.array(a1[:, image.shape[1] - cut_size + 1:])
        mir_l = np.zeros(add_l.shape, dtype=type)
        l = add_l.shape[1]
        for j in range(l):
            mir_l[:, l - 1 - j] = add_l[:, j]

        mir = np.concatenate((a1, mir_l), axis=1)  # a2 shape(h+cut_size-1,l+cut_size-1)

    return mir

# m_ms = mirror(image_ms,MS_CUT_SIZE)#多返回值函数在只返回一个返回元素时是元组
# m_pan = mirror(image_pan,PAN_CUT_SIZE)
# print(m_pan.dtype,m_pan.shape)
# print(m_ms.dtype,m_ms.shape)
'''
# print(type(a))
# print(a.shape)
# print(a.dtype)
# # print(l,l.shape)
# # print(lm,lm.shape)'''

# print(type(labels))
# print(labels.keys())
# print(labels.values())
#
# for key,value in labels.items():
#     print(key,':',value)
#
# print(labels['label'])
# label = labels['label']#获取标签
# label = np.array(label)#转换成numpy数组
# total_label = max(label)


#制作训练数据和测试数据位置和标签

# for i in range(1,total_label+1):
#     index_I = np.where(label==i) #第i类坐标
#     index_I = np.array(index_I).T
#     len_I = len(index_I)  # 索引总长度
#     len_train = int(len_I * TRAIN_RATE)  # 第i类训练样本数
#     len_test = int(len_I-len_train)     # 第i类测试样本数
#     index_train = np.arange(len_I)#建立第i类所有索引
#     np.random.shuffle(index_train)  # 打乱索引顺序
#     label_train_i = i * np.ones((len_train,))  # 第i类训练样本label
#     label_test_i = i * np.ones((len_test,)) # 第i类测试样本label
#     if i == 1:
#         train_data_label = label_train_i
#         train_data_loca = index_I[index_train[:len_train]]
#         test_data_label = label_test_i
#         test_data_loca = index_I[index_train[len_train:]]
#     else:
#         train_data_label=np.append(train_data_label,label_train_i,axis=0)
#         train_data_loca=np.append(train_data_loca,index_I[index_train[:len_train]],axis=0)#第i类训练样本坐标
#         test_data_label=np.append(test_data_label,label_test_i,axis=0)
#         test_data_loca=np.append(test_data_loca,index_I[index_train[len_train:]],axis=0)
    # label_l[counter:len_I+counter] = i
    # data_loca[counter:len_I+counter,:] = index_I[:,:]


#print(train_data_label,train_data_loca)
#print(train_data_label.shape,train_data_loca.shape)
# print(test_data_label,test_data_loca)
#print(test_data_label.shape,test_data_loca.shape)


'''制作数据集'''
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image-min_i)/(max_i-min_i)
    return image

'''双支路原始dataset'''
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mimage1, mimage2,local,cut_size,
                 transform=None,target_transform=None):
        super(MyDataset,self).__init__()
        self.cut_size_ms = cut_size
        self.cut_size_pan = int(4 * cut_size)
        self.local = local
        self.image1 = mimage1
        self.image2 = mimage2

    def __getitem__(self, index):
        x_ms, y_ms, label= self.local[index]
        x_pan = int(4 * x_ms)#计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        self.image_ms = self.image1[:, x_ms:x_ms+self.cut_size_ms,
                        y_ms:y_ms+self.cut_size_ms]
        # print(self.image_ms)
        self.image_pan = self.image2[:, x_pan:x_pan+self.cut_size_pan,
                          y_pan:y_pan+self.cut_size_pan]
        # print(self.image_pan.shape)
        # print(self.image_pan)

        self.label = label

        return self.image_ms, self.image_pan, self.label,x_ms,y_ms

    def __len__(self):
        return len(self.local)
'''单支路网络dataset'''
class MyDataset_signel(torch.utils.data.Dataset):
    def __init__(self, mimage,local,cut_size,
                 transform=None,target_transform=None):
        super(MyDataset_signel,self).__init__()
        self.cut_size_ms = cut_size
        self.local = local
        self.image = mimage

    def __getitem__(self, index):
        x_ms, y_ms, label = self.local[index]
        if self.cut_size_ms>33:
            x_ms = int(4 * x_ms)
            y_ms = int(4 * y_ms)
            self.image_ms = self.image[:, x_ms:x_ms+self.cut_size_ms,
                        y_ms:y_ms+self.cut_size_ms]
        else:
            self.image_ms = self.image[:, x_ms:x_ms+self.cut_size_ms,
                        y_ms:y_ms+self.cut_size_ms]
        # print(self.image_ms)
        # print(self.image_pan.shape)
        # print(self.image_pan)

        self.label = label

        return self.image_ms, self.label

    def __len__(self):
        return len(self.local)

'''双支路改进dataset'''
class MyDataset_double1(torch.utils.data.Dataset):
    def __init__(self, mimage1, mimage2,local,cut_size,
                 transform=None,target_transform=None):
        super(MyDataset_double1,self).__init__()
        self.cut_size = cut_size

        '''统一cutsize'''

        self.local = local
        self.image1 = mimage1
        self.image2 = mimage2

    def __getitem__(self, index):
        x_ms, y_ms, label = self.local[index]
        x_pan = int(4 * x_ms)#计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        self.image_ms = self.image1[:, x_pan:x_pan+self.cut_size,
                        y_pan:y_pan+self.cut_size]
        # print(self.image_ms)
        self.image_pan = self.image2[:, x_pan:x_pan+self.cut_size,
                          y_pan:y_pan+self.cut_size]
        # print(self.image_pan.shape)
        # print(self.image_pan)

        self.label = label

        return self.image_ms, self.image_pan, self.label

    def __len__(self):
        return len(self.local)



class MyDataset2(torch.utils.data.Dataset):
    def __init__(self, mimage1, mimage2,local,cut_size,
                 transform=None,target_transform=None):
        super(MyDataset2,self).__init__()
        self.cut_size_ms = cut_size
        self.cut_size_pan = int(4 * cut_size)
        self.local = local
        self.image1 = mimage1
        self.image2 = mimage2

    def __getitem__(self, index):
        x_ms, y_ms = self.local[index]
        x_pan = int(4 * x_ms)#计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        self.image_ms = self.image1[:, x_ms:x_ms+self.cut_size_ms,
                        y_ms:y_ms+self.cut_size_ms]
        #print(self.image_ms)
        self.image_pan = self.image2[:, x_pan:x_pan+self.cut_size_pan,
                          y_pan:y_pan+self.cut_size_pan]
        #print(self.image_pan.shape)
        #print(self.image_pan)



        return self.image_ms, self.image_pan

    def __len__(self):
        return len(self.local)