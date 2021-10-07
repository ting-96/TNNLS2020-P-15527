#_*_coding:utf8_*_from __future__ import absolute_import, print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ADRSS
import get_data
import CPM
import torch.utils.data
from torch.utils.data import DataLoader
from libtiff import TIFF
import scipy.io as scio
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import time

data_path_lab = './label.mat'
data_path_ms = './ms4.tif'
data_path_pan = './pan.tif'
#读取数据和标签
labels = scio.loadmat(data_path_lab)
ms = TIFF.open(data_path_ms, mode='r')
pan = TIFF.open(data_path_pan, mode='r')
image_ms = ms.read_image()
image_pan = pan.read_image()
image_ms = np.array(image_ms).transpose((2,0,1))#调整通道
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
#data preprosessing
MS_CUT_SIZE = 16
PAN_CUT_SIZE = 64
TRAIN_RATE = 0.01
EPOCH = 1
BATCH_SIZE = 32
LR = 0.0001
alpha1=0.3 #融合比例系数1
alpha2=0.57 #融合比例系数2

m_ms = get_data.mirror(image_ms,MS_CUT_SIZE)#多返回值函数在只返回一个返回元素时是元组
print(m_ms.shape)
m_pan = get_data.mirror(image_pan,PAN_CUT_SIZE)
#print("data")
label = labels['label']#获取标签
label = np.array(label)#转换成numpy数组
#print(label.shape)
total_label = np.max(label)#最大类别数
#print(total_label)
ADRSS(image_ms) #生成自适应扩张率
kernel=np.loadtxt('side_length.txt',dtype=np.int64,delimiter=',')#自适应卷积核扩张率
#print(kernel.shape)
#制作训练数据和测试数据位置和标签
index=CPM(image_ms) #中心像素迁移策略获取对应的偏移位置
for i in range(1, total_label+1):
    index_I = np.where(label==i) #第i类坐标
    index_I = np.array(index_I).T
    #print(index_I.shape)
    #print(index_I)
    len_I = len(index_I)  # 索引总长度
    len_train = int(len_I * TRAIN_RATE)  # 第i类训练样本数
    len_valid = int(len_I-len_train)     # 第i类测试样本数
    index_train = np.arange(len_I) #建立第i类所有索引
    np.random.shuffle(index_train)  # 打乱索引顺序
    label_train_i = i * np.ones((len_train, 1), dtype='int64')  # 第i类训练样本label
    label_valid_i = i * np.ones((len_valid, 1), dtype='int64') # 第i类测试样本label
    if i == 1:
        train_data_label = label_train_i
        train_data_loca = index_I[index_train[:len_train]]
        valid_data_label = label_valid_i
        valid_data_loca = index_I[index_train[len_train:]]
    else:
        train_data_label=np.append(train_data_label, label_train_i, axis=0)
        train_data_loca=np.append(train_data_loca, index_I[index_train[:len_train]], axis=0)#第i类训练样本坐标
        valid_data_label=np.append(valid_data_label, label_valid_i, axis=0)
        valid_data_loca=np.append(valid_data_loca, index_I[index_train[len_train:]], axis=0)
    # label_l[counter:len_I+counter] = i

# print(train_data_label.dtype, train_data_loca.dtype)
train_data_label = train_data_label - 1#label要从0开始
valid_data_label = valid_data_label - 1
train_data_loca = np.hstack((train_data_loca, train_data_label))
valid_data_loca = np.hstack((valid_data_loca, valid_data_label))
all_label_data = np.vstack((train_data_loca, valid_data_loca))
# print(train_data_loca.shape, valid_data_loca.shape)
np.random.shuffle(train_data_loca)
np.random.shuffle(valid_data_loca)
np.random.shuffle(all_label_data)
#数据归一化
m_ms = get_data.to_tensor(m_ms)
m_pan = get_data.to_tensor(m_pan)
m_pan = np.expand_dims(m_pan, axis=0)# 二维数据进网络前要加一维
# print(m_ms.dtype, m_pan.dtype)
train_data = get_data.MyDataset(m_ms, m_pan, train_data_loca, cut_size=MS_CUT_SIZE)
# print(train_data[0][0])
# print(train_data[0][1])
# print(train_data[0][2])
# print(train_data[0][3])
train_loder = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
valid_data = get_data.MyDataset(m_ms, m_pan, valid_data_loca, cut_size=MS_CUT_SIZE)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

test_data = get_data.MyDataset(m_ms, m_pan, all_label_data, cut_size=MS_CUT_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#print(all_label_data.shape)
true_label = all_label_data[:, 2]
pred_label = np.zeros(len(true_label), dtype=int)

atrous_rates=[6, 12, 18, 24]
n_classes=7
n_blocks=[3, 3, 3, 3]

class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

class spatial_attention(nn.Sequential):
        def __init__(self, kernel_size=7):
            super(spatial_attention, self).__init__()

            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = 3 if kernel_size == 7 else 1

            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)

class channel_attention(nn.Sequential):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DeepLabV3_pan(nn.Sequential):

    def __init__(self, n_blocks, atrous_rates):
        super(DeepLabV3_pan, self).__init__()

        # Stride and dilation

        s = [1, 2, 2, 1]
        d = [1, 1, 2, 4]
        s1 = [1, 1, 1, 1]
        d1 = [1, 1, 1, 4]

        ch = [64 * 2 ** p for p in range(6)]
        self.layer1_pan=_Stem(ch[0])
        self.layer2_pan=_ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.fuse1_pan=spatial_attention()
        self.layer3_pan =_ResLayer(n_blocks[1], ch[2]*2, ch[3], s[1], d[1])
        self.layer4_pan=_ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
        self.fuse2_pan=spatial_attention()
        self.conv1=nn.Conv2d(ch[2], ch[2], 3, 3, 0, 1, bias=False)
        self.conv2 = nn.Conv2d(ch[2], ch[2], 3, 1, 7, 1, bias=False)

        self.layer1_ms= _Stem1(ch[0])
        self.layer2_ms=_ResLayer(n_blocks[0], ch[0], ch[2], s1[0], d1[0])
        self.fuse1_ms=channel_attention(ch[2])
        self.layer3_ms=_ResLayer(n_blocks[1], ch[2]*2, ch[3], s1[1], d1[1])
        self.layer4_ms=_ResLayer(n_blocks[2], ch[3], ch[4], s1[2], d1[2])
        self.fuse2_ms=channel_attention(ch[4])

        self.aspp= _ASPP(ch[4]*2, 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.fc1=_ConvBnReLU(concat_ch, 256, 1, 1, 0, 1)

    def forward(self, pan, ms):
        pan = self.layer1_pan(pan)
        pan = self.layer2_pan(pan)
        x1=self.fuse1_pan(pan)

        ms=self.layer1_ms(ms)
        ms=self.layer2_ms(ms)
        x2=self.fuse1_ms(ms)

        ms1=pan*x2*alpha1
        ms1 = self.conv1(ms1)
        ms1 = torch.cat((ms1, ms), 1)
        ms2=self.conv2(ms)
        pan1=ms2*x1*(1-alpha1)
        pan1 = torch.cat((pan1,pan),1)


        pan1=self.layer3_pan(pan1)
        ms1=self.layer3_ms(ms1)
        pan1 = self.layer4_pan(pan1)
        ms1 = self.layer4_ms(ms1)

        x1 = self.fuse2_pan(pan1)
        x2 = self.fuse2_ms(ms1)

        ms2=pan1*x2*alpha2
        ms=torch.cat((ms2,ms1),1)

        pan2=ms1*x1*(1-alpha2)
        pan=torch.cat((pan2,pan1),1)

        pan1 = self.aspp(pan)
        pan = self.fc1(pan1)
        ms1=self.aspp(ms)
        ms=self.fc1(ms1)

        return pan, ms

class SEnet(nn.Sequential):
    def __init__(self,filters):
        super(SEnet,self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(filters, filters // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filters // 16, filters, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, input):
        x1=self.se(input)
        output=input*x1
        return output


class Net(nn.Sequential):
        def __init__(self):
            super(Net, self).__init__()
            # 全连接层
            self.fuse = DeepLabV3_pan(
                n_blocks,
                atrous_rates
            )
            self.fuse3_pan = spatial_attention()
            self.fuse3_ms = channel_attention(512)
            self.ms=SEnet(768)
            self.fc = nn.Sequential(
                nn.Linear(768 * 5 * 5 * 2, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 11))

        def forward(self, x1, x2):
            pan,ms=self.fuse(x1,x2)
            fuse = torch.cat((pan, ms), 1)

            pan1 = self.fuse3_pan(fuse)
            ms1 = self.fuse3_ms(fuse)

            pan2 = fuse * pan1
            ms2 = fuse * ms1

            pan = torch.cat((pan, pan2), 1)
            ms = torch.cat((ms, ms2), 1)
            ms =self.ms(ms)

            x = torch.cat((pan, ms), 1)
            x = x.view(x.size(0), 768 * 5 * 5 * 2)

            output = self.fc(x)

            return output


try:
    from torch.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _Stem(nn.Sequential):
    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(1, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

class _Stem1(nn.Sequential):
    def __init__(self, out_ch):
        super(_Stem1, self).__init__()
        self.add_module("conv1", _ConvBnReLU(4, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

model = Net()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.00005)
loss_fuc = nn.CrossEntropyLoss()
torch.distributed.init_process_group('nccl',init_method='file:///home/.../my_file',world_size=1,rank=0)
model.to(device)   # 记得用cuda()
kernel1=np.zeros(32)

for epoch in range(EPOCH):
    valid_batch = iter(valid_loader)  # 验证集迭代器
    for step, (x1, x2, y, x_ms, y_ms) in enumerate(train_loder):
        model.train()
        batch_x1 = torch.tensor(x1, dtype=torch.float32).to(device)
        batch_x2 = torch.tensor(x2, dtype=torch.float32).to(device)
        y = y.to(device)
        for i in range(len(x_ms)):
            if kernel[x_ms[i]][y_ms[i]]==0:
                kernel1[i] = 18
            else:
                kernel1[i]=kernel[x_ms[i]][y_ms[i]]
        atrous_rates[3]=kernel1[step%32]
        output = model(batch_x2,batch_x1)
        # print(output.size())
        # print(y.size())
        loss = loss_fuc(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100 == 0:
            ls = loss.data.cpu().numpy()
            model.eval()
            #print(ls)
            x1, x2, y, x_ms, y_ms= next(valid_batch)
            y1 = y.data.numpy()
            # print(y1.dtype)
            # print(y1)
            batch_v1 = torch.tensor(x1, dtype=torch.float32).to(device)
            batch_v2 = torch.tensor(x2, dtype=torch.float32).to(device)
            batch_vy = torch.tensor(y)
            valid_output = model(batch_v2,batch_v1)
            valid_outputs = valid_output.data.cpu().numpy()
            #print(valid_outputs)
            valid_result = np.argmax(valid_outputs,axis=1)
            # print(valid_result.dtype)
            # print(valid_result)
            v_accuracy = 0
            for i,x in enumerate(valid_result):
                 temp = np.where(y1[i]==x, 1, 0)
                 v_accuracy = v_accuracy + temp

            v_accuracy = (v_accuracy/len(valid_result))
            print('Epoch: ', epoch,
                  '|| batch: ', step,
                  '|| train loss: %.4f' % loss.data.cpu().numpy(),
                  '|| valid accuracy: %.2f' % v_accuracy)

out_color = np.zeros((2001, 2101, 3))
out_result_metricx = np.zeros((2001, 2101))
loc = 0
teststart =time.time()
model.eval()
for j, (ms, pan, label,ms_x,ms_y) in enumerate(test_loader):
    batch_t1 = torch.tensor(ms, dtype=torch.float32).to(device)
    batch_t2 = torch.tensor(pan, dtype=torch.float32).to(device)
    test_label = torch.tensor(label).to(device)
    for w in range(len(ms_x)):
        if kernel[ms_x[w]][ms_y[w]] == 0:
            kernel1[w] = 18
        else:
            kernel1[w] = kernel[ms_x[w]][ms_x[w]]
    atrous_rates[3] = kernel1[j % 32]
    test_output = model(batch_t2,batch_t1)
    test_out_data = test_output.data.cpu().numpy()
    test_result = np.argmax(test_out_data, axis=1)
    b, h, w, c = batch_t1.shape
    testend = time.time()
    for k, cls in enumerate(test_result):
        x1, y1, lab1 = all_label_data[k + b * j]
        pred_label[loc] = cls
        loc = loc + 1
        if cls == 0:
            out_color[x1][y1] = [255, 255, 0]
            out_result_metricx[x1][y1] = 0
        elif cls == 1:
            out_color[x1][y1] = [255, 0, 0]
            out_result_metricx[x1][y1] = 1
        elif cls == 2:
            out_color[x1][y1] = [33, 145, 237]
            out_result_metricx[x1][y1] = 2
        elif cls == 3:
            out_color[x1][y1] = [0, 255, 0]
            out_result_metricx[x1][y1] = 3
        elif cls == 4:
            out_color[x1][y1] = [240, 32, 160]
            out_result_metricx[x1][y1] = 4
        elif cls == 5:
            out_color[x1][y1] = [221, 160, 221]
            out_result_metricx[x1][y1] = 5
        elif cls == 6:
            out_color[x1][y1] = [140, 230, 240]
            out_result_metricx[x1][y1] = 6
        elif cls == 7:
            out_color[x1][y1] = [0, 0, 255]
            out_result_metricx[x1][y1] = 7
        elif cls == 8:
            out_color[x1][y1] = [0, 255, 255]
            out_result_metricx[x1][y1] = 8
        elif cls == 9:
            out_color[x1][y1] = [127, 255, 0]
            out_result_metricx[x1][y1] = 9
        elif cls == 10:
            out_color[x1][y1] = [255, 0, 255]
            out_result_metricx[x1][y1] = 10

cv2.imwrite('./output_fusion'+'.png', out_color)
cm = confusion_matrix(true_label, pred_label,labels=[0, 1, 2, 3, 4, 5, 6, 7 , 8, 9, 10])
print(['0', '1', '2', '3', '4', '5', '6','7,','8','9','10'])
print(cm)

dsa=0
pe = 0
ses = np.sum(cm,axis=1)
sep = np.sum(cm,axis=0)
print(ses,sep)
for i, cla in enumerate(cm):
    pred = cla[i]
    dsa+=cla[i]
    pe+=sep[i]*ses[i]
    acr = pred / sep[i]#准确率
    rep = pred / ses[i]
    f1 = 2*acr*rep / (acr + rep)#F1
total_e=np.sum(cm)
p=dsa/total_e
pe=pe/(total_e*total_e)
kappa = (p-pe)/(1-pe)#KAPPA
print(testend-teststart)#时间

