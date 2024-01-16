import torch  # 需要的各种包
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision  # 数据库模块

# 数据预处理
# 将training data转化成torch能够使用的DataLoader，这样可以方便使用batch进行训练
torch.manual_seed(1)  # reproducible 将随机数生成器的种子设置为固定值，这样，当调用时torch.rand(x)，结果将可重现

# Hyper Parameters
EPOCH = 1  # 训练迭代次数
BATCH_SIZE = 50  # 分块送入训练器
LR = 0.001  # 学习率 learning rate

train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存位置 若没有就新建
    train=True,  # training set
    transform=torchvision.transforms.ToTensor(),  #
    # converts a PIL.Image or numpy.ndarray to torch.FloatTensor(C*H*W) in range(0.0,1.0)
    download=True
)

test_data = torchvision.datasets.MNIST(root='./MNIST/')

# 如果是普通的Tensor数据，想使用 torch_dataset = data.TensorDataset(data_tensor=x, target_tensor=y)
# 将Tensor转换成torch能识别的dataset
# 批训练， 50 samples, 1 channel, 28*28, (50, 1, 28 ,28)
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255.
# torch.unsqueeze 返回一个新的张量，对输入的既定位置插入维度 1

test_y = test_data.targets[:2000]


# 数据预处理


# 定义网络结构
# 1）class CNN需要·继承·Module
# 2）需要·调用·父类的构造方法：super(CNN, self).__init__()
# 3）在Pytorch中激活函数Relu也算是一层layer
# 4）需要·实现·forward()方法，用于网络的前向传播，而反向传播只需要·调用·Variable.backward()即可。
# 输入的四维张量[N, C, H, W]
class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(CNN, self).__init__()
        # nn.Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数
        # nn.Conv2d 二维卷积 先实例化再使用 在Pytorch的nn模块中，它是不需要你手动定义网络层的权重和偏置的
        self.conv1 = nn.Sequential(  # input shape (1,28,28)
            nn.Conv1d(in_channels=self.in_channels,  # input height 必须手动提供 输入张量的channels数
                      out_channels=self.out_channels,  # n_filter 必须手动提供 输出张量的channels数
                      kernel_size=5,  # filter size 必须手动提供 卷积核的大小
                      # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
                      stride=1,  # filter step 卷积核在图像窗口上每次平移的间隔，即所谓的步长
                      padding=2,
                      ),
            # nn.Conv2d(in_channels=self.in_channels,  # input height 必须手动提供 输入张量的channels数
            #           out_channels=self.out_channels,  # n_filter 必须手动提供 输出张量的channels数
            #           kernel_size=5,  # filter size 必须手动提供 卷积核的大小
            #           # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
            #           stride=1,  # filter step 卷积核在图像窗口上每次平移的间隔，即所谓的步长
            #           padding=2  # con2d出来的图片大小不变 Pytorch与Tensorflow在卷积层实现上最大的差别就在于padding上
            #           ),  # output shape (16,28,28) 输出图像尺寸计算公式是唯一的 # O = （I - K + 2P）/ S +1
            nn.ReLU(),  # 分段线性函数，把所有的负值都变为0，而正值不变，即单侧抑制
            nn.MaxPool2d(kernel_size=2)
            # 2x2采样，28/2=14，output shape (16,14,14) maxpooling有局部不变性而且可以提取显著特征的同时降低模型的参数，从而降低模型的过拟合
        )
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 5, 1, 2),  # output shape (32,7,7)
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        # 因上述几层网络处理后的output为[32,7,7]的tensor，展开即为7*7*32的一维向量，接上一层全连接层，最终output_size应为10，即识别出来的数字总类别数
        # 在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层 7*7*32, num_classes

    def forward(self, x):
        x = self.conv1(x)  # 卷一次
        x = self.conv2(x)  # 卷两次
        x = x.view(x.size(0), -1)  # flat (batch_size, 32*7*7)
        # 将前面多维度的tensor展平成一维 x.size(0)指batchsize的值
        # view()函数的功能根reshape类似，用来转换size大小
        output = self.out(x)  # fc out全连接层 分类器
        return output


# 定义网络结构


# 查看网络结构
cnn = CNN()
print(cnn)  # 使用print(cnn)可以看到网络的结构详细信息，可以看到ReLU()也是一层layer
# 查看网络结构


# 训练 需要特别指出的是记得每次反向传播前都要清空上一次的梯度，optimizer.zero_grad()
# optimizer 可以指定程序优化特定的选项，例如学习速率，权重衰减等
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # torch.optim是一个实现了多种优化算法的包

# loss_fun CrossEntropyLoss 交叉熵损失
# 信息量：它是用来衡量一个事件的不确定性的；一个事件发生的概率越大，不确定性越小，则它所携带的信息量就越小。
# 熵：它是用来衡量一个系统的混乱程度的，代表一个系统中信息量的总和；信息量总和越大，表明这个系统不确定性就越大。
# 交叉熵：它主要刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近
loss_func = nn.CrossEntropyLoss()  # 该损失函数结合了nn.LogSoftmax()和nn.NLLLoss()两个函数 适用于分类

# training loop
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        output = cnn(batch_x)  # 输入训练数据
        loss = loss_func(output, batch_y)  # 计算误差 #　实际输出，　期望输出
        optimizer.zero_grad()  # 清空上一次梯度
        loss.backward()  # 误差反向传递 只需要调用.backward()即可
        optimizer.step()  # cnn的优化器参数更新
# 训练


# 预测结果
# cnn.eval()
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# torch.max(input, dim)函数
# torch.max(test_output, 1)[1]  取出来indices 每行最大值的索引
# 输入 input是softmax函数输出的一个tensor
# 输入 dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
# 输出 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
# squeeze()函数的功能是：从矩阵shape中，去掉维度为1的。例如一个矩阵是的shape是（5， 1），使用过这个函数后，结果为（5，）。
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
# 预测结果