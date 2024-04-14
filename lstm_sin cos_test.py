# -*- coding:UTF-8 -*-
#导入所需的 Python 库
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


#定义LSTM
class LstmRNN(nn.Module):#继承自nn.Module的类LstmRNN
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):#初始化参数
        super().__init__()#继承父类nn.Model中的初始化函数和方法
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) #调用torch.nn里的litm模型
        self.forwardCalculation = nn.Linear(hidden_size, output_size)#建立LSTM层到输出层的全连接层,将LSTM层的输出映射到输出向量,该向量将被用于计算模型损失并更新模型权重

    def forward(self, input):#前向传播方法
        output, _ = self.lstm(input)  # 输入input, size (seq_len, batch, input_size) 输出output
        seq_len, batch_size, hidden_size = output.shape  # 输出output的形状, size (seq_len, batch, hidden_size)  ‘_’ 表示输出状态h和记忆单元状态c
        output = output.view(seq_len * batch_size, hidden_size)#改变output形状
        output = self.forwardCalculation(output)#output传入线性变换层，对输入张量进行线性变换和加偏置值运算，得到未经激活函数处理的线性变换结果
        output = output.view(seq_len, batch_size, -1)#output变回三维张量
        return output


if __name__ == '__main__':
    #建立数据
    data_len = 200
    t = np.linspace(0, 12 * np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    #建立数据集，放入数据
    dataset = np.zeros((data_len, 2))#创建一个长度为200，有两个特征的数据集，即2列200行的零矩阵
    dataset[:, 0] = sin_t
    dataset[:, 1] = cos_t
    dataset = dataset.astype('float32')#将数据集转换为float32类型，并存储在dataset中

    #画出原始数据的sint和cost
    plt.figure()
    plt.plot(t[0:60], dataset[0:60, 0], label='sin(t)')
    plt.plot(t[0:60], dataset[0:60, 1], label='cos(t)')
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = t1')
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = t2')
    plt.text(2.5, -1.3, "t1", size=10, alpha=1.0)
    plt.text(6.8, -1.3, "t2", size=10, alpha=1.0)
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')#添加图例



    #划分训练集和测试集
    train_data_ratio = 0.5  #数据集一半训练集一半测试集合
    train_data_len = int(data_len * train_data_ratio)#训练集数据长度

    train_x = dataset[:train_data_len, 0]#100行2列数据
    train_y = dataset[:train_data_len, 1]
    t_for_training = t[:train_data_len]#在时间序列数据中，将样本的时间戳截取到train_data_len个

    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

# ------------------------------------------训练--------------------------------------------------------------------------------------
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM)#改变形状，其中-1表示自动计算剩余的维度大小，这里将batch size设置为5，将所有的特征reshape到第二个维度
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM)

    train_x_tensor = torch.from_numpy(train_x_tensor)#将numpy数组转换为PyTorch张量
    train_y_tensor = torch.from_numpy(train_y_tensor)

    #初始化一个LSTM模型
    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 32, output_size=OUTPUT_FEATURES_NUM, num_layers=2)

    print('LSTM model:', lstm_model)#打印LSTM模型的信息，包括输入大小，隐藏层大小，输出大小及层数
    print('model.parameters:', lstm_model.parameters)#打印LSTM模型的参数。该输出将返回模型的可学习参数，包括权重和偏置项

    loss_function = nn.MSELoss()#定义损失函数，这里选择均方误差（Mean Squared Error）作为损失函数
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)#定义优化器，这里选择Adam优化器，学习率为1e-2，优化目标是LSTM模型的可学习参数

    max_epochs = 10000#设定最大迭代次数为10000，即训练模型最多迭代10000次
    epoch_list=[]
    loss_list=[]
    for epoch in range(max_epochs):
        #将训练集输入模型，得到模型对训练集的预测输出。这里lstm_model是我们定义的LSTM模型，train_x_tensor是我们将训练集转换为PyTorch张量后的输入，是模型的输入
        output = lstm_model(train_x_tensor)

        loss = loss_function(output, train_y_tensor)#损失函数
        loss.backward()#损失值对模型参数的梯度进行反向传播计算
        optimizer.step()#对模型的参数进行更新
        optimizer.zero_grad()#清除模型的梯度缓存

        if loss.item() < 1e-4:#如果损失函数的值小于1e-4，说明当前模型已经收敛，可以结束训练
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:#如果当前迭代次数是100的倍数，则打印训练结果
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        epoch_list.append(epoch)
        loss_list.append(loss.item())

    plt.figure()
    plt.plot(epoch_list, loss_list,'b',label='Loss')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    #plt.show()

    #训练集上的预测cost
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()#变换形状并转换为numpy数组

# --------------------------------------- 测试 ---------------------------------------------------------------------------------
    lstm_model = lstm_model.eval()#将模型转换为测试模式

    #测试集上预测cost
    test_x_tensor = test_x.reshape(-1, 5,INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)

    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

# ---------------------------------------- 绘图 --------------------------------------------------------------------------------------
    plt.figure()
   #plt.plot(t_for_training, train_x, 'g', label='sin_train')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_train')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_train')
    #plt.plot(t_for_testing, test_x, 'c', label='sin_test')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_test')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_test')
    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line')  # separation line
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size=15, alpha=1.0)
    plt.text(20, 2, "test", size=15, alpha=1.0)
    plt.show()