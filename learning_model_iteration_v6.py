# learing sum(FP+1)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import csv
import random


def data_progress():
    cvxcnn_data = list()
    cvxcnn_target = list()
    with open("iteration_data3.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            cvxcnn_data.append(list(map(float, line[:-3])))
            cvxcnn_target.append(list(map(float, line[-3:])))
            # print("cvxcnn_data:",cvxcnn_data)

    cvxcnn_data = np.array(cvxcnn_data)
    cvxcnn_target = np.array(cvxcnn_target)
    x_train, x_test, y_train, y_test = train_test_split(cvxcnn_data, cvxcnn_target, test_size=0.00001, random_state=42)
    print("x_train:", x_train.shape, "x_test:", x_test.shape, "y_train:", y_train.shape, "y_test:",
          y_test.shape)  # (14448, 8) (6192, 8) (14448,) (6192,)

    # scale = StandardScaler()
    # x_train = scale.fit_transform(x_train)
    # # x_test = scale.transform(x_test)
    train_xt = torch.from_numpy(x_train.astype(np.float32))
    train_yt = torch.from_numpy(y_train.astype(np.float32))
    test_xt = torch.from_numpy(x_test.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))

    print("train_xt:", train_xt)
    print("train_yt:", train_yt)

    train_data = Data.TensorDataset(train_xt, train_yt)

    test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)
    # print("train_loader:", train_loader)

    idx = random.randint(0, 299)
    # idx = 249，132，272，160
    idx = 1
    print("idx:", idx)
    # === training set ====
    single_data = np.array([cvxcnn_data[idx].tolist()])
    single_target = cvxcnn_target[idx]
    single_data = torch.from_numpy(single_data.astype(np.float32))

    # === testing set ====
    # single_data = torch.reshape(test_xt[idx], (1,len(test_xt[idx])))
    # single_target = np.array(test_yt[idx])

    return train_loader, train_xt, train_yt, test_xt, test_yt, y_test, single_data, single_target


# test_loader = Data.DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0)
class MLPPregression(nn.Module):
    def __init__(self):
        super(MLPPregression, self).__init__()
        self.in_num = 6
        self.hid_neu_num = 100
        self.hidden0_1 = nn.Linear(in_features=self.in_num, out_features=self.hid_neu_num, bias=True)
        self.hidden0_2 = nn.Linear(self.hid_neu_num, self.hid_neu_num)
        self.hidden0_3 = nn.Linear(self.hid_neu_num, self.hid_neu_num)
        self.predict0 = nn.Linear(self.hid_neu_num, 1)

        self.hidden1_1 = nn.Linear(in_features=self.in_num, out_features=self.hid_neu_num, bias=True)
        self.hidden1_2 = nn.Linear(self.hid_neu_num, self.hid_neu_num)
        self.hidden1_3 = nn.Linear(self.hid_neu_num, self.hid_neu_num)
        self.predict1 = nn.Linear(self.hid_neu_num, 1)

        self.hidden2_1 = nn.Linear(in_features=self.in_num, out_features=self.hid_neu_num, bias=True)
        self.hidden2_2 = nn.Linear(self.hid_neu_num, self.hid_neu_num)
        self.hidden2_3 = nn.Linear(self.hid_neu_num, self.hid_neu_num)
        self.predict2 = nn.Linear(self.hid_neu_num, 1)

    def forward(self, x):
        x0 = x
        # print("x0:",x0)
        G01 = x0[0][1]
        G02 = x0[0][2]
        G10 = x0[0][3]
        G12 = x0[0][5]
        G20 = x0[0][6]
        G21 = x0[0][7]
        sigma0 = x0[0][12]
        sigma1 = x0[0][13]
        sigma2 = x0[0][14]
        p_bar0 = x0[0][15]
        p_bar1 = x0[0][16]
        p_bar2 = x0[0][17]
        w = torch.reshape(x0[0][9:12], (3, 1))
        alpha = torch.reshape(x0[0][18:27], (3, 3))
        p0 = p_bar0
        p1 = p_bar1
        p2 = p_bar2
        p_bar = torch.reshape(x0[0][15:18], (3, 1))

        GP0 = torch.tensor([[G01 * p1, G02 * p2, 0.05]])
        GP1 = torch.tensor([[G10 * p0, G12 * p2, 0.05]])
        GP2 = torch.tensor([[G20 * p0, G21 * p1, 0.05]])

        d0 = w[1] * G10 / GP1 + w[2] * G20 / GP2
        res0 = torch.mm(w.T, torch.reshape(alpha[:, 0], (3, 1)))
        # p0 = res0 / d0
        feat0 = torch.cat((res0/(w[1] * G10)*GP0, res0/(w[2] * G20)*GP2),1)
        # print("feat0:",feat0)

        p0 = F.relu(self.hidden0_1(feat0))
        p0 = F.relu(self.hidden0_2(p0))
        p0 = F.relu(self.hidden0_3(p0))
        p0 = F.sigmoid(self.predict0(p0))*p_bar0
        # print("p0:",p0)

        d1 = w[0] * G01 / GP0 + w[2] * G21 / GP2
        res1 = torch.mm(w.T, torch.reshape(alpha[:, 1], (3, 1)))
        # p1 = res1 / d1
        feat1 = torch.cat((res1/(w[0] * G01)*GP0, res1/(w[2] * G21)*GP2),1)
        # print("feat1:",feat1)

        p1 = F.relu(self.hidden0_1(feat1))
        p1 = F.relu(self.hidden0_2(p1))
        p1 = F.relu(self.hidden0_3(p1))
        p1 = F.sigmoid(self.predict1(p1))*p_bar1
        # print("p1:",p1)

        d2 = w[0] * G02 / GP0 + w[1] * G12 / GP1
        res2 = torch.mm(w.T, torch.reshape(alpha[:, 2], (3, 1)))
        # p2 = res2 / d2
        feat2 = torch.cat((res2/(w[0] * G02)*GP0, res2/(w[1] * G12)*GP1),1)
        # print("feat2:", feat2)
        p2 = F.relu(self.hidden0_1(feat2))
        p2 = F.relu(self.hidden0_2(p2))
        p2 = F.relu(self.hidden0_3(p2))
        p2 = F.sigmoid(self.predict2(p2))*p_bar2
        # print("p2:",p2)

        p = torch.cat((p0, p1, p2), 1)
        # print("p:",p)
        # p = p_bar.t() - F.relu(p_bar.t() - p)

        # layer 2
        GP0 = torch.tensor([[G01 * p1, G02 * p2, 0.05]])
        GP1 = torch.tensor([[G10 * p0, G12 * p2, 0.05]])
        GP2 = torch.tensor([[G20 * p0, G21 * p1, 0.05]])

        d0 = w[1] * G10 / GP1 + w[2] * G20 / GP2
        res0 = torch.mm(w.T, torch.reshape(alpha[:, 0], (3, 1)))
        # p0 = res0 / d0
        feat0 = torch.cat((res0 / (w[1] * G10) * GP0, res0 / (w[2] * G20) * GP2), 1)
        # print("feat0:",feat0)

        p0 = F.relu(self.hidden1_1(feat0))
        p0 = F.relu(self.hidden1_2(p0))
        p0 = F.relu(self.hidden1_3(p0))
        p0 = F.sigmoid(self.predict0(p0)) * p_bar0
        # print("p0:",p0)

        d1 = w[0] * G01 / GP0 + w[2] * G21 / GP2
        res1 = torch.mm(w.T, torch.reshape(alpha[:, 1], (3, 1)))
        # p1 = res1 / d1
        feat1 = torch.cat((res1 / (w[0] * G01) * GP0, res1 / (w[2] * G21) * GP2), 1)
        # print("feat1:",feat1)

        p1 = F.relu(self.hidden1_1(feat1))
        p1 = F.relu(self.hidden1_2(p1))
        p1 = F.relu(self.hidden1_3(p1))
        p1 = F.sigmoid(self.predict1(p1)) * p_bar1
        # print("p1:",p1)

        d2 = w[0] * G02 / GP0 + w[1] * G12 / GP1
        res2 = torch.mm(w.T, torch.reshape(alpha[:, 2], (3, 1)))
        # p2 = res2 / d2
        feat2 = torch.cat((res2 / (w[0] * G02) * GP0, res2 / (w[1] * G12) * GP1), 1)
        # print("feat2:", feat2)
        p2 = F.relu(self.hidden1_1(feat2))
        p2 = F.relu(self.hidden1_2(p2))
        p2 = F.relu(self.hidden1_3(p2))
        p2 = F.sigmoid(self.predict2(p2)) * p_bar2
        # print("p2:",p2)

        p = torch.cat((p0, p1, p2), 1)

        #layer 3
        GP0 = torch.tensor([[G01 * p1, G02 * p2, 0.05]])
        GP1 = torch.tensor([[G10 * p0, G12 * p2, 0.05]])
        GP2 = torch.tensor([[G20 * p0, G21 * p1, 0.05]])

        d0 = w[1] * G10 / GP1 + w[2] * G20 / GP2
        res0 = torch.mm(w.T, torch.reshape(alpha[:, 0], (3, 1)))
        # p0 = res0 / d0
        feat0 = torch.cat((res0 / (w[1] * G10) * GP0, res0 / (w[2] * G20) * GP2), 1)
        # print("feat0:",feat0)

        p0 = F.relu(self.hidden2_1(feat0))
        p0 = F.relu(self.hidden2_2(p0))
        p0 = F.relu(self.hidden2_3(p0))
        p0 = F.sigmoid(self.predict0(p0)) * p_bar0
        # print("p0:",p0)

        d1 = w[0] * G01 / GP0 + w[2] * G21 / GP2
        res1 = torch.mm(w.T, torch.reshape(alpha[:, 1], (3, 1)))
        # p1 = res1 / d1
        feat1 = torch.cat((res1 / (w[0] * G01) * GP0, res1 / (w[2] * G21) * GP2), 1)
        # print("feat1:",feat1)

        p1 = F.relu(self.hidden2_1(feat1))
        p1 = F.relu(self.hidden2_2(p1))
        p1 = F.relu(self.hidden2_3(p1))
        p1 = F.sigmoid(self.predict1(p1)) * p_bar1
        # print("p1:",p1)

        d2 = w[0] * G02 / GP0 + w[1] * G12 / GP1
        res2 = torch.mm(w.T, torch.reshape(alpha[:, 2], (3, 1)))
        # p2 = res2 / d2
        feat2 = torch.cat((res2 / (w[0] * G02) * GP0, res2 / (w[1] * G12) * GP1), 1)
        # print("feat2:", feat2)
        p2 = F.relu(self.hidden2_1(feat2))
        p2 = F.relu(self.hidden2_2(p2))
        p2 = F.relu(self.hidden2_3(p2))
        p2 = F.sigmoid(self.predict2(p2)) * p_bar2
        # print("p2:",p2)

        p = torch.cat((p0, p1, p2), 1)
        return p


def custom_mse(predicted, target):
    total_mse = 0
    # print("predicted:", predicted)
    # print("target:", target)

    for i in range(target.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse += nn.MSELoss()(predicted.T[i], target.T[i])
    return total_mse


def training_progress(train_loader, train_xt, train_yt, test_xt, test_yt, y_test, single_data, single_target):
    epoch_num = 60
    cvxcnnreg = MLPPregression()
    optimizer = SGD(cvxcnnreg.parameters(), lr=0.0001, weight_decay=0.0001)
    loss_func = nn.MSELoss()  # 均方误差损失函数
    train_loss_all = []
    single_tp = []
    # sing_feat, w, G, v, p_max = single_instance_feat()
    # print("s_feat_st:", s_feat_st)
    # optimal_value = np.sum(single_target)
    optimal_value = single_target
    optimal_value = [optimal_value] * epoch_num
    # print("optimal_value:", optimal_value)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            # print("step:", step)
            output = cvxcnnreg(b_x)

            # print("output:", output)
            # loss = loss_func(output,b_y)
            loss = custom_mse(output, b_y)
            # print("output:", output)
            # print("b_y:", b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        single_predict = cvxcnnreg(single_data)
        p = single_predict.data.numpy()
        # print("====p:",p)
        # single_tp.append(np.sum(p))
        single_tp.append(p[0])
        print("train_loss_all:", train_loss / train_num)

    # plot loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # plt.figure(figsize=(13,9))
    plt.plot(train_loss_all, c='cornflowerblue', marker='o', markerfacecolor='none', label="Training loss")
    plt.legend(fontsize=10)
    plt.xlabel("epoch", fontsize=10)
    plt.ylabel("loss", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.savefig("./loss.pdf")
    # plt.show()

    # plt.figure(figsize=(13, 9))
    plt.subplot(1, 2, 2)
    plt.plot(single_tp, label="Supervised learning")
    plt.plot(optimal_value, linewidth=2, linestyle="-.", label="Optimal value")
    plt.legend(fontsize=10)
    plt.xlabel("epoch", fontsize=10)
    plt.ylabel("$w^Tr$", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig("./sinr_sum.pdf")
    plt.show()


if __name__ == '__main__':
    train_loader, train_xt, train_yt, test_xt, test_yt, y_test, single_data, single_target = data_progress()
    training_progress(train_loader, train_xt, train_yt, test_xt, test_yt, y_test, single_data, single_target)