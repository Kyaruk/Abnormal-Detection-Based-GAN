import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from random import sample


class MyGAN:
    def __init__(self, N_IDEAS, BATCH_SIZE, data):
        self.LR_G = 0.0001
        self.LR_D = 0.0001
        self.BATCH_SIZE = BATCH_SIZE
        self.N_IDEAS = N_IDEAS
        self.ART_COMPONETS = 1
        for value in data.values():
            self.ART_COMPONETS = len(value)
            break
        # self.ART_COMPONETS = ART_COMPONETS
        self.data = data
        self.PAINT_POINTS = np.vstack([np.linspace(-1, 1, self.ART_COMPONETS) for _ in range(self.BATCH_SIZE)])
        self.G = nn.Sequential(
            # cnn_g,
            nn.Linear(self.N_IDEAS, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, self.ART_COMPONETS)
        )
        self.D = nn.Sequential(
            # cnn_d,
            nn.Linear(self.ART_COMPONETS, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def artist_work(self):
        a = np.random.uniform(1, 2, size=self.BATCH_SIZE)[:, np.newaxis]
        paints = a * np.power(self.PAINT_POINTS, 2) + (a - 1)
        paints = torch.from_numpy(paints).float()
        return paints

    def learning(self):
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.LR_G)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.LR_D)

        plt.ion()

        true_datas = []
        for key, value in self.data.items():
            true_data = np.array(value)
            # true_datas.append(torch.from_numpy(true_data))
            true_datas.append(true_data)
        # true_datas = np.array(true_datas)

        train_datas = torch.from_numpy(np.array(sample(true_datas, self.BATCH_SIZE)))
        # artist_painting = torch.tensor(train_datas, dtype=torch.float32)
        artist_painting = train_datas.clone().detach().type(torch.float32)

        for step in range(1000):
            # artist_painting = self.artist_work()
            # artist_painting = torch.tensor(true_datas, dtype=torch.float32)
            G_idea = torch.randn(self.BATCH_SIZE, self.N_IDEAS)
            # G_idea = torch.randn(artist_painting.shape[0], self.N_IDEAS)

            G_paintings = self.G(G_idea)

            pro_atrist0 = self.D(artist_painting)
            pro_atrist1 = self.D(G_paintings)
            # print("step = " + str(step) + ", pro_atrist0 = " + str(
            #     pro_atrist0.data.numpy().mean()) + ", pro_atrist1 = " + str(pro_atrist1.data.numpy().mean()))

            G_loss = -1 / torch.mean(torch.log(1. - pro_atrist1))
            D_loss = -torch.mean(torch.log(pro_atrist0) + torch.log(1 - pro_atrist1))

            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_G.step()
            optimizer_D.step()

            # if step % 2000 == 0:  # plotting
            #     print("step = " + str(step) + ", pro_atrist0 = " + str(
            #         pro_atrist0.data.numpy().mean()) + ", pro_atrist1 = " + str(pro_atrist1.data.numpy().mean()))
            #     plt.cla()
            #     plt.plot(self.PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3,
            #              label='Generated painting', )
            #     # plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            #     # plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            #     plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pro_atrist0.data.numpy().mean(),
            #              fontdict={'size': 13})
            #     # plt.text(-.5, 2, 'G_loss= %.2f ' % G_loss.data.numpy(), fontdict={'size': 13})
            #
            #     plt.ylim((0, 3))
            #     plt.legend(loc='upper right', fontsize=10)
            #     plt.draw()
            #     plt.pause(0.1)

        # plt.ioff()
        # plt.show()
        return self.G, self.D

    def detect_G(self, data):
        G_idea = torch.randn(self.BATCH_SIZE, self.N_IDEAS)
        G_paintings = self.G(G_idea)
        G_paintings = G_paintings.detach().numpy()
        result = {}
        for key, value in data.items():
            G_data = np.array(torch.tensor(value, dtype=torch.float32))
            res = np.linalg.norm(G_data - G_paintings)
            result[key] = res
        return result

    def detect_D(self, data):
        result = {}
        for key, value in data.items():
            pro_atrist = self.D(torch.tensor(value, dtype=torch.float32))
            res = pro_atrist.data.numpy().mean()
            result[key] = res
        return result
