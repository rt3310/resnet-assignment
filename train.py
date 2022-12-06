from data.DataLoader import DataLoader
from model.ResNet34 import ResNet34
from test import test
from trainer.Trainer import Trainer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    batch_size = 10
    image_h = 64
    image_w = 64
    dataset = DataLoader("./resources/dataset/digit_data", "train_data.txt", batch_size, image_h, image_w)

    model = ResNet34(20)

    init_lr = 0.01
    train = Trainer(model, dataset, 20, init_lr)
    loss = []
    accurate = []
    temp = 0

    model.train()
    plt.figure(figsize=(10, 5))
    plt.ion()
    for i in range(2000):
        temp = train.iterate()
        loss.append(temp / 10)
        print("iteration = {} || loss = {}".format(str(i), str(temp / 10)))
        if i % 100 == 0 and i != 0:
            model.eval()
            accurate.append(test(model, "./resources/dataset/digit_data", "valid_data.txt", image_h, image_w).get())
            model.save("model2")
            model.train()
        plt.cla()
        plt.subplot(1, 2, 1)
        plt.plot(loss)
        plt.subplot(1, 2, 2)
        plt.plot(accurate)
        plt.pause(0.1)

        if i == 15000:
            train.set_lr(0.001)


    plt.ioff()
    plt.show()
