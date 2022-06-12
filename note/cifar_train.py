import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 获取机器上的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载cifar数据
dataset_path = "../dataset/data_cifar10"
dataset_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
dataset_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

dataloader_train = DataLoader(dataset=dataset_train, batch_size=64)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=64)

dataset_train_len = len(dataset_train)
dataset_test_len = len(dataset_test)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x.to(device)
        return self.model(x)


# define the net and constant
my_model = MyModel()
my_model.to(device)

# 告知更新的参数，和设置学习率
my_optimize = torch.optim.SGD(my_model.parameters(), lr=1e-2)
# 设置学习率
my_loss_fn = torch.nn.CrossEntropyLoss()

train_step = 0
test_step = 0
max_epoch = 100
writer = SummaryWriter("./logs")
# begin training and testing
for epoch in range(max_epoch):
    print("-------The {} Epoch is Running!-------".format(epoch))

    # train the data
    my_model.train()
    train_sum_loss = 0
    for images, targets in dataloader_train:
        outputs = my_model(images)
        train_loss = my_loss_fn(outputs, targets)
        train_sum_loss += train_loss.item()
        if train_step % 100 == 0:
            print(f"train {epoch}, step:{train_step}, train_loss{train_loss}")
        # 梯度清零，不清零则会累积，通常一个step清空一次，
        my_optimize.zero_grad()

        # 反向传递求所有的参数的梯度
        train_loss.backward()

        # 根据梯度更新参数，更新my_optimize设置的参数，已在前面声明
        my_optimize.step()

        train_step += 1

    print(f"train {epoch}, train_epoch_loss{train_sum_loss}")
    writer.add_scalar("train epoch loss", train_sum_loss, epoch)

    # test the data
    my_model.eval()
    with torch.no_grad():
        test_sum_loss = 0
        predict_right_cnt = 0
        for images, targets in dataloader_test:
            output = my_model(images)
            test_loss = my_loss_fn(output, targets)
            test_sum_loss += test_loss.item()
            predict_right_cnt += (torch.argmax(output, dim=1) == targets).sum()

    writer.add_scalar(f"predict right rate", predict_right_cnt / dataset_test_len, epoch)
    print(f"test {epoch}, test_epoch_loss{test_sum_loss}")
    writer.add_scalar("test epoch loss", test_sum_loss, epoch)

    # save the model!!!!!!!
    torch.save(my_model.state_dict(), f"../output/cifar/train_model.pth")

writer.close()