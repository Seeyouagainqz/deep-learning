import json
import sys

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from pythonProject1.Googlenet.model import GoogLeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }

    # train_data = torchvision.datasets.CIFAR10(root="../data",train=True, transform=data_transform["train"],
    #                                           download=True)
    # test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=data_transform["val"],
    #                                          download=True)
    train_data = torchvision.datasets.ImageFolder(root="../data/flower_data/train", transform=data_transform["train"])
    test_data = torchvision.datasets.ImageFolder(root="../data/flower_data/val", transform=data_transform["val"])

    train_num = len(train_data)
    test_num = len(test_data)
    print(train_num)     # 3306
    print(test_num)      # 364

    flwoer_list = train_data.class_to_idx   # 得到他的一个索引
    cla_dict = dict((val, key) for key, val in flwoer_list.items())   # 实现val和 key 的一个反转
    json_str = json.dumps(cla_dict, indent=4)   # 将cla_dict  编码成json 格式
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)   # 将 json_str 写入到 json_file文件中

    batch_size = 48
    train_downloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_downloader = DataLoader(test_data, batch_size, shuffle=True)

    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
        loss_fn = loss_fn.cuda()

    # model_weigths_path = "./GoogleNet.pth"
    # net.load_state_dict(torch.load(model_weigths_path))

    learning_rate = 0.0003
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # save_path = "./GoogleNet.pth"
    save_path = "./flower-axu1.pth"
    best_acc = 0.0
    epochs = 30

    writer = SummaryWriter('./weigths-flower1')
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_downloader, file=sys.stdout)
        train_acc = 0.0
        for step, data in enumerate(train_bar):
            img, labels = data
            # print(labels)
            if torch.cuda.is_available():
                img = img.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            # 这一部分出现了问题
            # logits, logits_aux2, logits_aux1 = net(img)
            logits_aux2 = net(img)
            # print("logits:{}".format(logits))

            # loss0 = loss_fn(logits, labels)
            # loss_aux1 = loss_fn(logits_aux1, labels)
            loss_aux2 = loss_fn(logits_aux2, labels)
            # loss = loss0 + loss_aux1*0.3 + loss_aux2*0.3
            loss = loss_aux2
            # loss0 = loss_fn(logits, labels)
            # loss1 = loss_fn(logits_aux1, labels)
            # loss2 = loss_fn(logits_aux2, labels)
            # loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            # print("loss0:{}".format(loss0))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()   # 张量转换成熟知的形式
            # 打印训练的进度
            # rate = (step + 1) / len(train_downloader)
            # a = "*" * int(rate * 50)
            # b = "." * int((1 - rate) * 50)
            # print("\rtrain_loss:{:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
            predict_x = torch.max(logits_aux2, dim=1)[1]
            train_acc = torch.eq(predict_x, labels).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,
                                                                     epochs+0,
                                                                     loss)

        # 测试过程你    验证过程中 sekf.training = False
        net.eval()
        acc = 0.0

        with torch.no_grad():
            # val_bar = tqdm(test_downloader, file=sys.stdout)
            for val_data in test_downloader:

                val_images, val_labels = val_data
                if torch.cuda.is_available():
                    val_images = val_images.cuda()
                    val_labels = val_labels.cuda()
                outputs = net(val_images)  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / test_num
        print('val [epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_num, val_accurate))

        train_accurate = train_acc/train_num
        writer.add_scalar("train_accutacy", train_accurate, epoch+1)
        writer.add_scalar("train_loss", running_loss/train_num, epoch+1)
        writer.add_scalar("test_accuracy", val_accurate, epoch+1)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()





