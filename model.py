import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, vocab_size, text_size, embedding_dim, num_filter, kernel_size, num_classes):
        super(Net, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 卷积层
        self.conv2d = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1, out_channels=num_filter, kernel_size=(n_gram, embedding_dim), bias=True
                ) for n_gram in kernel_size
            ]
        )

        # 池化层
        self.maxpool2d = nn.ModuleList(
            [
                nn.MaxPool2d(
                    kernel_size=(text_size - n_gram + 1, 1)
                ) for n_gram in kernel_size
            ]
        )

        self.relu = nn.ReLU()

        # 线性层
        self.fc = nn.Linear(num_filter * len(kernel_size), num_classes)

    def forward(self, input):
        embedding = self.embedding(input).unsqueeze(1)

        x = [conv(embedding) for conv in self.conv2d]
        x = [self.relu(x) for x in x]
        x = torch.cat([maxpool(x) for x, maxpool in zip(x, self.maxpool2d)], dim=1).squeeze()

        x = self.fc(x)
        return x
