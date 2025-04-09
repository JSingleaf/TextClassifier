import os.path

import jieba
import pandas as pd
from collections import Counter

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import Net
from config import *

# 初始化超参数
config = Config()

# 预处理数据
data = pd.read_csv(config.data_path)

origin_texts = data['text'].values
origin_labels = data['label'].values

words = []
for text in origin_texts:
    words.extend(jieba.lcut(text))
counted_words = Counter(words)
filtered_words = [word for word, count in counted_words.items() if count > Config().min_freq]

vocab = {"<PAD>": 0, "UNK": 1}
for word in filtered_words:
    if word not in vocab:
        vocab[word] = len(vocab)

divided_texts = [
    [vocab[word] if word in vocab else 1 for word in jieba.lcut(origin_texts[index])]
    for index in range(len(origin_texts))
]


# 构建数据集


class MyDataset(Dataset):
    def __init__(self, texts, labels, vocab, text_size):
        texts = [F.pad(torch.tensor(text), (0, config.text_size - len(text))) for text in divided_texts]
        self.texts = torch.stack(texts)
        self.labels = torch.tensor(labels)
        self.vocab = vocab
        self.text_size = text_size

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        return text, label

    def __len__(self):
        return len(self.texts)


dataset = MyDataset(divided_texts, origin_labels, vocab, config.text_size)

# 拆分数据集为训练集和测试集
train_data, test_data = random_split(dataset=dataset,
                                     lengths=[int(config.train_data_rate * len(dataset)),
                                              len(dataset) - int(config.train_data_rate * len(dataset))])

# 载入数据集
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True, drop_last=True)

# 构建模型实例
model = Net(
    vocab_size=len(vocab),
    text_size=config.text_size,
    embedding_dim=config.embedding_dim,
    num_filter=config.num_filter,
    kernel_size=config.kernel_size,
    num_classes=config.num_classes
).to(config.device)

if config.is_Trained and os.path.exists(config.model_save_path):
    model.load_state_dict(torch.load(config.model_save_path))

loss_fn = nn.CrossEntropyLoss().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

# 创建Tensorboard
writer = SummaryWriter(config.log_path)

# 训练和验证
total_train_step = 0
best_test_acc = 0

for epoch in range(config.epoch_num):

    print('-' * 10, f"第{epoch + 1}轮训练", '-' * 10)

    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(config.device), labels.to(config.device)

        outputs = model(texts)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        total_loss += loss.item()

        writer.add_scalar('Loss', loss.item(), total_train_step)

    print(f"Total_loss: {total_loss}")

    total_correct = 0
    model.eval()
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(config.device), labels.to(config.device)
            outputs = torch.argmax(model(texts), dim=1)
            total_correct += (outputs == labels).sum()

        writer.add_scalar('Accuracy', total_correct / len(test_data), epoch + 1)

        print(f"Accuracy: {total_correct / len(test_data)}")

        # Save Model
        if total_correct > best_test_acc:
            best_test_acc = total_correct
            torch.save(model.state_dict(), config.model_save_path)
