from model import Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

# categories = ["n02085620-Chihuahua", "n02085782-Japanese_spaniel"]
categories = ["香り立つ旨み綾鷹", "伊藤園おーいお茶", "綾鷹コラボ", "颯"]
nb_classes = len(categories)

X_train = np.load("./tea_X_train_data.npy")
X_test = np.load("./tea_X_test_data.npy")
y_train = np.load("./tea_Y_train_data.npy")
y_test = np.load("./tea_Y_test_data.npy")

# データの正規化
X_train = torch.from_numpy(X_train.astype("float") / 255)
X_test = torch.from_numpy(X_test.astype("float") / 255)

# テンソルに変換
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# カテゴリをベクトルに変換
y_train = torch.nn.functional.one_hot(y_train.to(torch.int64), num_classes=nb_classes)
y_test = torch.nn.functional.one_hot(y_test.to(torch.int64), num_classes=nb_classes)

# データセットとデータローダーの作成
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

model = Model()
# model.load_state_dict(torch.load("./dog_model.pth"))

num_epochs = 10
batch_size = 6


# トレーニングループ
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = 0

    for batch_inputs, batch_labels in train_loader:
        model.optimizer.zero_grad()

        # 順伝播
        batch_outputs = model(batch_inputs)
        batch_outputs = batch_outputs.to(torch.double)
        batch_labels = batch_labels.to(torch.double)
        loss = model.criterion(batch_outputs, batch_labels)

        # 逆伝播とパラメータの更新
        loss.backward()
        model.optimizer.step()

        # ロスと正解率の計算
        epoch_loss += loss.item() * batch_inputs.size(0)
        predicted_labels = torch.round(batch_outputs)
        batch_accuracy = (predicted_labels == batch_labels).sum().item() / batch_labels.size(0)
        epoch_accuracy += batch_accuracy * batch_inputs.size(0)

        num_batches += 1

    epoch_loss /= len(train_dataset)
    epoch_accuracy /= len(train_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


# テストデータの評価
model.eval()
test_loss = 0.0
test_accuracy = 0.0
num_test_batches = 0

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_outputs = model(batch_inputs)
        loss = model.criterion(batch_outputs, batch_labels)

        test_loss += loss.item() * batch_inputs.size(0)
        predicted_labels = torch.round(batch_outputs)
        batch_accuracy = (predicted_labels == batch_labels).sum().item() / batch_labels.size(0)
        test_accuracy += batch_accuracy * batch_inputs.size(0)

        num_test_batches += 1

test_loss /= len(test_dataset)
test_accuracy /= len(test_dataset)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

torch.save(model.state_dict(), "./dog_model.pth")
