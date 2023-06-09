from model import Model

import torch, datetime
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

start = datetime.datetime.now()
print("開始 :", start)

categories = ["香り立つ旨み綾鷹", "伊藤園おーいお茶", "颯"]
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

num_epochs = 10

# トレーニングループ
for epoch in range(num_epochs):
    model.train()
    epoch_accuracy = 0.0
    total_data_len = 0
    total_correct = 0
    total_loss = 0.0

    for batch_inputs, batch_labels in train_loader:

        # 順伝播
        batch_outputs = model(batch_inputs)
        model.optimizer.zero_grad()
        batch_outputs = batch_outputs.to(torch.double)
        batch_labels = batch_labels.to(torch.double)
        loss = model.criterion(batch_outputs, batch_labels)

        # 逆伝播とパラメータの更新
        loss.backward()
        model.optimizer.step()

        # ロスと正解率の計算
        for i in range(len(batch_labels)):
            total_data_len += 1
            if torch.argmax(batch_outputs[i]) == torch.argmax(batch_labels[i]):
                total_correct += 1
            total_loss += loss.item()

    epoch_accuracy = total_correct / total_data_len

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/total_data_len:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # テストデータの評価
    model.eval()
    total_data_len = 0.0
    total_correct = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_outputs = model(batch_inputs)

            for i in range(len(batch_labels)):
                total_data_len += 1
                if torch.argmax(batch_outputs[i]) == torch.argmax(batch_labels[i]):
                    total_correct += 1

    test_accuracy = total_correct / total_data_len

    print(f"Test Accuracy: {test_accuracy:.4f}")
    torch.save(model.state_dict(), f"./{datetime.datetime.now().minute}_model.pth")

print("終了 :", datetime.datetime.now() - start)

# torch.save(model.state_dict(), "./tea_model.pth")
