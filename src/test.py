from model import Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from PIL import Image

# categories = ["n02085620-Chihuahua", "n02085782-Japanese_spaniel"]
categories = ["おーいお茶", "麦茶"]
nb_classes = len(categories)
"""
X_data = np.load("./tea_X_data.npy")
Y_data = np.load("./tea_Y_data.npy")

# データの正規化、テンソルに変換
X_data = torch.from_numpy(X_data.astype("float") / 255)
Y_data = torch.from_numpy(Y_data)

# カテゴリをベクトルに変換
Y_data = torch.nn.functional.one_hot(Y_data.to(torch.int64), num_classes=nb_classes)
"""
model = Model()
model.load_state_dict(torch.load("./dog_model.pth"))


def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")  # RGB形式に変換
    img = img.resize((150, 150))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)


# テストデータの評価
model.eval()
with torch.no_grad():
    if input("おーいお茶: 1, 麦茶:2") == "1":
        files = [(0, f"D:/Python/TeaRecognition/src/Images/おーいお茶/0000{input('ファイル番号(0000nn): ')}.jpg")]
    else:
        files = [(1, f"D:/Python/TeaRecognition/src/Images/麦茶/0000{input('ファイル番号(0000nn): ')}.jpg")]
    X, Y = [], []
    for cat, frame in files:
        add_sample(cat, frame)
    X_data, Y_data = np.array(X), np.array(Y)
    # データの正規化、テンソルに変換
    X_data = torch.from_numpy(X_data.astype("float") / 255)
    Y_data = torch.from_numpy(Y_data)

    # カテゴリをベクトルに変換
    Y_data = torch.nn.functional.one_hot(Y_data.to(torch.int64), num_classes=nb_classes)
    batch_outputs = model(X_data)
    print(batch_outputs[0])
    print(batch_outputs[0].argmax(), Y_data)
