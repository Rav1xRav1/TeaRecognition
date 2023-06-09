from PIL import Image
import os, glob
import numpy as np
import random, math

import matplotlib.pyplot as plt

# 画像ディレクトリのパス
root_dir = "./Images"
# 商品名
# categories = ["n02085620-Chihuahua", "n02085782-Japanese_spaniel"]
categories = ["香り立つ旨み綾鷹", "伊藤園おーいお茶", "颯"]

# 画像データ用配列
X = []
# ラベルデータ用配列
Y = []


# 画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files):
    global X, Y
    X, Y = [], []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)


# 渡された画像データを読み込んでXに格納し、また、
# 画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")  # RGB形式に変換
    img = img.resize((240, 320))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)


def labeling_one_data():
    # files = [(0, "./Images/n02085620-Chihuahua/n02085620_12101.jpg")]
    # files = [(0, "D:/Python/TeaRecognition/src/Images/麦茶/000010.jpg")]
    files = [(0, "D:/Python/TeaRecognition/src/Images/おーいお茶/000010.jpg")]
    global X, Y
    X, Y = [], []
    for cat, frame in files:
        add_sample(cat, frame)
    X_data, Y_data = np.array(X), np.array(Y)
    np.save("./tea_X_data.npy", X_data)
    np.save("./tea_Y_data.npy", Y_data)


def show_image(image_array):
    plt.imshow(image_array)
    plt.axis("off")
    plt.show()


# labeling_one_data()
# exit()

# 全データ格納用配列
allfiles = []

# カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat  # それぞれの商品用のパス
    files = glob.glob(image_dir + "/*.jpg")  # jpgファイルをすべて取得
    print(cat, len(files))
    for f in files[:95]:
        allfiles.append((idx, f))

# シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)

# データを保存する（データの名前を「tea_data.npy」としている）
np.save("./tea_X_train_data.npy", X_train)
np.save("./tea_Y_train_data.npy", y_train)
np.save("./tea_X_test_data.npy", X_test)
np.save("./tea_Y_test_data.npy", y_test)
