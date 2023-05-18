from PIL import Image
import os, glob
import numpy as np
import random, math

# 画像ディレクトリのパス
root_dir = "./Images"
# 商品名
categories = ["n02085620-Chihuahua", "n02085782-Japanese_spaniel"]

# 画像データ用配列
X = []
# ラベルデータ用配列
Y = []


# 画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
        print(len(X), len(Y))
    return np.array(X), np.array(Y)


# 渡された画像データを読み込んでXに格納し、また、
# 画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")  # RGB形式に変換
    img = img.resize((150, 150))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)


# 全データ格納用配列
allfiles = []

# カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat  # それぞれの商品用のパス
    files = glob.glob(image_dir + "/*.jpg")  # jpgファイルをすべて取得
    for f in files[:100]:
        allfiles.append((idx, f))

# シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)

print(X_train.size, y_train.size)
print(len(X_train), len(y_train))
print(y_train)

# データを保存する（データの名前を「tea_data.npy」としている）
np.save("./tea_X_train_data.npy", X_train)
np.save("./tea_Y_train_data.npy", y_train)
np.save("./tea_X_test_data.npy", X_test)
np.save("./tea_Y_test_data.npy", y_test)
