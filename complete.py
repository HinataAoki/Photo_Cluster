import os
import shutil
import numpy as np
from PIL import Image
from skimage import data
from sklearn.cluster import KMeans
import cv2
from pathlib import Path

#関数を定義
def get_img_paths(img_dir):
    """
    画像のパスを取得する。
    """
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
    img_paths = [p for p in img_dir.iterdir() if p.suffix in IMG_EXTENSIONS]

    return img_paths

print('クラスタ数を入力')
n_clusters=int(input())

# 1. 国旗画像のサイズをそろえて保存する
# ./flag_origin 以下に国旗画像
# ./flag_convert 以下に300*200のサイズに変換したjpgを保存
for path in os.listdir('./flag_origin'):
    img = Image.open(f'./flag_origin/{path}')
    img = img.convert('L')
    img_resize = img.resize((300, 200))
    os.makedirs('./flag_convert',exist_ok=True)
    img_resize.save(f'./flag_convert/{path}.jpg')

# 2. 3次元配列の画像データを2次元配列のデータに変換
feature = np.array([cv2.imread(f'./flag_convert/{path}') for path in os.listdir('./flag_convert')])
feature = feature.reshape(len(feature), -1).astype(np.float64)

# 3. 学習(15種類のグループにクラスタリングする)
model = KMeans(n_clusters=n_clusters).fit(feature)

# 4. 学習結果のラベル
labels = model.labels_

# 5. 学習結果(クラスタリング結果の表示 + ラベルごとにフォルダ分け)
# ./flag_group 以下に画像を分けて保存する
os.makedirs("./flag_group",exist_ok=True)
for label, path in zip(labels, os.listdir('./flag_convert')):
    os.makedirs(f"./flag_group/{label}", exist_ok=True)
    shutil.copyfile(f"./flag_origin/{path.replace('.jpg', '')}", f"./flag_group/{label}/{path.replace('.jpg', '')}")
    print(label, path)

#flag_groupの中にconvertフォルダを作成,分類
for label in labels:
    for path in os.listdir(f'./flag_group/{label}'):
        img = Image.open(f'./flag_group/{label}/{path}')
        img_resize = img.resize((300, 200))
        os.makedirs(f"./flag_group/convert/{label}", exist_ok=True)
        img_resize.save(f'./flag_group/convert/{label}/{path}.jpg')

#convert>number>にmean.pngを生成
for label in labels:
    imgs = []
    input_dir = Path(f'./flag_group/convert/{label}/')
    for path in get_img_paths(input_dir):
        # 画像を読み込む。
        img = cv2.imread(str(path))
        imgs.append(img)
    imgs = np.array(imgs)
    mean_img = imgs.mean(axis=0)
    cv2.imwrite(f"./flag_group/convert/{label}/mean.png", mean_img)

os.makedirs(f"./flag_means", exist_ok=True)
for label in labels:
    img=Image.open(f'./flag_group/convert/{label}/mean.png')
    img.save(f'./flag_means/mean_{label}.png')