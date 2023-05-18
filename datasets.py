import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        # 初始化方法，用於設置物件的屬性和執行一些初始操作
        # root: 資料集的根目錄
        # transforms_: 圖像轉換操作的列表
        # mode: 模式，可以是 "train" 或 "test"

        # 將 transforms_ 轉換操作列表組合成一個串聯的轉換操作
        self.transform = transforms.Compose(transforms_)

        # 使用 glob.glob 方法找到指定目錄下的所有檔案，並將它們排序
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

        # 如果模式是 "train"，則將 "test" 目錄下的檔案也加入 self.files 列表中
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
            
    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
