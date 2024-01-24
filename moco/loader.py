import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def data_check(q_list, k_list):
    for idx, (q_file, k_file) in enumerate(zip(q_list, k_list)):
        print(f"Pair {idx}")
        assert q_file[:-8] == k_file[:-4]


class MoCoData(Dataset):
    def __init__(self, data_dir, pair_list):
        super(MoCoData, self).__init__()
        """
        后续计划做成外部list.txt形式
        目前先采用现有数据测试
        !!!使用时记得做归一化
        """
        self.data_dir = data_dir
        with open(pair_list, "r") as f:
            self.pair_info = f.readlines()
        # data_check(self.q_file_list, self.k_file_list)

        self.img_size = 1280
        self.patch_size = 64

    def __getitem__(self, idx):
        pair_info_ = self.pair_info[idx]
        pair_info_ = pair_info_[:-1].split(" ")
        k_file = pair_info_[1]
        q_file = pair_info_[0]
        is_aug = True if pair_info_[2] == "1" else False

        img_k = self.read_and_resize_(k_file)
        img_q = self.read_and_resize_(q_file)

        if is_aug:
            img_q, img_k = self.patch_merge_(img_q, img_k)

        # Convert
        img_q = img_q.reshape((img_q.shape[0], img_q.shape[1], 1))
        img_q = img_q.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_q = np.ascontiguousarray(img_q)

        img_k = img_q.reshape((img_k.shape[0], img_k.shape[1], 1))
        img_k = img_k.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_k = np.ascontiguousarray(img_k)

        return {
            "img_q": torch.from_numpy(img_q.copy()),
            "img_k": torch.from_numpy(img_k.copy())
        }

    def __len__(self):
        return len(self.pair_info)

    def read_and_resize_(self, file_name):
        img = cv2.imread(os.path.join(self.data_dir, file_name), 0)
        h_, w_ = img.shape
        ratio_ = self.img_size / max(h_, w_)
        tmp_ = cv2.resize(img, (int(w_ * ratio_), int(h_ * ratio_)))
        h_, w_ = tmp_.shape
        dst = np.ones((self.img_size, self.img_size)) * 114
        start_y_ = int((self.img_size - h_) * 0.5)
        start_x_ = int((self.img_size - w_) * 0.5)
        dst[start_y_:start_y_ + h_, start_x_:start_x_ + w_] = tmp_
        return dst.astype(np.uint8)

    def patch_merge_(self, img_q, img_k):
        patch_cnt_y = img_q.shape[0] // self.patch_size
        patch_cnt_x = img_q.shape[1] // self.patch_size
        cand_list = [i for i in range(patch_cnt_y * patch_cnt_x)]

        patch_q = img_q.reshape((patch_cnt_y, self.patch_size, patch_cnt_x, self.patch_size))
        patch_k = img_k.reshape((patch_cnt_y, self.patch_size, patch_cnt_x, self.patch_size))
        tmp_ = patch_q.copy()

        random.shuffle(cand_list)
        for i in range(len(cand_list) // 2):
            y_ = int(cand_list[i] // patch_cnt_x)
            x_ = int(cand_list[i] % patch_cnt_x)
            patch_q[y_, :, x_, :] = patch_k[y_, :, x_, :]
            patch_k[y_, :, x_, :] = tmp_[y_, :, x_, :]

        img_q = patch_q.reshape((img_q.shape[0], img_q.shape[1]))
        img_k = patch_k.reshape((img_k.shape[0], img_k.shape[1]))
        return img_q, img_k


if __name__ == "__main__":
    print("Clone Dolly Presents.")
    train_set_ = MoCoData("E:\\datasets\\MoCoDataset", "E:\\datasets\\MoCoDataset\\train_list.txt")
    train_loader_ = DataLoader(train_set_, batch_size=8, shuffle=True)
    for data_ in train_loader_:
        input_q = data_["img_q"]
        input_k = data_["img_k"]
