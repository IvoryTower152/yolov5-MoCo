import os
import cv2
import rawpy
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_dir = "E:\\datasets\\Sony\\Sony\\short"
    dst_dir = "E:\\datasets\\Sony\\Sony_data\\short"
    for file_ in os.listdir(data_dir):
        print(file_)

        # for long
        """
        raw_img = rawpy.imread(os.path.join(data_dir, file_))
        img = raw_img.postprocess(half_size=False, no_auto_bright=True, output_bps=16)
        img = img.astype(np.float32) / 256.0
        img[img > 256] = 256
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(dst_dir, f"{file_[:-4]}.jpg"), img)
        """

        # for short
        raw_img = rawpy.imread(os.path.join(data_dir, file_))
        bayer_visible = raw_img.raw_image_visible
        height, width = bayer_visible.shape
        bayer_visible = bayer_visible.reshape((height * width))
        cand_list = [i for i in range(0, height * width, 4)]
        img = bayer_visible[cand_list] / 256.0
        # for i in range(int(height//2)):
        #     offset = i * int(width//2)
        #     for j in range(int(width//2)):
        #         img[i, j] = bayer_visible[(offset + j) * 4] / 256.0
        img[img > 256] = 256
        img = img.reshape((int(height//2), int(width//2)))
        img = img.astype(np.uint8)
        cv2.imwrite(os.path.join(dst_dir, f"{file_[:-4]}.jpg"), img)
