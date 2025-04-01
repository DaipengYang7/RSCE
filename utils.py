import os
import shutil
import numpy as np
import cv2


def rm_mkdir_my(dir, isRemove=False):
    # If the fileDir is BEING and it is need to remove, remove the old and create the new
    if os.path.exists(dir) and isRemove:
        shutil.rmtree(dir)
        os.mkdir(dir)
    elif os.path.exists(dir) and not isRemove:
        pass
    else:
        os.mkdir(dir)

def writeImage(image, imageName, isMaxValTo1, dirname):
    curDir = os.curdir  # 当前目录
    savedDir = os.path.join(curDir, dirname)  # 图片保存目录
    rm_mkdir_my(savedDir, False)  # 创建图片保存目录

    savedPath = os.path.join(savedDir, imageName + '.png')
    # uint8
    if image.dtype == np.uint8:
        cv2.imwrite(savedPath, image)
    elif image.dtype == np.float32 or image.dtype == np.float64:
        if isMaxValTo1:
            image = image / (image.max() + 1e-8)
        image = image * 255
        cv2.imwrite(savedPath, image)
    elif image.dtype == np.int32:  # 可视化掩模
        image = image * 255
        image = image.astype(np.uint8)
        cv2.imwrite(savedPath, image)
    else:
        print("The saved image's format is incorrect, please check the code.")

def computeIdxDis(i1, i2, mod):
    if i1 < i2:
        return min(i2 - i1, (i1 + mod - i2) % mod)
    else:
        return min(i1 - i2, (i2 + mod - i1) % mod)

def computeColorSimilar(c1, c2):
    dot_product = 0
    square_sum_x = 0
    square_sum_y = 0
    for i in range(c1.shape[0]):
        dot_product += c1[i] * c2[i]
        square_sum_y += c1[i] * c1[i]
        square_sum_y += c2[i] * c2[i]
    cosSimilar = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))
    return 0.5 * cosSimilar + 0.5  # 归一化到 [0, 1] 区间

