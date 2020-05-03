import  cv2
import math
import numpy as np
import skimage
from skimage.measure import compare_ssim


def getPSNR(img_1, img_2):
    psnr = cv2.PSNR(img_1, img_2)
    print("PSNR IS ", psnr)
    return psnr


def getEntropy(img):
    """
    param: img:narray 二维灰度图像
    return: float 图像约清晰越大
    """
    out = 0
    count = np.shape(img)[0] * np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i] != 0:
            out -= p[i] * math.log(p[i] / count) / count
    return out


def getMean(img):
    mean = np.mean(img)
    img_size = np.shape(img)[0] * np.shape(img)[1]
    mean = mean / img_size
    return mean


def variance(img):
    """
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    """
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out


def getSSIM(img_1, img_2):
    ssim = compare_ssim(img_1, img_2)
    return ssim


if __name__ == "__main__":
    img_src = cv2.imread(r'C:/Users/lenovo/Desktop/src/src.jpg', 0)  # 读取灰度图像
