import  cv2
import math
import numpy as np



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




if __name__ == "__main__":
    img_gray = cv2.imread(r'C:/Users/lenovo/Desktop/src/src.jpg', 0)  # 读取灰度图像
    img_bgr = cv2.imread(r'C:/Users/lenovo/Desktop/thing/project/image/chip.png')
