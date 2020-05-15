import numpy as np
import random
import matplotlib.pyplot as plt
import pylab
import cv2
import skimage as sk

#设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
pylab.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
pylab.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def GaussianNoise(img, mean=0, var=0.01):
    """
    添加高斯噪声
    mean : 均值
    var : 方差
    """
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)  # 标准正态分布
    output = img + noise
    output = np.clip(output, 0.0, 1.0)  # 防止像素溢出
    output = np.uint8(output * 255)

    return output


def SaltAndPepperNoise(img, prob):
    """
    添加椒盐噪声
    image:原始图像
    prob:噪声比例(在0到1之间)
    """
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  # 返回随机的浮点数，在半开区间 [0.0, main)
            if rdn < prob:
                output[i, j] = 0
            elif rdn > thres:
                output[i, j] = 255
            else:
                output[i, j] = img[i, j]
    return output

def EquColor(img):  # 待画出直方图
    Hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = cv2.equalizeHist(Hsv[:, :, 2])
    Hsv[:, :, 2] = v
    output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2RGB)
    return output

def AdaptiveEquColor(img):
    Hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    v = clahe.apply(Hsv[:, :, 2])
    Hsv[:, :, 2] = v
    output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2RGB)
    return output, v

def GammaTransform(img, gamma):
    Hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    Value = np.power(Hsv[:, :, 2] / 255., gamma)
    Value = np.clip(Value * 255, 0, 255)  # 防止像素溢出
    Hsv[:, :, 2] = np.uint8(Value)
    output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2RGB)
    return output

def MSRCR(imgg, beta=46.0, alpha=125.0, G=5.0, b=25.0, low_clip=0.01, high_clip=0.99):

        sigma_list = [15, 80, 250]

        # multiScaleRetinex
        img = np.float64(imgg) + 1.0  # 防止0取对数的情况发生
        img_retinex = np.zeros_like(img)
        for sigma in sigma_list:
            img_retinex += np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))  # 主要是此步骤计算时间过长
            print(sigma)
        img_retinex = img_retinex / len(sigma_list)

        # colorRestoration
        img_sum = np.sum(img, axis=2, keepdims=True)
        img_color = beta * (np.log10(alpha * img) - np.log10(img_sum))

        # 经验公式
        img_msrcr = G * (img_retinex * img_color + b)

        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                 (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                 255

        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        # simplestColorBalance
        total = img_msrcr.shape[0] * img_msrcr.shape[1]
        for i in range(img_msrcr.shape[2]):
            unique, counts = np.unique(img_msrcr[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c

            img_msrcr[:, :, i] = np.maximum(np.minimum(img_msrcr[:, :, i], high_val), low_val)
        return img_msrcr

img = cv2.imread(r'C:/Users/lenovo/Desktop/thing/project/image/cat.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


'''''
noise_1 = SaltAndPepperNoise(img, 0.01)
noise_2 = SaltAndPepperNoise(img, 0.05)
plt.figure("椒盐噪声")
ax = plt.subplot(131)
ax.set_title('prob=0')
plt.imshow(img), plt.axis('off')
ax = plt.subplot(132)
ax.set_title('prob=0.01')
plt.imshow(noise_1), plt.axis('off')
ax = plt.subplot(133)
ax.set_title('prob=0.05')
plt.imshow(noise_2), plt.axis('off')
plt.show()
'''''


'''''
noise_guass = GaussianNoise(img)

plt.figure("高斯噪声")
plt.subplot(121)
plt.imshow(img), plt.axis('off'), plt.title('origin')
plt.subplot(122)
plt.imshow(noise_guass), plt.axis('off'), plt.title('Gaussian noise')
plt.show()
'''''

'''''
noise_guass = GaussianNoise(img)
noise_salt = SaltAndPepperNoise(img, 0.05)
blur_origin = cv2.blur(img, (5, 5))
blur_gauss = cv2.blur(noise_guass, (5, 5))
blur_salt = cv2.blur(noise_salt, (5, 5))
plt.figure("均值滤波")
plt.subplot(221)
plt.imshow(noise_salt), plt.axis('off'), plt.title(u"椒盐噪声")
plt.subplot(222)
plt.imshow(noise_guass), plt.axis('off'), plt.title(u"高斯噪声")
plt.subplot(223)
plt.imshow(blur_salt), plt.axis('off'), plt.title(u"均值滤波去除椒盐噪声")
plt.subplot(224)
plt.imshow(blur_gauss), plt.axis('off'), plt.title(u"均值滤波去除高斯噪声")
plt.show()
'''''

'''''
noise = np.random.normal(0, 0.01 ** 0.5, img.shape)
noise = noise * 255
noise = np.clip(noise, 0.0, 255.0)
noise_guas = GaussianNoise(img)

plt.figure("高斯噪声")
plt.subplot(131)
plt.imshow(img), plt.axis('off'), plt.title('Src')
plt.subplot(132)
plt.imshow(noise), plt.axis('off'), plt.title('Noise')
plt.subplot(133)
plt.imshow(noise_guas), plt.axis('off'), plt.title('Dst')
plt.show()
'''''

'''''
noise_guass = GaussianNoise(img)
noise_salt = SaltAndPepperNoise(img, 0.05)
dst_1 = cv2.GaussianBlur(noise_guass, (7, 7), 0, 0)
dst_2 = cv2.GaussianBlur(noise_salt, (7,7), 0, 0)
plt.figure("高斯滤波")
plt.subplot(221)
plt.imshow(noise_guass), plt.axis('off'), plt.title(u"高斯噪声")
plt.subplot(222)
plt.imshow(noise_salt), plt.axis('off'), plt.title(u"椒盐噪声")
plt.subplot(223)
plt.imshow(dst_1), plt.axis('off'), plt.title(u"高斯滤波去除高斯噪声")
plt.subplot(224)
plt.imshow(dst_2), plt.axis('off'), plt.title(u"高斯滤波去除椒盐噪声")
plt.show()
'''''

'''''
noise_guass = GaussianNoise(img)
noise_salt = SaltAndPepperNoise(img, 0.05)
blur_origin = cv2.medianBlur(img, 3)
blur_gauss = cv2.medianBlur(noise_guass, 3)
blur_salt = cv2.medianBlur(noise_salt, 3)
plt.figure("中值滤波")
plt.subplot(221)
plt.imshow(noise_salt), plt.axis('off'), plt.title(u"椒盐噪声")
plt.subplot(222)
plt.imshow(noise_guass), plt.axis('off'), plt.title(u"高斯噪声")
plt.subplot(223)
plt.imshow(blur_salt), plt.axis('off'), plt.title(u"中值滤波去除椒盐噪声")
plt.subplot(224)
plt.imshow(blur_gauss), plt.axis('off'), plt.title(u"中值滤波去除高斯噪声")
plt.show()
'''''

'''''
kernel_1 = np.ones((5, 5), dtype=np.uint8)
kernel_2 = np.ones((7, 7), dtype=np.uint8)
erosion_1 = cv2.dilate(img, kernel_1)
erosion_2 = cv2.dilate(img, kernel_2)
plt.subplot(131)
plt.imshow(img), plt.axis('off'), plt.title(u"原图像")
plt.subplot(132)
plt.imshow(erosion_1), plt.axis('off'), plt.title(u"结构元5x5")
plt.subplot(133)
plt.imshow(erosion_2), plt.axis('off'), plt.title(u"结构元7x7")
plt.show()
'''''

'''''
hsv_1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
v_1 = hsv_1[:, :, 2]
light, v_2 = AdaptiveEquColor(img)

plt.subplot(221)
plt.imshow(img), plt.axis('off'), plt.title(u"原图像")
plt.subplot(222)
plt.imshow(light), plt.axis('off'), plt.title(u"局部直方图均衡")
plt.subplot(223)
plt.hist(v_1.ravel(), 256), plt.title(u"原直方图")
plt.subplot(224)
plt.hist(v_2.ravel(), 256), plt.title(u"均衡后直方图")
plt.show()
'''''

'''''
dst_1 = GammaTransform(img, 0.5)
dst_2 = GammaTransform(img, 1.5)
dst_3 = GammaTransform(img, 1)

plt.subplot(221)
plt.imshow(img), plt.axis('off'), plt.title(u"原图像")
plt.subplot(222)
plt.imshow(dst_1), plt.axis('off'), plt.title(u"gamma=0.5")
plt.subplot(223)
plt.imshow(dst_2), plt.axis('off'), plt.title(u"gamma=1.5")
plt.subplot(224)
plt.imshow(dst_3), plt.axis('off'), plt.title(u"gamma=1")
plt.show()
'''''

'''''
dst_1 = MSRCR(img)
plt.subplot(121)
plt.imshow(img), plt.axis('off'), plt.title(u"原图像")
plt.subplot(122)
plt.imshow(dst_1), plt.axis('off'), plt.title(u"SSR")
plt.show()
'''''




